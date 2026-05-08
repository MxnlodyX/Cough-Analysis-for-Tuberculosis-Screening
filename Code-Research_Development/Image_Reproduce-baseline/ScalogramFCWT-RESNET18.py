"""
Implementation of Scalogram FCWT with ResNet18 model
for tuberculosis cough classification.
This implementation is adapted from the architecture described in:
https://zenodo.org/records/10431329

Notes:
    - Some hyperparameters were modified to optimize training speed on local hardware.
"""

import gc
import os
import random
import tempfile
import time
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    f1_score,
    recall_score,
)

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
@dataclass
class ExperimentConfig:
    project_root: str = "/sp2026tb"
    coda_dir_name: str = "coda"
    csv_dir_name: str = "csv"

    coda_csv_name: str = "combine_scalogram.csv"

    si_tb_rel: str = "SI-scalogram/Cough_sounds_patients_with_ptb"
    si_non_tb_rel: str = "SI-scalogram/Cough_sounds_healthy_individuals"

    coda_file_type: str = "Longitudnal"
    require_coda_status_ok: bool = True
    require_coda_file_exists: bool = True

    n_folds: int = 5
    seed: int = 42

    image_size_hw: Tuple[int, int] = (224, 448)
    force_resize: bool = False
    batch_size: int = 256

    # Match tb_spectrogram_reproduce_baseline_TBscreen.ipynb
    epochs: int = 50
    weight_decay: float = 0.0
    lr_backbone: float = 1e-6
    lr_head: float = 1e-5
    step_size: int = 20
    gamma: float = 0.1
    early_stop_patience: int = 10
    early_stop_min_delta: float = 0.0005

    # Throughput and visibility tuning (balanced for 20 CPU cores + RTX 3090 Ti)
    num_workers: int = 4
    persistent_workers: bool = True
    prefetch_factor: int = 4
    batch_log_every: int = 50
    normalization_mode: str = "minmax"  # "minmax" is faster, "percentile" is more robust

    # GPU acceleration
    use_amp: bool = True
    amp_dtype: str = "float16"  # "float16" or "bfloat16"
    use_channels_last: bool = True
    enable_cudnn_benchmark: bool = True
    allow_tf32: bool = True
    use_torch_compile: bool = False  # Disabled by default; can cause OOM or crash on Windows

    # Keep CPU headroom to avoid pegging 100%
    cpu_thread_fraction: float = 0.15

    threshold: float = 0.5
    use_pretrained: bool = True
    freeze_backbone: bool = False

    output_rel_dir: str = "output_model_a_py"
    run_name: Optional[str] = None
    save_fold_models: bool = True
    start_fold: int = 1
    resume_from_existing_run: bool = True
    skip_completed_folds: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ResNet18 FCWT training with fold resume controls",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--run-all-folds",
        action="store_true",
        help="Run all folds from 1..n_folds and do not skip completed folds",
    )
    mode_group.add_argument(
        "--start-fold",
        type=int,
        default=None,
        help="Start from this fold index (e.g., 5 to run fold 5 onward)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Existing run folder name under output_rel_dir to resume from",
    )

    args, unknown = parser.parse_known_args()
    if unknown:
        print("Ignoring unknown CLI args:", unknown)
    return args


def apply_cli_overrides(config: ExperimentConfig, args: argparse.Namespace) -> None:
    if args.run_name:
        config.run_name = args.run_name

    if args.run_all_folds:
        config.start_fold = 1
        config.resume_from_existing_run = False
        config.skip_completed_folds = False
        return

    if args.start_fold is not None:
        config.start_fold = max(1, int(args.start_fold))
        config.resume_from_existing_run = True
        config.skip_completed_folds = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_project_root(config: ExperimentConfig) -> Path:
    candidates = [
        Path(config.project_root),
        Path.cwd(),
        Path.cwd().parent,
        Path.cwd().parent.parent,
        Path.cwd().parent.parent.parent,
    ]
    for root in candidates:
        if (root / config.coda_dir_name).exists() and (root / config.csv_dir_name).exists():
            return root.resolve()
    raise FileNotFoundError("Could not locate project root containing both coda and csv directories")


def configure_runtime(config: ExperimentConfig) -> int:
    cpu_total = os.cpu_count() or 1
    main_threads = max(1, min(cpu_total - 1, int(cpu_total * config.cpu_thread_fraction)))

    os.environ["OMP_NUM_THREADS"] = str(main_threads)
    os.environ["MKL_NUM_THREADS"] = str(main_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(main_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(main_threads)
    torch.set_num_threads(main_threads)

    torch.set_float32_matmul_precision("high")

    if torch.cuda.is_available():
        if config.enable_cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
        if config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    return main_threads


cli_args = parse_cli_args()
config = ExperimentConfig()
apply_cli_overrides(config, cli_args)
set_seed(config.seed)
PROJECT_ROOT = find_project_root(config)
CSV_DIR = PROJECT_ROOT / config.csv_dir_name
BASE_DIR_CODA = PROJECT_ROOT
COMBINE_CSV = CSV_DIR / config.coda_csv_name
SI_TB_PATH = PROJECT_ROOT / config.si_tb_rel
SI_NON_TB_PATH = PROJECT_ROOT / config.si_non_tb_rel
MAIN_THREADS = configure_runtime(config)

run_tag = config.run_name or datetime.now().strftime("resnet18_fcwt_%Y%m%d_%H%M%S")
OUTPUT_DIR = (PROJECT_ROOT / config.output_rel_dir / run_tag).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("COMBINE_CSV:", COMBINE_CSV)
print("SI TB PATH:", SI_TB_PATH)
print("SI NON-TB PATH:", SI_NON_TB_PATH)
print("MAIN_THREADS:", MAIN_THREADS)
print("AMP:", config.use_amp, "AMP dtype:", config.amp_dtype)
print("channels_last:", config.use_channels_last)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("start_fold:", config.start_fold)
print("resume_from_existing_run:", config.resume_from_existing_run)
print("skip_completed_folds:", config.skip_completed_folds)
print("run_all_folds_mode:", bool(cli_args.run_all_folds))
if config.resume_from_existing_run and config.run_name is None:
    print("NOTE: run_name is None, so a new run directory is created and resume will not reuse old folds")


def print_table(df: pd.DataFrame, max_rows: int = 20) -> None:
    if df.empty:
        print("(empty)")
        return
    print(df.head(max_rows).to_string(index=False))


def format_seconds(seconds: float) -> str:
    s = max(0, int(seconds))
    d, rem = divmod(s, 86400)
    h, rem = divmod(rem, 3600)
    m, sec = divmod(rem, 60)
    if d > 0:
        return f"{d}d {h:02d}h {m:02d}m"
    if h > 0:
        return f"{h}h {m:02d}m {sec:02d}s"
    return f"{m}m {sec:02d}s"
def scan_data_files(directory_path: Path, extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp")) -> List[str]:
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    paths: List[str] = []
    with os.scandir(str(directory_path)) as entries:
        for e in entries:
            if e.is_file() and Path(e.name).suffix.lower() in extensions:
                paths.append(e.path)
    return sorted(paths)


def build_si_df(tb_files: List[str], non_tb_files: List[str]) -> pd.DataFrame:
    rows = []
    for f in tb_files:
        sid = Path(f).name.split("_")[0]
        rows.append((str(f), sid, 1))
    for f in non_tb_files:
        sid = Path(f).name.split("_")[0]
        rows.append((str(f), sid, 0))

    df = pd.DataFrame(rows, columns=["file_path", "subject_id", "label"])
    df["subject_id"] = df["subject_id"].astype("category")
    df["label"] = df["label"].astype("int8")
    return df.reset_index(drop=True)


def load_coda_longitudinal_df() -> pd.DataFrame:
    if not COMBINE_CSV.exists():
        raise FileNotFoundError(f"Missing CSV: {COMBINE_CSV}")

    df_coda = pd.read_csv(COMBINE_CSV)

    required_cols = {"participant", "tb_status", "file_type", "scalogram_pathfile"}
    missing = required_cols - set(df_coda.columns)
    if missing:
        raise KeyError(f"Missing required columns in combine_scalogram.csv: {sorted(missing)}")

    df_coda = df_coda[df_coda["file_type"] == config.coda_file_type].reset_index(drop=True)
    df_coda = df_coda[df_coda["scalogram_pathfile"].notna()].reset_index(drop=True)

    if config.require_coda_status_ok and "status" in df_coda.columns:
        df_coda = df_coda[df_coda["status"].astype(str).str.lower() == "ok"].reset_index(drop=True)

    if config.require_coda_file_exists and "file_exists" in df_coda.columns:
        mask_exists = df_coda["file_exists"].astype(str).str.lower().isin(["true", "1"])
        df_coda = df_coda[mask_exists].reset_index(drop=True)

    df_coda["tb_status"] = df_coda["tb_status"].astype("int8")
    df_coda["participant"] = df_coda["participant"].astype("category")
    return df_coda


def load_si_df() -> pd.DataFrame:
    tb_files = scan_data_files(SI_TB_PATH)
    non_tb_files = scan_data_files(SI_NON_TB_PATH)
    return build_si_df(tb_files, non_tb_files)
df_coda = load_coda_longitudinal_df()
df_si = load_si_df()

print("CODA Longitudinal clips (image-ready):", len(df_coda))
print("CODA participants:", df_coda["participant"].nunique())
print("CODA TB+ clips:", int((df_coda["tb_status"] == 1).sum()))
print("CODA TB- clips:", int((df_coda["tb_status"] == 0).sum()))
print("Sample CODA scalogram path:", df_coda["scalogram_pathfile"].iloc[0])

print("SI clips:", len(df_si))
print("SI subjects:", df_si["subject_id"].nunique())
print("SI TB+ clips:", int((df_si["label"] == 1).sum()))
print("SI TB- clips:", int((df_si["label"] == 0).sum()))
def subject_label_table(df: pd.DataFrame, subj_col: str, label_col: str) -> pd.DataFrame:
    def mode_int(x: pd.Series) -> int:
        return int(x.value_counts().idxmax())

    return (
        df.groupby(subj_col)[label_col]
        .apply(mode_int)
        .reset_index()
        .rename(columns={label_col: "subject_label"})
        .sort_values(subj_col)
        .reset_index(drop=True)
    )


def build_coda_5block_si_fulltest_folds(
    df_coda: pd.DataFrame,
    df_si: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
    coda_subj_col: str = "participant",
    coda_label_col: str = "tb_status",
    si_subj_col: str = "subject_id",
    si_label_col: str = "label",
) -> List[dict]:
    st_coda = subject_label_table(df_coda, coda_subj_col, coda_label_col)
    x_coda = st_coda[coda_subj_col].values
    y_coda = st_coda["subject_label"].values

    assert df_coda.groupby(coda_subj_col)[coda_label_col].nunique().max() == 1, (
        "CODA participant has mixed labels"
    )

    st_si = subject_label_table(df_si, si_subj_col, si_label_col)
    x_si = st_si[si_subj_col].values

    assert df_si.groupby(si_subj_col)[si_label_col].nunique().max() == 1, (
        "SI subject has mixed labels"
    )

    all_si_subj = sorted(x_si.tolist())
    df_si_full = df_si.reset_index(drop=True)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds: List[dict] = []
    val_seen: List = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_coda, y_coda), start=1):
        tr_subj = x_coda[tr_idx]
        val_subj = x_coda[val_idx]

        assert set(tr_subj).isdisjoint(val_subj), f"Fold {fold}: train/val overlap"

        folds.append(
            {
                "fold": fold,
                "seed": seed + fold,
                "train_subj": sorted(tr_subj.tolist()),
                "val_subj": sorted(val_subj.tolist()),
                "test_subj": all_si_subj,
                "df_train": df_coda[df_coda[coda_subj_col].isin(tr_subj)].reset_index(drop=True),
                "df_val": df_coda[df_coda[coda_subj_col].isin(val_subj)].reset_index(drop=True),
                "df_test": df_si_full,
            }
        )

        val_seen.extend(val_subj.tolist())

    val_counts = pd.Series(val_seen).value_counts()
    assert val_counts.min() == 1 and val_counts.max() == 1, (
        "Each CODA participant must appear in validation exactly once"
    )

    return folds


cv_folds = build_coda_5block_si_fulltest_folds(
    df_coda=df_coda,
    df_si=df_si,
    n_folds=config.n_folds,
    seed=config.seed,
    coda_subj_col="participant",
    coda_label_col="tb_status",
    si_subj_col="subject_id",
    si_label_col="label",
)

print(f"Built {len(cv_folds)} folds")
def print_fold_overview(cv_folds: List[dict], df_coda: pd.DataFrame, df_si: pd.DataFrame) -> None:
    total_coda_subj = df_coda["participant"].nunique()
    total_si_subj = df_si["subject_id"].nunique()

    print("=" * 90)
    print("CODA 5-BLOCK SPLIT + SI FULL TEST OVERVIEW")
    print("=" * 90)
    print("CODA participants:", total_coda_subj)
    print("SI subjects:", total_si_subj)

    for fold_info in cv_folds:
        fold = fold_info["fold"]
        n_train_subj = len(fold_info["train_subj"])
        n_val_subj = len(fold_info["val_subj"])
        n_test_subj = len(fold_info["test_subj"])
        n_train = len(fold_info["df_train"])
        n_val = len(fold_info["df_val"])
        n_test = len(fold_info["df_test"])

        print("-" * 90)
        print(
            f"Fold {fold}: train_subj={n_train_subj} val_subj={n_val_subj} test_subj={n_test_subj} "
            f"| train={n_train} val={n_val} test={n_test}"
        )

    print("=" * 90)
    print("SI test set is fixed and identical across all folds")
    print("=" * 90)


print_fold_overview(cv_folds, df_coda, df_si)
weights = ResNet18_Weights.DEFAULT

def build_tensor_transforms() -> T.Compose:
    ops: List[object] = []
    if config.force_resize:
        ops.append(T.Resize(config.image_size_hw, antialias=True))

    ops.append(T.Normalize(mean=weights.transforms().mean, std=weights.transforms().std))
    return T.Compose(ops)


train_tf = build_tensor_transforms()
eval_tf = build_tensor_transforms()


def resolve_existing_path(raw_path: str, base_dir: Optional[Path] = None) -> Path:
    p = Path(raw_path)
    if p.is_absolute() and p.exists():
        return p

    candidates: List[Path] = []
    if base_dir is not None:
        candidates.append((base_dir / raw_path).resolve())
    candidates.append((PROJECT_ROOT / raw_path).resolve())
    candidates.append((Path.cwd() / raw_path).resolve())

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(f"Could not resolve path: {raw_path}")

class ImageScalogramDataset(Dataset):
    _to_tensor = T.ToTensor()

    def __init__(self, df, path_col, label_col, base_dir=None, transform=None):
        self.df = df.reset_index(drop=True)
        self.path_col = path_col
        self.label_col = label_col
        self.base_dir = base_dir.resolve() if base_dir is not None else None
        self.transform = transform
        raw_paths = self.df[self.path_col].astype(str).tolist()
        self.labels = self.df[self.label_col].astype(int).tolist()

        # Resolve all paths once to eliminate per-sample stat() calls
        self.resolved_paths: List[str] = []
        for rp in raw_paths:
            self.resolved_paths.append(str(resolve_existing_path(rp, base_dir=self.base_dir)))

    def __len__(self): return len(self.resolved_paths)

    def __getitem__(self, idx):
        path = self.resolved_paths[idx]

        # cv2 decodes PNG ~2-3x faster than PIL
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"cv2 could not read: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # HWC uint8 -> CHW float32 [0,1]
        img = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)

        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label
def check_path_existence(
    df: pd.DataFrame,
    path_col: str,
    base_dir: Optional[Path],
    sample_n: Optional[int] = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a report of missing files for either a sample or full dataframe."""
    paths = df[path_col].dropna().astype(str)
    if sample_n is not None:
        n = min(sample_n, len(paths))
        paths = paths.sample(n=n, random_state=seed)
        scope = f"sample={n}"
    else:
        scope = f"full={len(paths)}"

    missing_rows = []
    checked = 0
    for raw in paths:
        checked += 1
        try:
            _ = resolve_existing_path(raw, base_dir=base_dir)
        except FileNotFoundError:
            missing_rows.append({"raw_path": raw})

    missing_df = pd.DataFrame(missing_rows)
    miss = len(missing_df)
    ratio = (miss / checked * 100.0) if checked > 0 else 0.0
    print(f"Path check ({scope}) -> checked={checked}, missing={miss} ({ratio:.2f}%)")
    if miss > 0:
        print_table(missing_df, max_rows=20)
    return missing_df


def save_fold_artifacts(
    fold: int,
    train_out: Dict,
    metrics: Dict[str, float],
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Path:
    fold_dir = OUTPUT_DIR / f"fold_{fold:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    if config.save_fold_models:
        torch.save(
            {
                "fold": fold,
                "best_epoch": int(train_out["best_epoch"]),
                "best_val_loss": float(train_out["best_val_loss"]),
                "config": asdict(config),
                "metrics": metrics,
                "model_state_dict": train_out["model"].state_dict(),
            },
            fold_dir / "best_model.pt",
        )

    pd.DataFrame(train_out["history"]).to_csv(fold_dir / "history.csv", index=False)
    pd.DataFrame([metrics]).to_csv(fold_dir / "metrics.csv", index=False)

    np.savez_compressed(
        fold_dir / "test_predictions.npz",
        y_true=np.asarray(y_true, dtype=np.int32),
        y_score=np.asarray(y_score, dtype=np.float32),
    )

    return fold_dir
def resnetmodel18(use_pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    w = ResNet18_Weights.DEFAULT if use_pretrained else None
    model = resnet18(weights=w)
    in_features = model.fc.in_features  # ค่านี้คือ 512

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 128),
        nn.ReLU(inplace=False),
        nn.Dropout(0.5),
        nn.Linear(128,1))
    
    # model.conv1 = nn.Conv2d(
    #     3, 64, 
    #     kernel_size=(7, 7), 
    #     stride=(2, 2), 
    #     padding=(3,3), 
    #     bias=False)

    if freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = False
    return model


class EarlyStopper:
    """Early stopping with best-state checkpoint saved to disk (not RAM)."""

    def __init__(self, patience: int, min_delta: float, save_dir: Optional[Path] = None):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val = float("inf")
        self.best_epoch = 0
        self.bad_epochs = 0
        self._has_checkpoint = False

        # Save best state to disk instead of cloning to RAM to avoid OOM
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            self._ckpt_path = str(save_dir / "_early_stop_best.pt")
        else:
            fd, path = tempfile.mkstemp(suffix=".pt")
            os.close(fd)
            self._ckpt_path = path

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        if val_loss < (self.best_val - self.min_delta):
            self.best_val = val_loss
            self.best_epoch = epoch
            torch.save(model.state_dict(), self._ckpt_path)
            self._has_checkpoint = True
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore(self, model: nn.Module, device: str) -> None:
        if self._has_checkpoint:
            state = torch.load(self._ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            del state

    def cleanup(self) -> None:
        """Remove temp checkpoint file to free disk space."""
        try:
            if os.path.exists(self._ckpt_path):
                os.remove(self._ckpt_path)
        except OSError:
            pass


def _worker_init_fn(_worker_id: int) -> None:
    # Keep each worker single-threaded to avoid CPU oversubscription.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def make_fold_loaders(fold_info: dict) -> Dict[str, DataLoader]:
    train_ds = ImageScalogramDataset(
        fold_info["df_train"],
        path_col="scalogram_pathfile",
        label_col="tb_status",
        base_dir=BASE_DIR_CODA,
        transform=train_tf,
    )
    val_ds = ImageScalogramDataset(
        fold_info["df_val"],
        path_col="scalogram_pathfile",
        label_col="tb_status",
        base_dir=BASE_DIR_CODA,
        transform=eval_tf,
    )
    test_ds = ImageScalogramDataset(
        fold_info["df_test"],
        path_col="file_path",
        label_col="label",
        base_dir=None,
        transform=eval_tf,
    )

    pin = torch.cuda.is_available()
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": pin,
    }
    if pin:
        loader_kwargs["pin_memory_device"] = "cuda"

    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(config.persistent_workers)
        loader_kwargs["prefetch_factor"] = int(config.prefetch_factor)
        loader_kwargs["worker_init_fn"] = _worker_init_fn

    return {
        "train": DataLoader(
            train_ds,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        ),
        "val": DataLoader(
            val_ds,
            shuffle=False,
            **loader_kwargs,
        ),
        "test": DataLoader(
            test_ds,
            shuffle=False,
            **loader_kwargs,
        ),
    }


def train_one_fold(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    fold_seed: int,
    fold_idx: int,
) -> Dict:
    set_seed(fold_seed)
    device = config.device
    use_cuda = torch.cuda.is_available() and str(device).startswith("cuda")
    amp_enabled = bool(config.use_amp and use_cuda)
    amp_dtype = torch.float16 if str(config.amp_dtype).lower() == "float16" else torch.bfloat16

    model = model.to(device)
    if bool(config.use_channels_last) and use_cuda:
        model = model.to(memory_format=torch.channels_last)

    criterion = nn.BCEWithLogitsLoss()

    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("fc."):
            head_params.append(p)
        else:
            backbone_params.append(p)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": config.lr_backbone})
    if head_params:
        param_groups.append({"params": head_params, "lr": config.lr_head})
    if not param_groups:
        raise RuntimeError("No trainable parameters found")

    optimizer = optim.Adam(
        param_groups,
        weight_decay=config.weight_decay,
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.step_size,
        gamma=config.gamma,
    )

    stopper = EarlyStopper(
        patience=config.early_stop_patience,
        min_delta=config.early_stop_min_delta,
        save_dir=OUTPUT_DIR / f"fold_{fold_idx:02d}",
    )

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history = {"epoch": [], "train_loss": [], "val_loss": []}

    n_train_steps = len(loaders["train"])

    n_val_steps = len(loaders["val"])
    epoch_times_sec: List[float] = []


    print(
        f"Fold {fold_idx}: train={len(loaders['train'].dataset)} val={len(loaders['val'].dataset)} "
        f"test={len(loaders['test'].dataset)} | train_steps={n_train_steps} val_steps={n_val_steps} "
        f"| amp={amp_enabled} channels_last={bool(config.use_channels_last and use_cuda)}",
        flush=True,
    )

    for epoch in range(1, config.epochs + 1):
        epoch_t0 = time.perf_counter()
        model.train()
        train_losses: List[float] = []

        prev_step_end = time.perf_counter()
        for step, (x, y) in enumerate(loaders["train"], start=1):
            step_start = time.perf_counter()
            fetch_wait = step_start - prev_step_end

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).unsqueeze(1)
            if bool(config.use_channels_last) and use_cuda:
                x = x.contiguous(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())

            step_time = time.perf_counter() - step_start
            prev_step_end = time.perf_counter()

            if step == 1 or step % config.batch_log_every == 0 or step == n_train_steps:
                print(
                    f"    [Fold {fold_idx}][Epoch {epoch:03d}] train {step}/{n_train_steps} "
                    f"loss={loss.item():.4f} fetch_wait={fetch_wait:.2f}s step_time={step_time:.2f}s",
                    flush=True,
                )

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for step, (x, y) in enumerate(loaders["val"], start=1):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True).unsqueeze(1)
                if bool(config.use_channels_last) and use_cuda:
                    x = x.contiguous(memory_format=torch.channels_last)

                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                    logits = model(x)
                    loss = criterion(logits, y)
                val_losses.append(loss.item())

                if step == 1 or step == n_val_steps:
                    print(
                        f"    [Fold {fold_idx}][Epoch {epoch:03d}] val {step}/{n_val_steps} "
                        f"loss={loss.item():.4f}",
                        flush=True,
                    )

        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        epoch_time = time.perf_counter() - epoch_t0
        epoch_times_sec.append(epoch_time)
        avg_epoch_time = float(np.mean(epoch_times_sec)) if epoch_times_sec else epoch_time
        remain_epochs = max(0, config.epochs - epoch)
        eta_fold_sec = remain_epochs * avg_epoch_time
        print(
            f"  Epoch {epoch:03d}/{config.epochs} | train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f} | epoch_time={epoch_time:.1f}s "
            f"| eta_fold~{format_seconds(eta_fold_sec)}",
            flush=True,
        )

        should_stop = stopper.step(val_loss, model, epoch)
        if should_stop:
            print("  Early stopping", flush=True)
            break

    stopper.restore(model, device)
    stopper.cleanup()

    return {
        "model": model,
        "history": history,
        "best_epoch": int(stopper.best_epoch),
        "best_val_loss": float(stopper.best_val),
    }


@torch.no_grad()
def predict_scores(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    device = config.device
    use_cuda = torch.cuda.is_available() and str(device).startswith("cuda")
    amp_enabled = bool(config.use_amp and use_cuda)
    amp_dtype = torch.float16 if str(config.amp_dtype).lower() == "float16" else torch.bfloat16

    model.eval()

    ys: List[np.ndarray] = []
    ss: List[np.ndarray] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        if bool(config.use_channels_last) and use_cuda:
            x = x.contiguous(memory_format=torch.channels_last)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            logits = model(x).squeeze(1)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        ys.append(y.numpy().astype(int))
        ss.append(probs.astype(float))

    y_true = np.concatenate(ys) if ys else np.array([], dtype=int)
    y_score = np.concatenate(ss) if ss else np.array([], dtype=float)
    return y_true, y_score
# Quick verify: safely test one training sample path from fold 1
raw_path = cv_folds[0]["df_train"]["scalogram_pathfile"].sample(1, random_state=config.seed).iloc[0]
print("raw path:", raw_path)

try:
    sample_df = cv_folds[0]["df_train"][cv_folds[0]["df_train"]["scalogram_pathfile"] == raw_path].head(1)
    sample_ds = ImageScalogramDataset(
        sample_df,
        path_col="scalogram_pathfile",
        label_col="tb_status",
        base_dir=BASE_DIR_CODA,
        transform=train_tf,
    )
    x, y = sample_ds[0]
    resolved = sample_ds.resolved_paths[0]
    print("resolved path:", resolved)
    print("is_image:", Path(resolved).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    print("tensor shape:", tuple(x.shape), "label:", int(y.item()))
except FileNotFoundError:
    print("Missing file for this sample. Run full check cell to list missing paths.")
# Quick check: ResNet18 head and key training hyperparameters
_m = resnetmodel18(use_pretrained=False, freeze_backbone=False)
print("ResNet18 head:")
print(_m.fc)
print("-")
print("epochs:", config.epochs)
print("lr_backbone:", config.lr_backbone)
print("lr_head:", config.lr_head)
print("weight_decay:", config.weight_decay)
print("step_size:", config.step_size)
print("gamma:", config.gamma)
print("early_stop_patience:", config.early_stop_patience)
print("early_stop_min_delta:", config.early_stop_min_delta)
print("threshold:", config.threshold)
# Sanity check: one sample from fold 1 train set (image scalogram)
sanity_ds = ImageScalogramDataset(
    cv_folds[0]["df_train"].head(4),
    path_col="scalogram_pathfile",
    label_col="tb_status",
    base_dir=BASE_DIR_CODA,
    transform=train_tf,
)
x0, y0 = sanity_ds[0]
print("sample tensor shape:", tuple(x0.shape))
print("sample label:", float(y0.item()))
print("sample path:", sanity_ds.resolved_paths[0])
def compute_block_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    if len(y_true) == 0:
        raise ValueError("Empty y_true")

    y_pred = (y_score >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    sen = rec
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan

    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, y_score))
    else:
        auc = np.nan

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "acc": float(acc),
        "f1": float(f1),
        "recall": float(rec),
        "sen": float(sen),
        "spec": float(spec),
        "auc": float(auc) if np.isfinite(auc) else np.nan,
    }


def upsert_fold_result(results: List[Dict], fold_result: Dict) -> None:
    fold = int(fold_result["fold"])
    for i, row in enumerate(results):
        if int(row["fold"]) == fold:
            results[i] = fold_result
            return
    results.append(fold_result)


def load_completed_fold_result(fold: int) -> Optional[Dict]:
    fold_dir = OUTPUT_DIR / f"fold_{fold:02d}"
    history_path = fold_dir / "history.csv"
    metrics_path = fold_dir / "metrics.csv"
    pred_path = fold_dir / "test_predictions.npz"

    required = [history_path, metrics_path, pred_path]
    if not all(p.exists() for p in required):
        return None

    try:
        hist_df = pd.read_csv(history_path)
        if hist_df.empty:
            return None

        metrics_df = pd.read_csv(metrics_path)
        if metrics_df.empty:
            return None

        metrics_raw = metrics_df.iloc[0].to_dict()
        metrics: Dict[str, float] = {}
        for k, v in metrics_raw.items():
            if pd.isna(v):
                metrics[k] = np.nan
            elif k in {"tn", "fp", "fn", "tp"}:
                metrics[k] = int(v)
            else:
                metrics[k] = float(v)

        with np.load(pred_path) as npz:
            y_true = np.asarray(npz["y_true"]).astype(int)
            y_score = np.asarray(npz["y_score"]).astype(float)

        if "epoch" in hist_df.columns:
            epochs = [int(x) for x in hist_df["epoch"].tolist()]
        else:
            epochs = list(range(1, len(hist_df) + 1))

        train_loss = [float(x) for x in hist_df["train_loss"].tolist()]
        val_loss = [float(x) for x in hist_df["val_loss"].tolist()]
        best_idx = int(np.argmin(val_loss)) if len(val_loss) > 0 else 0
        best_epoch = int(epochs[best_idx]) if len(epochs) > best_idx else int(best_idx + 1)
        best_val_loss = float(val_loss[best_idx]) if len(val_loss) > 0 else float("nan")

        roc_data = None
        if len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_data = {"fpr": fpr, "tpr": tpr}

        return {
            "fold": int(fold),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "history": {
                "epoch": epochs,
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            "y_true": y_true,
            "y_score": y_score,
            "metrics": metrics,
            "roc": roc_data,
        }
    except Exception as e:
        print(f"Warning: could not load fold {fold} artifacts for resume ({e})")
        return None


def collect_resumed_results(n_folds: int) -> Tuple[List[Dict], set]:
    resumed_results: List[Dict] = []
    completed_folds = set()

    if not config.resume_from_existing_run:
        return resumed_results, completed_folds

    for fold in range(1, n_folds + 1):
        loaded = load_completed_fold_result(fold)
        if loaded is not None:
            upsert_fold_result(resumed_results, loaded)
            completed_folds.add(fold)

    if completed_folds:
        print("Resume detected completed folds:", sorted(completed_folds))
    else:
        print("Resume: no completed fold artifacts found")

    return resumed_results, completed_folds


def mean_sd(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return (np.nan, np.nan)
    if len(arr) == 1:
        return (float(arr.mean()), 0.0)
    return (float(arr.mean()), float(arr.std(ddof=1)))
all_fold_results, completed_folds = collect_resumed_results(config.n_folds)
run_start_time = time.perf_counter()
fold_durations_sec: List[float] = []

for fold_info in cv_folds:
    fold_t0 = time.perf_counter()
    fold = int(fold_info["fold"])
    fold_seed = int(fold_info["seed"])

    if fold < int(config.start_fold):
        print(f"SKIP FOLD {fold}: below start_fold={config.start_fold}")
        continue

    if bool(config.skip_completed_folds) and fold in completed_folds:
        print(f"SKIP FOLD {fold}: completed artifacts already exist")
        continue

    print("=" * 100)
    print(f"START FOLD {fold}")
    print("=" * 100)

    loaders = make_fold_loaders(fold_info)
    model = resnetmodel18(
        use_pretrained=config.use_pretrained,
        freeze_backbone=config.freeze_backbone,
    )

    # JIT-compile the model for faster forward/backward (PyTorch 2.x+)
    if config.use_torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    train_out = train_one_fold(
        model=model,
        loaders=loaders,
        fold_seed=fold_seed,
        fold_idx=fold,
    )

    y_true, y_score = predict_scores(train_out["model"], loaders["test"])
    metrics = compute_block_metrics(y_true, y_score, threshold=config.threshold)

    roc_data = None
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_data = {"fpr": fpr, "tpr": tpr}

    fold_dir = save_fold_artifacts(
        fold=fold,
        train_out=train_out,
        metrics=metrics,
        y_true=y_true,
        y_score=y_score,
    )

    upsert_fold_result(
        all_fold_results,
        {
            "fold": fold,
            "best_epoch": int(train_out["best_epoch"]),
            "best_val_loss": float(train_out["best_val_loss"]),
            "history": train_out["history"],
            "y_true": y_true,
            "y_score": y_score,
            "metrics": metrics,
            "roc": roc_data,
        },
    )
    completed_folds.add(fold)

    print(
        f"FOLD {fold} | best_epoch={train_out['best_epoch']} | "
        f"AUC={metrics['auc']:.4f} | ACC={metrics['acc']:.4f} | "
        f"F1={metrics['f1']:.4f} | Recall={metrics['recall']:.4f} | Spec={metrics['spec']:.4f}"
    )
    fold_time = time.perf_counter() - fold_t0
    fold_durations_sec.append(fold_time)
    done = len({int(r["fold"]) for r in all_fold_results})
    remain = max(0, config.n_folds - done)
    avg_fold = float(np.mean(fold_durations_sec))
    eta_all_sec = remain * avg_fold
    print(
        f"Saved fold artifacts -> {fold_dir} | fold_time={format_seconds(fold_time)} "
        f"| eta_all~{format_seconds(eta_all_sec)}",
        flush=True,
    )

    # --- Memory cleanup: release model, loaders, and training state ---
    del model, loaders, train_out
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"  GPU mem after cleanup: allocated={allocated_gb:.2f}GB reserved={reserved_gb:.2f}GB", flush=True)

print("=" * 100)
print("TRAINING COMPLETE")
print("=" * 100)
print("Total elapsed:", format_seconds(time.perf_counter() - run_start_time))
block_rows = []
for r in all_fold_results:
    m = r["metrics"]
    block_rows.append(
        {
            "block": r["fold"],
            "best_epoch": r["best_epoch"],
            "best_val_loss": r["best_val_loss"],
            "threshold": config.threshold,
            "tn": m["tn"],
            "fp": m["fp"],
            "fn": m["fn"],
            "tp": m["tp"],
            "acc": m["acc"],
            "f1": m["f1"],
            "recall": m["recall"],
            "sen": m["sen"],
            "spec": m["spec"],
            "auc": m["auc"],
        }
    )

df_block_metrics = pd.DataFrame(block_rows).sort_values("block").reset_index(drop=True)
df_block_metrics.to_csv(OUTPUT_DIR / "block_metrics.csv", index=False)
print("Per-fold metrics:")
print_table(df_block_metrics, max_rows=50)
def plot_training_loss_per_block(results: List[Dict]) -> None:
    n = len(results)
    if n == 0:
        print("No results to plot")
        return

    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows), squeeze=False)

    for i, r in enumerate(sorted(results, key=lambda x: x["fold"])):
        ax = axes[i // ncols][i % ncols]
        hist = r["history"]
        epochs = hist["epoch"]
        tr = hist["train_loss"]
        va = hist["val_loss"]
        best_epoch = int(r["best_epoch"])
        best_idx = max(0, best_epoch - 1)

        ax.plot(epochs, tr, label="train_loss", linewidth=2)
        ax.plot(epochs, va, label="val_loss", linewidth=2)
        ax.axvline(best_epoch, color="red", linestyle="--", linewidth=1.5, label=f"best_epoch={best_epoch}")

        if 0 <= best_idx < len(va):
            ax.scatter([best_epoch], [va[best_idx]], color="red", s=45, zorder=5)

        ax.set_title(f"Block {r['fold']} | Best Epoch = {best_epoch}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")

    total_axes = nrows * ncols
    for j in range(n, total_axes):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("Training and Validation Loss per Block", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_loss_per_block.png", dpi=160, bbox_inches="tight")
    #plt.show()


plot_training_loss_per_block(all_fold_results)
def plot_roc_per_block_with_mean(results: List[Dict]) -> None:
    valid = [r for r in sorted(results, key=lambda x: x["fold"]) if r["roc"] is not None]
    if len(valid) == 0:
        print("No valid ROC data")
        return

    n = len(valid)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows), squeeze=False)

    aucs = []
    for i, r in enumerate(valid):
        ax = axes[i // ncols][i % ncols]
        fpr = r["roc"]["fpr"]
        tpr = r["roc"]["tpr"]
        auc = r["metrics"]["auc"]
        aucs.append(auc)

        ax.plot(fpr, tpr, linewidth=2, label=f"AUC={auc:.4f}")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"Block {r['fold']} | Best Epoch = {r['best_epoch']}")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")

    total_axes = nrows * ncols
    for j in range(n, total_axes):
        axes[j // ncols][j % ncols].axis("off")

    mean_auc, sd_auc = mean_sd(aucs)
    fig.suptitle(f"ROC per Block | Mean AUC = {mean_auc:.4f} +/- {sd_auc:.4f}", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_per_block.png", dpi=160, bbox_inches="tight")
    plt.show()

    print(f"Mean AUC (+/- SD): {mean_auc:.4f} +/- {sd_auc:.4f}")


plot_roc_per_block_with_mean(all_fold_results)
summary_rows = []
for metric in ["acc", "f1", "recall", "sen", "spec", "auc"]:
    vals = [r["metrics"][metric] for r in all_fold_results]
    m, s = mean_sd(vals)
    summary_rows.append(
        {
            "metric": metric.upper(),
            "mean": m,
            "sd": s,
            "mean_sd": f"{m:.4f} +/- {s:.4f}",
        }
    )

df_mean_summary = pd.DataFrame(summary_rows)
df_mean_summary.to_csv(OUTPUT_DIR / "summary_mean_sd.csv", index=False)
print("Mean +/- SD summary:")
print_table(df_mean_summary, max_rows=50)

with open(OUTPUT_DIR / "run_summary.txt", "w", encoding="utf-8") as f:
    f.write(f"run_tag={run_tag}\n")
    f.write(f"output_dir={OUTPUT_DIR}\n")
    f.write(f"total_elapsed={format_seconds(time.perf_counter() - run_start_time)}\n")
    f.write(f"threshold={config.threshold}\n")
    f.write("\nblock_metrics:\n")
    f.write(df_block_metrics.to_string(index=False))
    f.write("\n\nsummary_mean_sd:\n")
    f.write(df_mean_summary.to_string(index=False))

print("Threshold used for confusion and classification metrics:", config.threshold)
print("Saved run artifacts in:", OUTPUT_DIR)