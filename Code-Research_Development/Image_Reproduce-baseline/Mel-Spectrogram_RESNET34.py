import gc, hashlib, json, logging, random, time, argparse
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2, librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score
from scipy.signal import wiener

from stft_train_utils import (
    percentile_ci, compute_metric_ci_bootstrap, bootstrap_mean_ci, mean_sd,
    write_status, resolve_run_tag, fold_is_completed, upsert_fold_result,
    save_epoch_checkpoint, load_epoch_checkpoint,
    load_completed_fold_result, collect_resumed_results,
    save_fold_artifacts_full, save_roc_all_folds_plot, save_summary_full,
)

import matplotlib; matplotlib.use("Agg")


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class ExperimentConfig:
    project_root: str = "/sp2026tb"
    coda_dir_name: str = "coda"
    csv_dir_name: str = "csv"
    coda_csv_name: str = "combine.csv"
    coda_audio_col: str = "pathfile"
    coda_participant_col: str = "participant"
    coda_label_col: str = "tb_status"
    coda_file_type_col: str = "file_type"
    coda_file_type: str = "Longitudnal"
    si_tb_rel: str = "SI_DATA/Cough_sounds_patients_with_ptb"
    si_non_tb_rel: str = "SI_DATA/Cough_sounds_healthy_individuals"
    n_folds: int = 5
    seed: int = 42
    sr_target: int = 44100
    use_wiener: bool = False
    min_wav_sec: float = 0.5
    n_fft: int = 2048
    hop_length: int = 512
    win_length: int = 2048
    n_mels: int = 128
    target_frames: int = 44 # (44100 * 0.5) / 512 ≈ 44
    power: float = 2.0
    model_name: str = "tb_resnet34_swaasa"
    batch_size: int = 128
    epochs: int = 50
    lr: float = 6e-5
    weight_decay: float = 0.01
    early_stop_patience: int = 10
    early_stop_min_delta: float = 0.0005
    fixed_threshold: float = 0.5
    ci_bootstrap_iterations: int = 2000
    ci_level: float = 0.95
    ci_seed: int = 20260424
    num_workers: int = 8
    prefetch_factor: int = 4
    persistent_workers: bool = True
    use_amp: bool = True
    output_rel_dir: str = "output_cnn_stft"
    cache_rel_dir: str = "cache_stft_img"
    run_name: Optional[str] = None
    start_fold: int = 1
    resume_from_existing_run: bool = True
    skip_completed_folds: bool = True
    auto_resume_latest: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--run-all-folds", action="store_true")
    g.add_argument("--start-fold", type=int, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--force-new-run", action="store_true")
    args, _ = p.parse_known_args(); return args


def apply_overrides(cfg, args):
    if args.run_name: cfg.run_name = args.run_name
    if getattr(args, "force_new_run", False): cfg.auto_resume_latest = False
    if args.run_all_folds:
        cfg.start_fold = 1; cfg.resume_from_existing_run = False; cfg.skip_completed_folds = False
    elif args.start_fold is not None:
        cfg.start_fold = max(1, args.start_fold)
        cfg.resume_from_existing_run = True; cfg.skip_completed_folds = True


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def find_project_root(cfg):
    for root in [Path(cfg.project_root), Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]:
        if (root / cfg.coda_dir_name).exists() and (root / cfg.csv_dir_name).exists():
            return root.resolve()
    raise FileNotFoundError("Cannot find project root")


def format_seconds(s):
    s = max(0, int(s)); h, r = divmod(s, 3600); m, sc = divmod(r, 60)
    return f"{h}h {m:02d}m {sc:02d}s" if h else f"{m}m {sc:02d}s"


def configure_logger(output_dir):
    logger = logging.getLogger("cnn_stft")
    logger.setLevel(logging.INFO); logger.handlers.clear(); logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for h in [logging.StreamHandler(), logging.FileHandler(output_dir / "run.log", encoding="utf-8")]:
        h.setFormatter(fmt); logger.addHandler(h)
    return logger


def get_feature_sig(cfg):
    import hashlib, json
    payload = {"sr": cfg.sr_target, "wiener": cfg.use_wiener, "min_sec": cfg.min_wav_sec,
               "n_fft": cfg.n_fft, "hop": cfg.hop_length, "win": cfg.win_length,
               "power": cfg.power, "size": [cfg.n_mels, cfg.target_frames]} 
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


# ── Feature pipeline ──────────────────────────────────────────────────────────
def wav_to_melspec_tensor(wav_path, cfg):
    y, _ = librosa.load(wav_path, sr=cfg.sr_target, mono=True)
    
    # กำหนดความยาวให้คงที่ที่ 0.5 วินาที
    target_len = int(cfg.sr_target * cfg.min_wav_sec)
    if len(y) == 0: 
        return np.zeros((1, cfg.n_mels, cfg.target_frames), dtype=np.float32)
    elif len(y) >= target_len: 
        y = y[:target_len]
    else: 
        y = np.pad(y, (0, target_len - len(y)))
        
    y = y / (np.max(np.abs(y)) + 1e-9)

    mel_power = librosa.feature.melspectrogram(
        y=y, sr=cfg.sr_target, n_fft=cfg.n_fft, 
        hop_length=cfg.hop_length, win_length=cfg.win_length, 
        n_mels=cfg.n_mels, power=2.0
    )
    mel_db = librosa.power_to_db(mel_power, ref=np.max).astype(np.float32)

    # Pad หรือ Trim ให้เฟรมเท่ากับ target_frames พอดี
    if mel_db.shape[1] > cfg.target_frames:
        mel_db = mel_db[:, :cfg.target_frames]
    elif mel_db.shape[1] < cfg.target_frames:
        pad_width = cfg.target_frames - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="edge")

    return np.expand_dims(mel_db, axis=0) # Shape: (1, 128, 44)

def get_spec_cached(wav_path, cache_dir, sig, cfg):
    h = hashlib.sha1((sig + "||" + str(Path(wav_path).resolve())).encode()).hexdigest()
    cp = cache_dir / f"{h}.npy"
    if cp.exists(): return np.load(cp)
    img = wav_to_melspec_tensor(wav_path, cfg) # เปลี่ยนมาเรียกใช้ฟังก์ชันใหม่
    np.save(cp, img)
    return img

# ── Dataset ───────────────────────────────────────────────────────────────────
class SpectrogramDataset(Dataset):
    def __init__(self, df, path_col, label_col, cache_dir, sig, cfg, transform=None):
        self.paths = df[path_col].astype(str).tolist()
        self.labels = df[label_col].astype(int).tolist()
        self.cache_dir = cache_dir; self.sig = sig; self.cfg = cfg; self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = get_spec_cached(self.paths[idx], self.cache_dir, self.sig, self.cfg)
        t = torch.from_numpy(img)
        if self.transform: t = self.transform(t)
        return t, torch.tensor(self.labels[idx], dtype=torch.float32)


# ── Model ─────────────────────────────────────────────────────────────────────
class TB_ResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1) 
        return self.resnet(x)


# ── Data loading ──────────────────────────────────────────────────────────────
def scan_wavs(d):
    return sorted(str(p) for p in Path(d).rglob("*.wav") if p.is_file())


def build_si_df(tb, non_tb):
    rows = [(f, Path(f).name.split("_")[0], 1) for f in tb] + \
           [(f, Path(f).name.split("_")[0], 0) for f in non_tb]
    df = pd.DataFrame(rows, columns=["file_path","subject_id","label"])
    df["label"] = df["label"].astype("int8"); return df.reset_index(drop=True)


def resolve_path(raw, root):
    p = Path(raw)
    if p.is_absolute() and p.exists(): return p
    for c in [root/raw, Path.cwd()/raw]:
        if c.resolve().exists(): return c.resolve()
    raise FileNotFoundError(raw)


def load_coda_df(cfg, root, csv_dir):
    df = pd.read_csv(csv_dir/cfg.coda_csv_name)
    df = df[df[cfg.coda_file_type_col].astype(str)==cfg.coda_file_type].reset_index(drop=True)
    df = df[df[cfg.coda_audio_col].notna()].reset_index(drop=True)
    paths, kept = [], []
    for i, raw in enumerate(df[cfg.coda_audio_col].astype(str)):
        try: paths.append(str(resolve_path(raw, root))); kept.append(i)
        except FileNotFoundError: pass
    df = df.iloc[kept].reset_index(drop=True)
    df["resolved_wav_path"] = paths
    df[cfg.coda_label_col] = df[cfg.coda_label_col].astype("int8"); return df


def subject_label_table(df, subj_col, label_col):
    return (df.groupby(subj_col)[label_col]
              .apply(lambda x: int(x.value_counts().idxmax()))
              .reset_index().rename(columns={label_col: "subject_label"}))


def build_folds(df_coda, df_si, cfg):
    st = subject_label_table(df_coda, cfg.coda_participant_col, cfg.coda_label_col)
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    x, y = st[cfg.coda_participant_col].values, st["subject_label"].values
    folds = []
    for fold, (tri, vai) in enumerate(skf.split(x, y), 1):
        folds.append({"fold": fold, "seed": cfg.seed+fold,
                      "df_train": df_coda[df_coda[cfg.coda_participant_col].isin(x[tri])].reset_index(drop=True),
                      "df_val":   df_coda[df_coda[cfg.coda_participant_col].isin(x[vai])].reset_index(drop=True),
                      "df_test":  df_si.reset_index(drop=True)})
    return folds


def _fmt_ids(ids: list, max_show: int = 6) -> str:
    if len(ids) <= max_show:
        return ", ".join(str(i) for i in ids)
    shown = ", ".join(str(i) for i in ids[:max_show])
    return f"{shown}  ... (+{len(ids) - max_show} more)"


def log_fold_summary(folds, df_coda, df_si, cfg, logger):
    n_coda = df_coda[cfg.coda_participant_col].nunique()
    n_si   = df_si["subject_id"].nunique()
    W = 82

    logger.info("=" * W)
    logger.info(f"  FOLD SUMMARY  ·  {len(folds)}-Block CODA CV  ·  Longitudinal only  ·  SI full test fixed")
    logger.info(f"  CODA participants : {n_coda}  |  TB+ {(df_coda[cfg.coda_label_col]==1).sum():,} clips  |  TB- {(df_coda[cfg.coda_label_col]==0).sum():,} clips")
    logger.info(f"  SI   subjects     : {n_si}  |  TB+ {(df_si['label']==1).sum():,} clips  |  TB- {(df_si['label']==0).sum():,} clips")
    logger.info("=" * W)

    for fold_info in folds:
        fold = fold_info["fold"]
        logger.info(f"\n{'-'*W}")
        logger.info(f"  FOLD {fold}")
        logger.info(f"{'-'*W}")

        for split_name, key_df in [("TRAIN", "df_train"), ("VAL  ", "df_val")]:
            sub_df = fold_info[key_df]
            subj = sub_df[cfg.coda_participant_col].unique()
            n_subj = len(subj)
            pct    = n_subj / n_coda * 100
            pos    = int((sub_df[cfg.coda_label_col] == 1).sum())
            neg    = int((sub_df[cfg.coda_label_col] == 0).sum())

            pos_ids = sorted([s for s in subj if int(sub_df.loc[sub_df[cfg.coda_participant_col] == s, cfg.coda_label_col].iloc[0]) == 1])
            neg_ids = sorted([s for s in subj if int(sub_df.loc[sub_df[cfg.coda_participant_col] == s, cfg.coda_label_col].iloc[0]) == 0])

            logger.info(f"  {split_name}  {n_subj:>4} subjects ({pct:4.1f}%)  |  TB+ {pos:>8,} clips  |  TB- {neg:>8,} clips")
            logger.info(f"    TB+ subjects [{len(pos_ids):>3}] : {_fmt_ids(pos_ids)}")
            logger.info(f"    TB- subjects [{len(neg_ids):>3}] : {_fmt_ids(neg_ids)}")

        test_df    = fold_info["df_test"]
        test_subj  = test_df["subject_id"].unique()
        n_test     = len(test_subj)
        pct_test   = n_test / (n_si if n_si > 0 else 1) * 100
        ptb_df     = test_df[test_df["label"] == 1]
        nontb_df   = test_df[test_df["label"] == 0]
        ptb_ids    = sorted([str(s) for s in ptb_df["subject_id"].unique()])
        nontb_ids  = sorted([str(s) for s in nontb_df["subject_id"].unique()])

        logger.info(f"  TEST  {n_test:>4} subjects ({pct_test:4.1f}%)  |  TB+ {len(ptb_df):>8,} clips  |  TB- {len(nontb_df):>8,} clips")
        logger.info(f"    TB+ subjects [{len(ptb_ids):>3}] : {_fmt_ids(ptb_ids)}")
        logger.info(f"    TB- subjects [{len(nontb_ids):>3}] : {_fmt_ids(nontb_ids)}")

    logger.info(f"\n{'='*W}")
    logger.info("  NO SUBJECT LEAKAGE ✓  (CODA train and val are disjoint in every fold;")
    logger.info("  SI full test set is fixed and independent)")
    logger.info("=" * W)


def make_loader(df, path_col, label_col, cache_dir, sig, cfg, shuffle):
    # เอา transform=norm ออก
    ds = SpectrogramDataset(df, path_col, label_col, cache_dir, sig, cfg, transform=None)
    nw = cfg.num_workers
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=nw,
                      pin_memory=torch.cuda.is_available(),
                      persistent_workers=(cfg.persistent_workers and nw > 0),
                      prefetch_factor=(cfg.prefetch_factor if nw > 0 else None))


def compute_metrics(y_true, y_score, thr=0.5):
    y_pred = (y_score >= thr).astype(int)
    tn,fp,fn,tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    auc_v = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true))==2 else np.nan
    sen=tp/(tp+fn+1e-9); spec=tn/(tn+fp+1e-9)
    ppv=tp/(tp+fp+1e-9); npv=tn/(tn+fn+1e-9)
    f1=2*ppv*sen/(ppv+sen+1e-9)
    return {"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),
            "acc":float(accuracy_score(y_true,y_pred)),
            "sensitivity":float(sen),"specificity":float(spec),
            "ppv":float(ppv),"npv":float(npv),"f1":float(f1),"auc":auc_v}


# ── EarlyStopper ──────────────────────────────────────────────────────────────
class EarlyStopper:
    def __init__(self, patience, min_delta):
        self.patience=patience; self.min_delta=min_delta
        self.best_val=float("inf"); self.best_epoch=0; self.bad=0

    def step(self, val_loss, epoch):
        if val_loss < self.best_val - self.min_delta:
            self.best_val=val_loss; self.best_epoch=epoch; self.bad=0; return False
        self.bad += 1; return self.bad >= self.patience

    def state_dict(self): return {"patience":self.patience,"min_delta":self.min_delta,
                                   "bad":self.bad,"best_val":self.best_val,"best_epoch":self.best_epoch}

    def load_state_dict(self, s):
        self.patience=s.get("patience",self.patience); self.min_delta=s.get("min_delta",self.min_delta)
        self.bad=s.get("bad",0); self.best_val=s.get("best_val",float("inf")); self.best_epoch=s.get("best_epoch",0)


# ── Training ──────────────────────────────────────────────────────────────────
def train_fold(fi, cfg, cache_dir, sig, output_dir, run_tag, logger):
    fold = fi["fold"]; set_seed(fi["seed"]); device = cfg.device
    tr_dl = make_loader(fi["df_train"],"resolved_wav_path",cfg.coda_label_col,cache_dir,sig,cfg,True)
    va_dl = make_loader(fi["df_val"],  "resolved_wav_path",cfg.coda_label_col,cache_dir,sig,cfg,False)
    te_dl = make_loader(fi["df_test"], "file_path","label",cache_dir,sig,cfg,False)
    model = TB_ResNet34().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay) # ใช้ AdamW
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr, steps_per_epoch=len(tr_dl), epochs=cfg.epochs, anneal_strategy='linear')
    stopper = EarlyStopper(cfg.early_stop_patience, cfg.early_stop_min_delta)
    fold_dir = output_dir/f"fold_{fold:02d}"; fold_dir.mkdir(parents=True, exist_ok=True)
    ckpt = fold_dir/"epoch_checkpoint.pt"; best_path = fold_dir/"best_model_state.pt"
    history = {"epoch":[],"train_loss":[],"val_loss":[]}; start_epoch = 1
    amp = cfg.use_amp and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    if ckpt.exists():
        try:
            last_epoch, stopper_state, history = load_epoch_checkpoint(ckpt, model, optimizer, device)
            stopper.load_state_dict(stopper_state); start_epoch = last_epoch + 1
            logger.info("Fold %d: resumed from epoch %d", fold, last_epoch)
        except Exception as e:
            logger.warning("Fold %d: checkpoint load failed (%s)", fold, e)

    logger.info("Fold %d | train=%d val=%d test=%d start_epoch=%d",
                fold, len(tr_dl.dataset), len(va_dl.dataset), len(te_dl.dataset), start_epoch)

    log_every = 500
    for epoch in range(start_epoch, cfg.epochs+1):
        t0 = time.perf_counter()
        write_status(output_dir, run_tag, "running", fold, epoch, "training")
        model.train(); tr_l = []; n_tr = len(tr_dl)
        for step, (x, y) in enumerate(tr_dl, 1):
            x, y = x.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp): loss = criterion(model(x), y)
            scaler.scale(loss).backward(); 
            scaler.step(optimizer); 
            scaler.update()
            scheduler.step()
            tr_l.append(loss.item())
            if step == 1 or step % log_every == 0 or step == n_tr:
                logger.info("Fold %d | epoch %03d | train %d/%d | loss=%.4f | elapsed=%s",
                            fold, epoch, step, n_tr, loss.item(), format_seconds(time.perf_counter()-t0))
        model.eval(); va_l = []; n_va = len(va_dl)
        with torch.no_grad():
            for step, (x, y) in enumerate(va_dl, 1):
                x, y = x.to(device), y.to(device).unsqueeze(1)
                with torch.autocast("cuda", enabled=amp): va_l.append(criterion(model(x), y).item())
                if step == 1 or step == n_va:
                    logger.info("Fold %d | epoch %03d | val %d/%d", fold, epoch, step, n_va)
        tl, vl = float(np.mean(tr_l)), float(np.mean(va_l))
        history["epoch"].append(epoch); history["train_loss"].append(tl); history["val_loss"].append(vl)
        pd.DataFrame(history).to_csv(fold_dir/"history_running.csv", index=False)
        improved = vl < stopper.best_val - stopper.min_delta
        should_stop = stopper.step(vl, epoch)
        if improved: torch.save(model.state_dict(), best_path)
        save_epoch_checkpoint(ckpt, model, optimizer, stopper.state_dict(), epoch, history)
        logger.info("Fold %d | epoch %03d/%03d | train=%.4f val=%.4f | best=%.4f@%d | %s",
                    fold, epoch, cfg.epochs, tl, vl, stopper.best_val, stopper.best_epoch,
                    format_seconds(time.perf_counter()-t0))
        if should_stop: logger.info("Fold %d: early stop @epoch %d", fold, epoch); break

    if best_path.exists(): model.load_state_dict(torch.load(best_path, map_location=device))

    model.eval(); ys, ss = [], []
    with torch.no_grad():
        for x, y in te_dl:
            with torch.autocast("cuda", enabled=amp): logits = model(x.to(device)).squeeze(1)
            ss.append(torch.sigmoid(logits).cpu().numpy()); ys.append(y.numpy().astype(int))
    return {"history":history,"best_epoch":stopper.best_epoch,"best_val_loss":stopper.best_val,
            "y_true":np.concatenate(ys),"y_score":np.concatenate(ss)}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args(); cfg = ExperimentConfig(); apply_overrides(cfg, args); set_seed(cfg.seed)
    root = find_project_root(cfg)
    run_root = (root/cfg.output_rel_dir).resolve(); run_root.mkdir(parents=True, exist_ok=True)
    run_tag = resolve_run_tag(cfg.run_name, run_root, "cnn_stft", cfg.auto_resume_latest, cfg.resume_from_existing_run)
    out_dir = (run_root/run_tag).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (root/cfg.cache_rel_dir).resolve(); cache_dir.mkdir(parents=True, exist_ok=True)
    sig = get_feature_sig(cfg)
    logger = configure_logger(out_dir)
    logger.info("Model=CNN | device=%s | sig=%s", cfg.device, sig)
    with open(out_dir/"config.json","w") as f: json.dump(asdict(cfg), f, indent=2)
    write_status(out_dir, run_tag, "starting", message="loading data")
    df_coda = load_coda_df(cfg, root, root/cfg.csv_dir_name)
    df_si = build_si_df(scan_wavs(root/cfg.si_tb_rel), scan_wavs(root/cfg.si_non_tb_rel))
    logger.info("CODA=%d SI=%d", len(df_coda), len(df_si))
    folds = build_folds(df_coda, df_si, cfg)
    log_fold_summary(folds, df_coda, df_si, cfg, logger)
    all_results, completed = collect_resumed_results(cfg.n_folds, out_dir, cfg.resume_from_existing_run, logger)
    t0 = time.perf_counter()
    for fi in folds:
        fold = fi["fold"]
        if fold < cfg.start_fold: continue
        if cfg.skip_completed_folds and fold in completed:
            logger.info("SKIP fold %d", fold); continue
        fold_dir = out_dir/f"fold_{fold:02d}"; fold_dir.mkdir(parents=True, exist_ok=True)
        write_status(out_dir, run_tag, "running", fold, message="training")
        logger.info("="*70); logger.info("START FOLD %d", fold)
        result = train_fold(fi, cfg, cache_dir, sig, out_dir, run_tag, logger)
        write_status(out_dir, run_tag, "running", fold, message="evaluating")
        metrics = compute_metrics(result["y_true"], result["y_score"], cfg.fixed_threshold)
        metrics_ci = compute_metric_ci_bootstrap(result["y_true"], result["y_score"],
                                                  cfg.fixed_threshold, cfg.ci_bootstrap_iterations,
                                                  cfg.ci_level, cfg.ci_seed+fold)
        save_fold_artifacts_full(fold_dir, result["history"], result["best_epoch"],
                                  result["best_val_loss"], metrics, metrics_ci,
                                  result["y_true"], result["y_score"], cfg.fixed_threshold)
        upsert_fold_result(all_results, {"fold":fold,"best_epoch":result["best_epoch"],
                                          "best_val_loss":result["best_val_loss"],
                                          "history":result["history"],"y_true":result["y_true"],
                                          "y_score":result["y_score"],"metrics":metrics,"metrics_ci":metrics_ci})
        completed.add(fold)
        logger.info("FOLD %d | AUC=%.4f SEN=%.4f SPEC=%.4f F1=%.4f",
                    fold, metrics["auc"], metrics["sensitivity"], metrics["specificity"], metrics["f1"])
        del result; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        write_status(out_dir, run_tag, "running", fold, message="fold completed")

    save_summary_full(out_dir, all_results, asdict(cfg), time.perf_counter()-t0,
                      cfg.fixed_threshold, cfg.model_name, cfg.ci_bootstrap_iterations,
                      cfg.ci_level, cfg.ci_seed, format_seconds)
    save_roc_all_folds_plot(out_dir, all_results, cfg.model_name, logger)
    write_status(out_dir, run_tag, "completed", message="all folds completed")
    logger.info("Done. Total: %s | Saved: %s", format_seconds(time.perf_counter()-t0), out_dir)


if __name__ == "__main__":
    main()
