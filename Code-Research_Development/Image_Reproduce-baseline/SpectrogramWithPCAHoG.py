"""
Implementation of Spectrogram with PCA-HoG features and Capsule Network,CNN,VGG16,ResNet50 model
for tuberculosis cough classification.
This implementation is adapted from the architecture described in:
https://link.springer.com/article/10.1007/s44163-024-00179-4

Notes:
    - Some hyperparameters were modified to optimize training speed on local hardware.
"""


import argparse
import gc
import hashlib
import json
import logging
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F

from scipy.signal import wiener
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class ExperimentConfig:
	project_root: str = "/sp2026tb"
	coda_dir_name: str = "coda"
	csv_dir_name: str = "csv"

	coda_csv_name: str = "combine.csv"
	coda_cached_csv_name: str = "cached_physical_paths.csv"
	coda_audio_col: str = "pathfile"
	coda_participant_col: str = "participant"
	coda_label_col: str = "tb_status"
	coda_file_type_col: str = "file_type"
	coda_file_type: str = "Longitudnal"
	require_coda_status_ok: bool = False
	require_coda_file_exists: bool = False

	# SI test set (full and fixed for every fold)
	si_tb_rel: str = "SI_DATA/Cough_sounds_patients_with_ptb"
	si_non_tb_rel: str = "SI_DATA/Cough_sounds_healthy_individuals"

	# split style matches RESNET18_fcwt_News.py
	n_folds: int = 5
	seed: int = 42

	# audio -> STFT/HOG pipeline from notebook
	sr_target: int = 16000
	use_wiener: bool = False
	min_wav_sec: float = 1.0
	min_spec_hw: int = 32

	n_fft: int = 1024
	hop_length: int = 256
	win_length: int = 1024
	power: float = 2.0

	hog_orientations: int = 9
	hog_pixels_per_cell: Tuple[int, int] = (16, 16)
	hog_cells_per_block: Tuple[int, int] = (2, 2)
	hog_block_norm: str = "L2-Hys"

	pca_components: int = 20
	max_coughs_per_subject: int = 0

	# model/training from notebook
	model_name: str = "capsnet"  # fcnn / capsnet / cnn
	epochs: int = 100
	batch_size: int = 32
	lr: float = 1e-3
	weight_decay: float = 0.0
	early_stop_patience: int = 10
	early_stop_min_delta: float = 0.0005
	fixed_threshold: float = 0.5
	ci_bootstrap_iterations: int = 2000
	ci_level: float = 0.95
	ci_seed: int = 20260424

	num_workers: int = 0

	output_rel_dir: str = "output_capsule_audio"
	cache_rel_dir: str = "cache_hog_capsule"
	run_name: Optional[str] = None

	start_fold: int = 1
	resume_from_existing_run: bool = True
	skip_completed_folds: bool = True

	# if no run_name is given and resume is enabled, reuse latest run dir
	auto_resume_latest: bool = True

	device: str = "cuda" if torch.cuda.is_available() else "cpu"


def parse_cli_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Capsule/FCNN/CNN on CODA wavs with RESNET-style split "
			"(CODA 5-fold train/val + SI full test), with log and resume"
		)
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

	parser.add_argument("--run-name", type=str, default=None, help="Existing run folder name to resume")
	parser.add_argument("--model-name", type=str, default=None, help="fcnn | capsnet | cnn")
	parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
	parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
	parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
	parser.add_argument(
		"--force-new-run",
		action="store_true",
		help="Always create a new run folder even when resume is enabled",
	)

	args, unknown = parser.parse_known_args()
	if unknown:
		print("Ignoring unknown CLI args:", unknown)
	return args


def apply_cli_overrides(config: ExperimentConfig, args: argparse.Namespace) -> None:
	if args.run_name:
		config.run_name = args.run_name

	if args.model_name:
		config.model_name = str(args.model_name).lower()

	if args.epochs is not None:
		config.epochs = int(args.epochs)

	if args.batch_size is not None:
		config.batch_size = int(args.batch_size)

	if args.lr is not None:
		config.lr = float(args.lr)

	if args.force_new_run:
		config.auto_resume_latest = False

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


def configure_logger(output_dir: Path) -> logging.Logger:
	logger = logging.getLogger("capsule_audio")
	logger.setLevel(logging.INFO)
	logger.handlers.clear()
	logger.propagate = False

	fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

	sh = logging.StreamHandler()
	sh.setFormatter(fmt)
	logger.addHandler(sh)

	fh = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
	fh.setFormatter(fmt)
	logger.addHandler(fh)

	return logger


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


def resolve_run_tag(config: ExperimentConfig, run_root: Path) -> str:
	if config.run_name:
		return config.run_name

	if config.resume_from_existing_run and config.auto_resume_latest and run_root.exists():
		existing = sorted([p for p in run_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime)
		if existing:
			return existing[-1].name

	return datetime.now().strftime("capsule_audio_%Y%m%d_%H%M%S")


def resolve_existing_path(raw_path: str, base_dir: Optional[Path], project_root: Path) -> Path:
	p = Path(raw_path)
	if p.is_absolute() and p.exists():
		return p

	candidates: List[Path] = []
	if base_dir is not None:
		candidates.append((base_dir / raw_path).resolve())
	candidates.append((project_root / raw_path).resolve())
	candidates.append((Path.cwd() / raw_path).resolve())

	for c in candidates:
		if c.exists():
			return c

	raise FileNotFoundError(f"Could not resolve path: {raw_path}")


def scan_wavs(directory_path: Path) -> List[str]:
	if not directory_path.exists():
		raise FileNotFoundError(f"Directory not found: {directory_path}")

	files: List[str] = []
	for p in directory_path.rglob("*.wav"):
		if p.is_file():
			files.append(str(p.resolve()))
	return sorted(files)


def build_si_df(tb_files: List[str], non_tb_files: List[str]) -> pd.DataFrame:
	rows: List[Tuple[str, str, int]] = []
	for f in tb_files:
		sid = Path(f).name.split("_")[0]
		rows.append((str(f), sid, 1))
	for f in non_tb_files:
		sid = Path(f).name.split("_")[0]
		rows.append((str(f), sid, 0))

	df = pd.DataFrame(rows, columns=["file_path", "subject_id", "label"])
	df["subject_id"] = df["subject_id"].astype(str)
	df["label"] = df["label"].astype("int8")
	return df.reset_index(drop=True)


def load_si_df(config: ExperimentConfig, project_root: Path) -> pd.DataFrame:
	tb_path = project_root / config.si_tb_rel
	non_tb_path = project_root / config.si_non_tb_rel
	tb_files = scan_wavs(tb_path)
	non_tb_files = scan_wavs(non_tb_path)
	return build_si_df(tb_files, non_tb_files)


def cap_samples_per_subject(df: pd.DataFrame, max_per_subject: int, seed: int, subj_col: str) -> pd.DataFrame:
	if max_per_subject <= 0:
		return df.reset_index(drop=True)
	if subj_col not in df.columns:
		return df.reset_index(drop=True)

	# Avoid pandas groupby.apply behavior differences across versions
	# where grouping columns can be dropped from the returned frame.
	rng = np.random.default_rng(int(seed))
	tmp = df.copy()
	tmp["__sample_rand"] = rng.random(len(tmp))
	tmp = tmp.sort_values([subj_col, "__sample_rand"], kind="mergesort")
	tmp["__sample_rank"] = tmp.groupby(subj_col).cumcount()
	out = tmp[tmp["__sample_rank"] < int(max_per_subject)].copy()
	out = out.drop(columns=["__sample_rand", "__sample_rank"])
	return out.reset_index(drop=True)


def load_coda_wav_df(config: ExperimentConfig, project_root: Path, csv_dir: Path) -> pd.DataFrame:
	csv_path = csv_dir / config.coda_csv_name
	if not csv_path.exists():
		raise FileNotFoundError(f"Missing CSV: {csv_path}")

	df = pd.read_csv(csv_path)

	# Notebook-aligned CODA selection:
	# combine.csv inner-joined with cached_physical_paths.csv by filename.
	cached_csv_path = csv_dir / config.coda_cached_csv_name
	if cached_csv_path.exists() and "filename" in df.columns:
		df_cached = pd.read_csv(cached_csv_path)
		if "filename" in df_cached.columns:
			df = df.merge(df_cached[["filename"]].drop_duplicates("filename"), on="filename", how="inner")

	required = {
		config.coda_participant_col,
		config.coda_label_col,
		config.coda_file_type_col,
		config.coda_audio_col,
	}
	missing = required - set(df.columns)
	if missing:
		raise KeyError(f"Missing required columns in {csv_path.name}: {sorted(missing)}")

	df = df[df[config.coda_file_type_col].astype(str) == str(config.coda_file_type)].reset_index(drop=True)
	df = df[df[config.coda_audio_col].notna()].reset_index(drop=True)

	if config.require_coda_status_ok and "status" in df.columns:
		df = df[df["status"].astype(str).str.lower() == "ok"].reset_index(drop=True)

	if config.require_coda_file_exists and "file_exists" in df.columns:
		mask_exists = df["file_exists"].astype(str).str.lower().isin(["true", "1"])
		df = df[mask_exists].reset_index(drop=True)

	resolved_paths: List[str] = []
	kept_idx: List[int] = []
	for i, raw in enumerate(df[config.coda_audio_col].astype(str).tolist()):
		try:
			rp = resolve_existing_path(raw, base_dir=project_root, project_root=project_root)
		except FileNotFoundError:
			continue
		resolved_paths.append(str(rp))
		kept_idx.append(i)

	df = df.iloc[kept_idx].reset_index(drop=True)
	df["resolved_wav_path"] = resolved_paths
	df[config.coda_participant_col] = df[config.coda_participant_col].astype(str)
	df[config.coda_label_col] = df[config.coda_label_col].astype("int8")

	df = cap_samples_per_subject(
		df,
		max_per_subject=config.max_coughs_per_subject,
		seed=config.seed,
		subj_col=config.coda_participant_col,
	)

	if config.coda_participant_col not in df.columns:
		raise KeyError(
			f"Column '{config.coda_participant_col}' disappeared after sampling. "
			"Check pandas version behavior and input columns."
		)

	return df


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
	n_folds: int,
	seed: int,
	coda_subj_col: str,
	coda_label_col: str,
	si_subj_col: str,
	si_label_col: str,
) -> List[Dict]:
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

	folds: List[Dict] = []
	val_seen: List[str] = []

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


def _binary_label_counts(df: pd.DataFrame, label_col: str) -> Tuple[int, int]:
	pos = int((df[label_col].astype(int) == 1).sum())
	neg = int((df[label_col].astype(int) == 0).sum())
	return pos, neg


def log_fold_split_summary(cv_folds: List[Dict], config: ExperimentConfig, logger: logging.Logger) -> None:
	if len(cv_folds) == 0:
		logger.warning("Fold split summary skipped: no folds available")
		return

	line = "=" * 90
	logger.info(line)
	logger.info("FOLD SPLIT SUMMARY | scheme: CODA train/val and SI full test")
	logger.info(
		"columns: train/val subject=%s label=%s | test subject=subject_id label=label",
		config.coda_participant_col,
		config.coda_label_col,
	)
	logger.info(line)

	for fold_info in cv_folds:
		fold = int(fold_info["fold"])

		df_train = fold_info["df_train"]
		df_val = fold_info["df_val"]
		df_test = fold_info["df_test"]

		train_subj_n = int(df_train[config.coda_participant_col].nunique())
		val_subj_n = int(df_val[config.coda_participant_col].nunique())
		test_subj_n = int(df_test["subject_id"].nunique())

		train_pos, train_neg = _binary_label_counts(df_train, config.coda_label_col)
		val_pos, val_neg = _binary_label_counts(df_val, config.coda_label_col)
		test_pos, test_neg = _binary_label_counts(df_test, "label")

		overlap_n = len(set(fold_info["train_subj"]).intersection(set(fold_info["val_subj"])))

		logger.info(
			"Fold %02d | train CODA: subj=%d clips=%d TB+=%d TB-=%d | "
			"val CODA: subj=%d clips=%d TB+=%d TB-=%d | "
			"test SI(full): subj=%d clips=%d TB+=%d TB-=%d | train-val overlap=%d",
			fold,
			train_subj_n,
			len(df_train),
			train_pos,
			train_neg,
			val_subj_n,
			len(df_val),
			val_pos,
			val_neg,
			test_subj_n,
			len(df_test),
			test_pos,
			test_neg,
			overlap_n,
		)

	logger.info(line)


def get_feature_signature(config: ExperimentConfig) -> str:
	payload = {
		"sr_target": config.sr_target,
		"use_wiener": config.use_wiener,
		"min_wav_sec": config.min_wav_sec,
		"min_spec_hw": config.min_spec_hw,
		"n_fft": config.n_fft,
		"hop_length": config.hop_length,
		"win_length": config.win_length,
		"power": config.power,
		"hog_orientations": config.hog_orientations,
		"hog_pixels_per_cell": list(config.hog_pixels_per_cell),
		"hog_cells_per_block": list(config.hog_cells_per_block),
		"hog_block_norm": config.hog_block_norm,
	}
	raw = json.dumps(payload, sort_keys=True).encode("utf-8")
	return hashlib.sha1(raw).hexdigest()[:12]


def ensure_min_len(y: np.ndarray, sr: int, min_sec: float) -> np.ndarray:
	min_len = int(sr * min_sec)
	if len(y) >= min_len:
		return y[:min_len]
	return np.pad(y, (0, min_len - len(y)), mode="constant")


def pad_spec_to_min_hw(s: np.ndarray, min_hw: int) -> np.ndarray:
	h, w = s.shape
	pad_h = max(0, min_hw - h)
	pad_w = max(0, min_hw - w)
	if pad_h == 0 and pad_w == 0:
		return s
	return np.pad(s, ((0, pad_h), (0, pad_w)), mode="edge")


def audio_to_spectrogram_db(wav_path: str, config: ExperimentConfig) -> np.ndarray:
	y, _ = librosa.load(wav_path, sr=config.sr_target, mono=True)
	if y is None or len(y) == 0:
		return np.zeros((config.min_spec_hw, config.min_spec_hw), dtype=np.float32)

	if config.use_wiener:
		y = wiener(y).astype(np.float32)

	y = y / (np.max(np.abs(y)) + 1e-9)
	y = ensure_min_len(y, config.sr_target, config.min_wav_sec)

	s = np.abs(
		librosa.stft(
			y,
			n_fft=config.n_fft,
			hop_length=config.hop_length,
			win_length=config.win_length,
			center=True,
		)
	) ** config.power

	s_db = librosa.power_to_db(s, ref=np.max).astype(np.float32)
	s_db = pad_spec_to_min_hw(s_db, config.min_spec_hw)
	return s_db


def spec_to_hog_feature(s_db: np.ndarray, config: ExperimentConfig) -> np.ndarray:
	mn, mx = float(s_db.min()), float(s_db.max())
	if mx - mn < 1e-9:
		img = np.zeros_like(s_db, dtype=np.float32)
	else:
		img = ((s_db - mn) / (mx - mn)).astype(np.float32)

	img = pad_spec_to_min_hw(img, config.min_spec_hw)

	feat = hog(
		img,
		orientations=config.hog_orientations,
		pixels_per_cell=config.hog_pixels_per_cell,
		cells_per_block=config.hog_cells_per_block,
		block_norm=config.hog_block_norm,
		feature_vector=True,
		channel_axis=None,
	).astype(np.float32)

	if not np.isfinite(feat).all():
		feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
	return feat


def feature_cache_path(wav_path: str, cache_dir: Path, feature_sig: str) -> Path:
	stable = hashlib.sha1((feature_sig + "||" + str(Path(wav_path).resolve())).encode("utf-8")).hexdigest()
	return cache_dir / f"{stable}.npy"


def get_hog_feature_cached(wav_path: str, cache_dir: Path, feature_sig: str, config: ExperimentConfig) -> np.ndarray:
	cpath = feature_cache_path(wav_path, cache_dir, feature_sig)
	if cpath.exists():
		return np.load(cpath).astype(np.float32)

	s_db = audio_to_spectrogram_db(wav_path, config)
	feat = spec_to_hog_feature(s_db, config)
	np.save(cpath, feat)
	return feat


def df_to_hog_matrix(
	df_part: pd.DataFrame,
	path_col: str,
	label_col: str,
	cache_dir: Path,
	feature_sig: str,
	config: ExperimentConfig,
	logger: logging.Logger,
	log_prefix: str,
) -> Tuple[np.ndarray, np.ndarray]:
	x_list: List[np.ndarray] = []
	y_list: List[int] = []

	n = len(df_part)
	for i, row in enumerate(df_part.itertuples(index=False), start=1):
		wav_path = str(getattr(row, path_col))
		y = int(getattr(row, label_col))
		feat = get_hog_feature_cached(wav_path, cache_dir, feature_sig, config)
		x_list.append(feat)
		y_list.append(y)

		if i == 1 or i % 500 == 0 or i == n:
			logger.info("%s feature %d/%d", log_prefix, i, n)

	x = np.stack(x_list, axis=0).astype(np.float32)
	y = np.asarray(y_list, dtype=np.int64)
	return x, y


class VectorDataset(Dataset):
	def __init__(self, x: np.ndarray, y: np.ndarray):
		self.x = torch.tensor(x, dtype=torch.float32)
		self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

	def __len__(self) -> int:
		return len(self.x)

	def __getitem__(self, idx: int):
		return self.x[idx], self.y[idx]


class FCNN(nn.Module):
	def __init__(self, in_dim: int = 20, hidden: int = 64, drop: float = 0.2):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(in_dim, hidden),
			nn.ReLU(),
			nn.Dropout(drop),
			nn.Linear(hidden, hidden),
			nn.ReLU(),
			nn.Dropout(drop),
			nn.Linear(hidden, 1),
			nn.Sigmoid(),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


def squash(s: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
	squared_norm = (s ** 2).sum(dim=dim, keepdim=True)
	scale = squared_norm / (1.0 + squared_norm)
	return scale * s / (torch.sqrt(squared_norm + eps))


class CapsuleLayer(nn.Module):
	def __init__(self, in_caps: int, in_dim: int, out_caps: int, out_dim: int, routing_iters: int = 3):
		super().__init__()
		self.in_caps = in_caps
		self.out_caps = out_caps
		self.routing_iters = routing_iters
		self.w = nn.Parameter(0.01 * torch.randn(1, in_caps, out_caps, out_dim, in_dim))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		b = x.size(0)
		x = x.unsqueeze(2).unsqueeze(-1)
		w = self.w.repeat(b, 1, 1, 1, 1)
		u_hat = torch.matmul(w, x)

		b_ij = torch.zeros(b, self.in_caps, self.out_caps, 1, 1, device=x.device)
		for _ in range(self.routing_iters):
			c_ij = torch.softmax(b_ij, dim=2)
			s_j = (c_ij * u_hat).sum(dim=1)
			v_j = squash(s_j, dim=2)
			v_j_expand = v_j.unsqueeze(1)
			agreement = torch.matmul(u_hat.transpose(-2, -1), v_j_expand)
			b_ij = b_ij + agreement

		return v_j.squeeze(-1)


class CapsNetBinary(nn.Module):
	def __init__(
		self,
		in_dim: int = 20,
		primary_caps: int = 8,
		primary_dim: int = 8,
		digit_caps: int = 2,
		digit_dim: int = 16,
		routing_iters: int = 3,
	):
		super().__init__()
		self.primary_caps = primary_caps
		self.primary_dim = primary_dim
		self.to_primary = nn.Linear(in_dim, primary_caps * primary_dim)
		self.digit_layer = CapsuleLayer(primary_caps, primary_dim, digit_caps, digit_dim, routing_iters=routing_iters)
		
		self.classifier = nn.Sequential(
			nn.Linear(digit_caps * digit_dim, 128), # ชั้นที่ 1
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(128, 64),                     # ชั้นที่ 2
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(64, 1),                       # ชั้นที่ 3 (Output layer)
			nn.Sigmoid(),                           
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		p = self.to_primary(x).view(-1, self.primary_caps, self.primary_dim)
		p = squash(p, dim=-1)
		d = self.digit_layer(p)
		return self.classifier(d.reshape(d.size(0), -1))

class CNN_Paper(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1 (Conv 1-2)
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 2 (Conv 3-4)
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 3 (Conv 5-6)
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 4 (Conv 7-8)
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1, 4, 5)
        # ✅ ต้อง interpolate ให้ใหญ่พอสำหรับ 3 ชั้น MaxPool2d
        x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        x = self.features(x)
        return self.classifier(x)

class PaperVGG16_Fast(nn.Module):
	def __init__(self):
		super().__init__()
		self.vgg = models.vgg16(weights=None)
		self.vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
		self.vgg.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.vgg.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(512, 128),
			nn.ReLU(True),
			nn.Dropout(0.5),
			nn.Linear(128, 1),
			nn.Sigmoid()
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = x.view(-1, 1, 4, 5)
		x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
		return self.vgg(x)

class PaperResNet50_Fast(nn.Module):
	def __init__(self):
		super().__init__()
		self.resnet = models.resnet50(weights=None)
		self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.resnet.fc = nn.Sequential(
			nn.Linear(2048, 128),
			nn.ReLU(True),
			nn.Dropout(0.5),
			nn.Linear(128, 1),
			nn.Sigmoid()
		)
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = x.view(-1, 1, 4, 5)
		x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
		return self.resnet(x)

class VectorCNN(nn.Module):
	def __init__(self, in_dim: int = 20):
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv1d(1, 16, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv1d(16, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.AdaptiveAvgPool1d(1),
			nn.Flatten(),
			nn.Linear(32, 1),
			nn.Sigmoid(),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x.unsqueeze(1))


def build_model(config: ExperimentConfig) -> nn.Module:
	name = str(config.model_name).lower()
	if name == "capsnet":
		return CapsNetBinary(in_dim=config.pca_components)
	if name == "cnn2d":              
		return CNN_Paper()
	if name == "vgg16":
		return PaperVGG16_Fast()
	if name == "resnet50":
		return PaperResNet50_Fast()
	if name == "cnn":
		return VectorCNN(in_dim=config.pca_components)
	return FCNN(in_dim=config.pca_components)


class FoldEarlyStopper:
	def __init__(self, patience: int, min_delta: float):
		self.patience = int(patience)
		self.min_delta = float(min_delta)
		self.bad_epochs = 0
		self.best_val = float("inf")
		self.best_epoch = 0

	def step(self, val_loss: float, epoch: int) -> bool:
		if val_loss < self.best_val - self.min_delta:
			self.best_val = float(val_loss)
			self.best_epoch = int(epoch)
			self.bad_epochs = 0
		else:
			self.bad_epochs += 1
		return self.bad_epochs >= self.patience

	def state_dict(self) -> Dict:
		return {
			"patience": self.patience,
			"min_delta": self.min_delta,
			"bad_epochs": self.bad_epochs,
			"best_val": self.best_val,
			"best_epoch": self.best_epoch,
		}

	def load_state_dict(self, state: Dict) -> None:
		self.patience = int(state.get("patience", self.patience))
		self.min_delta = float(state.get("min_delta", self.min_delta))
		self.bad_epochs = int(state.get("bad_epochs", self.bad_epochs))
		self.best_val = float(state.get("best_val", self.best_val))
		self.best_epoch = int(state.get("best_epoch", self.best_epoch))


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
	y_true = y_true.astype(int)
	y_pred = y_pred.astype(int)
	tn = int(np.sum((y_true == 0) & (y_pred == 0)))
	fp = int(np.sum((y_true == 0) & (y_pred == 1)))
	tp = int(np.sum((y_true == 1) & (y_pred == 1)))
	fn = int(np.sum((y_true == 1) & (y_pred == 0)))
	return tn, fp, tp, fn


def metrics_from_scores(y_true: np.ndarray, y_score: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
	y_pred = (y_score >= float(thr)).astype(int)
	tn, fp, tp, fn = confusion_counts(y_true, y_pred)
	acc = (tp + tn) / (tp + tn + fp + fn + 1e-9)
	sensitivity = tp / (tp + fn + 1e-9)
	specificity = tn / (tn + fp + 1e-9)
	ppv = tp / (tp + fp + 1e-9)
	npv = tn / (tn + fn + 1e-9)
	f1 = 2 * ppv * sensitivity / (ppv + sensitivity + 1e-9)

	if len(np.unique(y_true.astype(int))) < 2:
		auc = np.nan
	else:
		auc = float(roc_auc_score(y_true.astype(int), y_score.astype(float)))

	return {
		"acc": float(acc),
		"sensitivity": float(sensitivity),
		"specificity": float(specificity),
		"ppv": float(ppv),
		"npv": float(npv),
		"f1": float(f1),
		"auc": float(auc) if np.isfinite(auc) else np.nan,
		# Keep aliases for compatibility with older summaries/log statements.
		"sens": float(sensitivity),
		"spec": float(specificity),
		"prec": float(ppv),
		"tn": int(tn),
		"fp": int(fp),
		"tp": int(tp),
		"fn": int(fn),
	}


def percentile_ci(values: List[float], ci_level: float) -> Tuple[float, float]:
	arr = np.asarray(values, dtype=float)
	arr = arr[np.isfinite(arr)]
	if len(arr) == 0:
		return np.nan, np.nan

	alpha = 1.0 - float(ci_level)
	low_q = 100.0 * (alpha / 2.0)
	high_q = 100.0 * (1.0 - alpha / 2.0)
	return float(np.percentile(arr, low_q)), float(np.percentile(arr, high_q))


def compute_metric_ci_bootstrap(
	y_true: np.ndarray,
	y_score: np.ndarray,
	threshold: float,
	n_boot: int,
	ci_level: float,
	seed: int,
) -> Dict:
	y_true = np.asarray(y_true).astype(int)
	y_score = np.asarray(y_score).astype(float)
	if len(y_true) == 0:
		raise ValueError("Empty y_true for CI bootstrap")

	metric_names = ["auc", "sensitivity", "specificity", "ppv", "npv", "f1"]
	base = metrics_from_scores(y_true, y_score, thr=threshold)

	rng = np.random.default_rng(int(seed))
	n = len(y_true)
	boot_values: Dict[str, List[float]] = {k: [] for k in metric_names}

	for _ in range(int(n_boot)):
		idx = rng.integers(0, n, size=n)
		yt = y_true[idx]
		ys = y_score[idx]
		m = metrics_from_scores(yt, ys, thr=threshold)
		for key in metric_names:
			v = float(m[key])
			if np.isfinite(v):
				boot_values[key].append(v)

	metrics_ci: Dict[str, Dict[str, float]] = {}
	for key in metric_names:
		ci_low, ci_high = percentile_ci(boot_values[key], ci_level)
		point = float(base[key]) if np.isfinite(base[key]) else np.nan
		metrics_ci[key] = {
			"point": point,
			"ci_low": ci_low,
			"ci_high": ci_high,
			"n_valid_boot": int(len(boot_values[key])),
		}

	return {
		"method": "bootstrap_percentile_over_predictions",
		"n_boot": int(n_boot),
		"ci_level": float(ci_level),
		"seed": int(seed),
		"metrics": metrics_ci,
	}


def bootstrap_mean_ci(values: List[float], n_boot: int, ci_level: float, seed: int) -> Tuple[float, float, float, int]:
	arr = np.asarray(values, dtype=float)
	arr = arr[np.isfinite(arr)]
	if len(arr) == 0:
		return np.nan, np.nan, np.nan, 0

	mean_point = float(np.mean(arr))
	if len(arr) == 1:
		return mean_point, mean_point, mean_point, 1

	rng = np.random.default_rng(int(seed))
	boot_means: List[float] = []
	for _ in range(int(n_boot)):
		sample = rng.choice(arr, size=len(arr), replace=True)
		boot_means.append(float(np.mean(sample)))

	ci_low, ci_high = percentile_ci(boot_means, ci_level)
	return mean_point, ci_low, ci_high, int(len(arr))


def save_json(path: Path, data: Dict) -> None:
	with open(path, "w", encoding="utf-8") as f:
		json.dump(data, f, indent=2, ensure_ascii=True)


def load_json(path: Path) -> Dict:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def write_status(
	output_dir: Path,
	run_tag: str,
	state: str,
	fold: Optional[int] = None,
	epoch: Optional[int] = None,
	message: str = "",
) -> None:
	payload = {
		"timestamp": datetime.now().isoformat(timespec="seconds"),
		"run_tag": run_tag,
		"state": state,
		"fold": fold,
		"epoch": epoch,
		"message": message,
	}
	save_json(output_dir / "run_status.json", payload)


def fold_is_completed(fold_dir: Path) -> bool:
	required = [
		fold_dir / "history.csv",
		fold_dir / "metrics.csv",
		fold_dir / "test_predictions.npz",
		fold_dir / "fold_done.json",
	]
	return all(p.exists() for p in required)


def save_epoch_checkpoint(
	ckpt_path: Path,
	model: nn.Module,
	optimizer: optim.Optimizer,
	stopper: FoldEarlyStopper,
	epoch: int,
	history: Dict[str, List[float]],
) -> None:
	torch.save(
		{
			"epoch": int(epoch),
			"model_state": model.state_dict(),
			"optimizer_state": optimizer.state_dict(),
			"stopper_state": stopper.state_dict(),
			"history": history,
		},
		ckpt_path,
	)


def load_epoch_checkpoint(
	ckpt_path: Path,
	model: nn.Module,
	optimizer: optim.Optimizer,
	stopper: FoldEarlyStopper,
	device: str,
) -> Tuple[int, Dict[str, List[float]]]:
	state = torch.load(ckpt_path, map_location=device)
	model.load_state_dict(state["model_state"])
	optimizer.load_state_dict(state["optimizer_state"])
	stopper.load_state_dict(state["stopper_state"])
	last_epoch = int(state.get("epoch", 0))
	history = state.get("history", {"epoch": [], "train_loss": [], "val_loss": []})
	return last_epoch, history


def train_one_fold(
	fold: int,
	fold_seed: int,
	x_train: np.ndarray,
	y_train: np.ndarray,
	x_val: np.ndarray,
	y_val: np.ndarray,
	config: ExperimentConfig,
	fold_dir: Path,
	logger: logging.Logger,
	run_tag: str,
	output_dir: Path,
) -> Tuple[nn.Module, Dict[str, List[float]], int, float]:
	set_seed(fold_seed)
	device = config.device
	model = build_model(config).to(device)

	train_loader = DataLoader(
		VectorDataset(x_train, y_train.astype(np.float32)),
		batch_size=config.batch_size,
		shuffle=True,
		num_workers=config.num_workers,
	)
	val_loader = DataLoader(
		VectorDataset(x_val, y_val.astype(np.float32)),
		batch_size=config.batch_size,
		shuffle=False,
		num_workers=config.num_workers,
	)

	criterion = nn.BCELoss()
	optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	stopper = FoldEarlyStopper(config.early_stop_patience, config.early_stop_min_delta)

	ckpt_path = fold_dir / "epoch_checkpoint.pt"
	best_path = fold_dir / "best_model_state.pt"
	history_running_path = fold_dir / "history_running.csv"

	start_epoch = 1
	history: Dict[str, List[float]] = {"epoch": [], "train_loss": [], "val_loss": []}

	if ckpt_path.exists():
		try:
			last_epoch, history = load_epoch_checkpoint(ckpt_path, model, optimizer, stopper, device)
			start_epoch = last_epoch + 1
			logger.info("Fold %d: resumed from epoch checkpoint at epoch %d", fold, last_epoch)
		except Exception as e:
			logger.warning("Fold %d: failed to load checkpoint (%s); start from scratch", fold, e)

	if best_path.exists() and stopper.best_epoch <= 0:
		logger.info("Fold %d: best model file found from previous run", fold)

	logger.info(
		"Fold %d: train=%d val=%d | steps train=%d val=%d | start_epoch=%d",
		fold,
		len(x_train),
		len(x_val),
		len(train_loader),
		len(val_loader),
		start_epoch,
	)

	epoch_times: List[float] = []
	for epoch in range(start_epoch, config.epochs + 1):
		epoch_t0 = time.perf_counter()
		write_status(output_dir, run_tag, state="running", fold=fold, epoch=epoch, message="training")

		model.train()
		train_losses: List[float] = []
		for xb, yb in train_loader:
			xb = xb.to(device)
			yb = yb.to(device)

			optimizer.zero_grad()
			pred = model(xb)
			loss = criterion(pred, yb)
			loss.backward()
			optimizer.step()

			train_losses.append(float(loss.item()))

		model.eval()
		val_losses: List[float] = []
		with torch.no_grad():
			for xb, yb in val_loader:
				xb = xb.to(device)
				yb = yb.to(device)
				pred = model(xb)
				loss = criterion(pred, yb)
				val_losses.append(float(loss.item()))

		tr_loss = float(np.mean(train_losses)) if train_losses else float("nan")
		va_loss = float(np.mean(val_losses)) if val_losses else float("nan")

		history["epoch"].append(int(epoch))
		history["train_loss"].append(tr_loss)
		history["val_loss"].append(va_loss)

		pd.DataFrame(history).to_csv(history_running_path, index=False)

		improved = va_loss < stopper.best_val - stopper.min_delta
		should_stop = stopper.step(va_loss, epoch)
		if improved:
			torch.save(model.state_dict(), best_path)

		save_epoch_checkpoint(ckpt_path, model, optimizer, stopper, epoch, history)

		epoch_time = time.perf_counter() - epoch_t0
		epoch_times.append(epoch_time)
		avg_epoch = float(np.mean(epoch_times))
		remain = max(0, config.epochs - epoch)
		eta_fold = remain * avg_epoch

		logger.info(
			"Fold %d | epoch %03d/%03d | train_loss=%.4f val_loss=%.4f | best=%.4f@%d | epoch_time=%s eta_fold~%s",
			fold,
			epoch,
			config.epochs,
			tr_loss,
			va_loss,
			stopper.best_val,
			stopper.best_epoch,
			format_seconds(epoch_time),
			format_seconds(eta_fold),
		)

		if should_stop:
			logger.info("Fold %d: early stopping at epoch %d", fold, epoch)
			break

	if best_path.exists():
		state = torch.load(best_path, map_location=device)
		model.load_state_dict(state)

	return model, history, int(stopper.best_epoch), float(stopper.best_val)


@torch.no_grad()
def predict_scores_vector(model: nn.Module, x: np.ndarray, batch_size: int, device: str) -> np.ndarray:
	model.eval()
	loader = DataLoader(VectorDataset(x, np.zeros(len(x), dtype=np.float32)), batch_size=batch_size, shuffle=False)

	scores: List[np.ndarray] = []
	for xb, _ in loader:
		xb = xb.to(device)
		pred = model(xb).detach().cpu().numpy().reshape(-1)
		scores.append(pred)

	return np.concatenate(scores, axis=0) if scores else np.array([], dtype=float)


def save_fold_artifacts(
	fold_dir: Path,
	config: ExperimentConfig,
	history: Dict[str, List[float]],
	best_epoch: int,
	best_val_loss: float,
	metrics: Dict[str, float],
	metrics_ci: Dict,
	y_true: np.ndarray,
	y_score: np.ndarray,
) -> None:
	pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
	pd.DataFrame([metrics]).to_csv(fold_dir / "metrics.csv", index=False)
	np.savez_compressed(
		fold_dir / "test_predictions.npz",
		y_true=np.asarray(y_true, dtype=np.int32),
		y_score=np.asarray(y_score, dtype=np.float32),
	)

	save_json(
		fold_dir / "fold_done.json",
		{
			"fold": int(fold_dir.name.split("_")[-1]),
			"best_epoch": int(best_epoch),
			"best_val_loss": float(best_val_loss),
			"threshold": float(config.fixed_threshold),
			"metrics": metrics,
			"completed_at": datetime.now().isoformat(timespec="seconds"),
		},
	)

	save_json(fold_dir / "metrics_with_ci.json", metrics_ci)


def load_completed_fold_result(fold: int, output_dir: Path) -> Optional[Dict]:
	fold_dir = output_dir / f"fold_{fold:02d}"
	if not fold_is_completed(fold_dir):
		return None

	try:
		hist_df = pd.read_csv(fold_dir / "history.csv")
		metrics_df = pd.read_csv(fold_dir / "metrics.csv")
		metrics_ci_path = fold_dir / "metrics_with_ci.json"
		metrics_ci = load_json(metrics_ci_path) if metrics_ci_path.exists() else None
		with np.load(fold_dir / "test_predictions.npz") as npz:
			y_true = np.asarray(npz["y_true"]).astype(int)
			y_score = np.asarray(npz["y_score"]).astype(float)

		if hist_df.empty or metrics_df.empty:
			return None

		metrics_raw = metrics_df.iloc[0].to_dict()
		metrics: Dict[str, float] = {}
		for k, v in metrics_raw.items():
			if pd.isna(v):
				metrics[k] = np.nan
			elif k in {"tn", "fp", "tp", "fn"}:
				metrics[k] = int(v)
			else:
				metrics[k] = float(v)

		best_epoch = int(hist_df.iloc[int(np.argmin(hist_df["val_loss"].values))]["epoch"])
		best_val_loss = float(hist_df["val_loss"].min())

		return {
			"fold": int(fold),
			"best_epoch": best_epoch,
			"best_val_loss": best_val_loss,
			"history": {
				"epoch": hist_df["epoch"].astype(int).tolist(),
				"train_loss": hist_df["train_loss"].astype(float).tolist(),
				"val_loss": hist_df["val_loss"].astype(float).tolist(),
			},
			"y_true": y_true,
			"y_score": y_score,
			"metrics": metrics,
		}
	except Exception:
		return None


def upsert_fold_result(results: List[Dict], fold_result: Dict) -> None:
	fold = int(fold_result["fold"])
	for i, row in enumerate(results):
		if int(row["fold"]) == fold:
			results[i] = fold_result
			return
	results.append(fold_result)


def collect_resumed_results(config: ExperimentConfig, output_dir: Path, logger: logging.Logger) -> Tuple[List[Dict], set]:
	resumed_results: List[Dict] = []
	completed_folds: set = set()

	if not config.resume_from_existing_run:
		return resumed_results, completed_folds

	for fold in range(1, config.n_folds + 1):
		loaded = load_completed_fold_result(fold, output_dir)
		if loaded is not None:
			upsert_fold_result(resumed_results, loaded)
			completed_folds.add(fold)

	if completed_folds:
		logger.info("Resume detected completed folds: %s", sorted(completed_folds))
	else:
		logger.info("Resume: no completed fold artifacts found")

	return resumed_results, completed_folds


def mean_sd(values: List[float]) -> Tuple[float, float]:
	arr = np.asarray(values, dtype=float)
	arr = arr[np.isfinite(arr)]
	if len(arr) == 0:
		return np.nan, np.nan
	if len(arr) == 1:
		return float(arr.mean()), 0.0
	return float(arr.mean()), float(arr.std(ddof=1))


def save_roc_all_folds_plot(output_dir: Path, results: List[Dict], model_name: str, logger: logging.Logger) -> Optional[Path]:
	if len(results) == 0:
		logger.warning("ROC summary plot skipped: no fold results available")
		return None

	mean_fpr = np.linspace(0.0, 1.0, 200)
	fold_tprs: List[np.ndarray] = []
	fold_aucs: List[float] = []

	results_sorted = sorted(results, key=lambda x: int(x["fold"]))
	colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(results_sorted))))
	fig, ax = plt.subplots(figsize=(9, 7))

	for i, r in enumerate(results_sorted):
		fold = int(r["fold"])
		y_true = np.asarray(r.get("y_true", [])).astype(int)
		y_score = np.asarray(r.get("y_score", [])).astype(float)

		if len(y_true) == 0 or len(np.unique(y_true)) < 2:
			logger.warning("Fold %d: ROC curve skipped because test labels have <2 classes", fold)
			continue

		fpr, tpr, _ = roc_curve(y_true, y_score)
		fold_auc = float(auc(fpr, tpr))
		interp_tpr = np.interp(mean_fpr, fpr, tpr)
		interp_tpr[0] = 0.0
		interp_tpr[-1] = 1.0

		fold_tprs.append(interp_tpr)
		fold_aucs.append(fold_auc)
		ax.plot(
			fpr,
			tpr,
			linestyle="--",
			linewidth=1.4,
			color=colors[i % len(colors)],
			label=f"Fold {fold} (AUC={fold_auc:.3f})",
		)

	if len(fold_tprs) == 0:
		plt.close(fig)
		logger.warning("ROC summary plot skipped: no valid folds to plot")
		return None

	mean_tpr = np.mean(fold_tprs, axis=0)
	mean_tpr[0] = 0.0
	mean_tpr[-1] = 1.0
	mean_auc = float(auc(mean_fpr, mean_tpr))
	std_auc = float(np.std(fold_aucs)) if len(fold_aucs) > 1 else 0.0

	ax.plot(
		mean_fpr,
		mean_tpr,
		color="navy",
		linewidth=2.4,
		label=f"Mean ROC (AUC={mean_auc:.3f} +/- {std_auc:.3f})",
	)
	ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.6, label="Chance")

	ax.set_xlabel("False Positive Rate (1 - Specificity)")
	ax.set_ylabel("True Positive Rate (Sensitivity)")
	ax.set_xlim([0, 1])
	ax.set_ylim([0, 1.02])
	ax.grid(True, alpha=0.3)
	ax.set_title(f"ROC Across Folds ({model_name})")
	ax.legend(loc="lower right", fontsize=9)

	plot_path = output_dir / "roc_all_folds_mean.png"
	fig.tight_layout()
	fig.savefig(plot_path, dpi=200, bbox_inches="tight")
	plt.close(fig)

	logger.info("Saved ROC summary plot: %s", plot_path)
	return plot_path


def save_summary(output_dir: Path, config: ExperimentConfig, results: List[Dict], run_elapsed_sec: float) -> None:
	rows: List[Dict] = []
	target_metrics = ["auc", "sensitivity", "specificity", "ppv", "npv", "f1"]
	fold_ci_rows: List[Dict] = []
	for r in sorted(results, key=lambda x: x["fold"]):
		m = r["metrics"]
		rows.append(
			{
				"fold": int(r["fold"]),
				"best_epoch": int(r["best_epoch"]),
				"best_val_loss": float(r["best_val_loss"]),
				"threshold": float(config.fixed_threshold),
				"tn": int(m["tn"]),
				"fp": int(m["fp"]),
				"fn": int(m["fn"]),
				"tp": int(m["tp"]),
				"acc": float(m["acc"]),
				"sensitivity": float(m["sensitivity"]),
				"specificity": float(m["specificity"]),
				"ppv": float(m["ppv"]),
				"npv": float(m["npv"]),
				"f1": float(m["f1"]),
				"auc": float(m["auc"]) if np.isfinite(m["auc"]) else np.nan,
			}
		)

		metrics_ci = r.get("metrics_ci")
		if metrics_ci is None:
			metrics_ci = compute_metric_ci_bootstrap(
				y_true=np.asarray(r["y_true"]).astype(int),
				y_score=np.asarray(r["y_score"]).astype(float),
				threshold=float(config.fixed_threshold),
				n_boot=int(config.ci_bootstrap_iterations),
				ci_level=float(config.ci_level),
				seed=int(config.ci_seed + int(r["fold"])),
			)

		for metric in target_metrics:
			ci_data = metrics_ci["metrics"][metric]
			fold_ci_rows.append(
				{
					"fold": int(r["fold"]),
					"metric": metric.upper(),
					"point": float(ci_data["point"]),
					"ci_low": float(ci_data["ci_low"]),
					"ci_high": float(ci_data["ci_high"]),
					"ci_95": f"{float(ci_data['point']):.4f} ({float(ci_data['ci_low']):.4f}, {float(ci_data['ci_high']):.4f})",
					"ci_method": str(metrics_ci.get("method", "bootstrap_percentile_over_predictions")),
					"n_boot": int(metrics_ci.get("n_boot", config.ci_bootstrap_iterations)),
					"ci_level": float(metrics_ci.get("ci_level", config.ci_level)),
				}
			)

	df_block = pd.DataFrame(rows)
	if not df_block.empty:
		df_block = df_block.sort_values("fold").reset_index(drop=True)
	df_block.to_csv(output_dir / "block_metrics.csv", index=False)

	df_fold_ci = pd.DataFrame(fold_ci_rows)
	if not df_fold_ci.empty:
		df_fold_ci = df_fold_ci.sort_values(["fold", "metric"]).reset_index(drop=True)
	df_fold_ci.to_csv(output_dir / "fold_metrics_ci.csv", index=False)

	summary_rows: List[Dict] = []
	for i, metric in enumerate(target_metrics, start=1):
		vals = [float(r["metrics"][metric]) for r in results]
		mean_point, ci_low, ci_high, n_folds_used = bootstrap_mean_ci(
			values=vals,
			n_boot=int(config.ci_bootstrap_iterations),
			ci_level=float(config.ci_level),
			seed=int(config.ci_seed + 1000 + i),
		)
		summary_rows.append(
			{
				"metric": metric.upper(),
				"mean": mean_point,
				"ci_low": ci_low,
				"ci_high": ci_high,
				"mean_95ci": f"{mean_point:.4f} ({ci_low:.4f}, {ci_high:.4f})",
				"ci_method": "bootstrap_percentile_over_folds",
				"n_folds": int(n_folds_used),
				"n_boot": int(config.ci_bootstrap_iterations),
				"ci_level": float(config.ci_level),
			}
		)
	df_summary = pd.DataFrame(summary_rows)
	df_summary.to_csv(output_dir / "summary_mean_ci.csv", index=False)

	# Keep legacy mean+SD output for compatibility with existing downstream scripts.
	legacy_rows: List[Dict] = []
	for metric in ["acc", "sensitivity", "specificity", "ppv", "npv", "f1", "auc"]:
		vals = [float(r["metrics"][metric]) for r in results]
		m, s = mean_sd(vals)
		legacy_rows.append({"metric": metric.upper(), "mean": m, "sd": s, "mean_sd": f"{m:.4f} +/- {s:.4f}"})
	pd.DataFrame(legacy_rows).to_csv(output_dir / "summary_mean_sd.csv", index=False)

	with open(output_dir / "run_summary.txt", "w", encoding="utf-8") as f:
		f.write(f"total_elapsed={format_seconds(run_elapsed_sec)}\n")
		f.write(f"threshold={config.fixed_threshold}\n")
		f.write(f"model_name={config.model_name}\n")
		f.write(
			"ci_method_per_fold=bootstrap percentile over prediction-level samples with replacement\n"
		)
		f.write("ci_method_mean=bootstrap percentile over fold-level means with replacement\n")
		f.write(f"ci_bootstrap_iterations={config.ci_bootstrap_iterations}\n")
		f.write(f"ci_level={config.ci_level}\n")
		f.write("\nblock_metrics:\n")
		f.write(df_block.to_string(index=False))
		f.write("\n\nfold_metrics_ci:\n")
		f.write(df_fold_ci.to_string(index=False))
		f.write("\n\nsummary_mean_ci:\n")
		f.write(df_summary.to_string(index=False))


def main() -> None:
	cli_args = parse_cli_args()
	config = ExperimentConfig()
	apply_cli_overrides(config, cli_args)
	set_seed(config.seed)

	project_root = find_project_root(config)
	csv_dir = project_root / config.csv_dir_name

	run_root = (project_root / config.output_rel_dir).resolve()
	run_root.mkdir(parents=True, exist_ok=True)

	run_tag = resolve_run_tag(config, run_root)
	output_dir = (run_root / run_tag).resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	cache_dir = (project_root / config.cache_rel_dir).resolve()
	cache_dir.mkdir(parents=True, exist_ok=True)
	feature_sig = get_feature_signature(config)

	logger = configure_logger(output_dir)
	logger.info("Run Pretrain Model Again with ImageNet Weights")
	logger.info("torch=%s cuda=%s", torch.__version__, torch.cuda.is_available())
	logger.info("PROJECT_ROOT=%s", project_root)
	logger.info("OUTPUT_DIR=%s", output_dir)
	logger.info("CACHE_DIR=%s", cache_dir)
	logger.info("feature_sig=%s", feature_sig)
	logger.info("device=%s model=%s", config.device, config.model_name)
	logger.info(
		"start_fold=%d resume=%s skip_completed=%s run_all_folds_mode=%s",
		config.start_fold,
		config.resume_from_existing_run,
		config.skip_completed_folds,
		bool(cli_args.run_all_folds),
	)

	save_json(output_dir / "config.json", asdict(config))
	write_status(output_dir, run_tag, state="starting", message="loading data")

	df_coda = load_coda_wav_df(config, project_root, csv_dir)
	df_si = load_si_df(config, project_root)

	logger.info(
		"CODA clips=%d participants=%d TB+=%d TB-=%d",
		len(df_coda),
		df_coda[config.coda_participant_col].nunique(),
		int((df_coda[config.coda_label_col] == 1).sum()),
		int((df_coda[config.coda_label_col] == 0).sum()),
	)
	logger.info(
		"SI clips=%d subjects=%d TB+=%d TB-=%d",
		len(df_si),
		df_si["subject_id"].nunique(),
		int((df_si["label"] == 1).sum()),
		int((df_si["label"] == 0).sum()),
	)

	cv_folds = build_coda_5block_si_fulltest_folds(
		df_coda=df_coda,
		df_si=df_si,
		n_folds=config.n_folds,
		seed=config.seed,
		coda_subj_col=config.coda_participant_col,
		coda_label_col=config.coda_label_col,
		si_subj_col="subject_id",
		si_label_col="label",
	)
	logger.info("Built %d folds", len(cv_folds))
	log_fold_split_summary(cv_folds, config, logger)

	all_fold_results, completed_folds = collect_resumed_results(config, output_dir, logger)

	run_t0 = time.perf_counter()
	fold_durations: List[float] = []

	for fold_info in cv_folds:
		fold_t0 = time.perf_counter()
		fold = int(fold_info["fold"])

		if fold < int(config.start_fold):
			logger.info("SKIP FOLD %d: below start_fold=%d", fold, config.start_fold)
			continue

		if bool(config.skip_completed_folds) and fold in completed_folds:
			logger.info("SKIP FOLD %d: completed artifacts already exist", fold)
			continue

		fold_dir = output_dir / f"fold_{fold:02d}"
		fold_dir.mkdir(parents=True, exist_ok=True)
		write_status(output_dir, run_tag, state="running", fold=fold, message="building features")

		logger.info("=" * 90)
		logger.info("START FOLD %d", fold)
		logger.info("=" * 90)

		x_tr_hog, y_tr = df_to_hog_matrix(
			fold_info["df_train"],
			path_col="resolved_wav_path",
			label_col=config.coda_label_col,
			cache_dir=cache_dir,
			feature_sig=feature_sig,
			config=config,
			logger=logger,
			log_prefix=f"Fold {fold} train",
		)
		x_va_hog, y_va = df_to_hog_matrix(
			fold_info["df_val"],
			path_col="resolved_wav_path",
			label_col=config.coda_label_col,
			cache_dir=cache_dir,
			feature_sig=feature_sig,
			config=config,
			logger=logger,
			log_prefix=f"Fold {fold} val",
		)
		x_te_hog, y_te = df_to_hog_matrix(
			fold_info["df_test"],
			path_col="file_path",
			label_col="label",
			cache_dir=cache_dir,
			feature_sig=feature_sig,
			config=config,
			logger=logger,
			log_prefix=f"Fold {fold} test",
		)

		scaler = StandardScaler()
		x_tr_scaled = scaler.fit_transform(x_tr_hog)
		x_va_scaled = scaler.transform(x_va_hog)
		x_te_scaled = scaler.transform(x_te_hog)

		pca = PCA(n_components=config.pca_components, random_state=config.seed)
		x_tr = pca.fit_transform(x_tr_scaled).astype(np.float32)
		x_va = pca.transform(x_va_scaled).astype(np.float32)
		x_te = pca.transform(x_te_scaled).astype(np.float32)

		model, history, best_epoch, best_val_loss = train_one_fold(
			fold=fold,
			fold_seed=int(fold_info["seed"]),
			x_train=x_tr,
			y_train=y_tr,
			x_val=x_va,
			y_val=y_va,
			config=config,
			fold_dir=fold_dir,
			logger=logger,
			run_tag=run_tag,
			output_dir=output_dir,
		)

		write_status(output_dir, run_tag, state="running", fold=fold, message="evaluating")
		y_score = predict_scores_vector(model, x_te, batch_size=max(128, config.batch_size), device=config.device)
		metrics = metrics_from_scores(y_te, y_score, thr=config.fixed_threshold)
		metrics_ci = compute_metric_ci_bootstrap(
			y_true=y_te,
			y_score=y_score,
			threshold=float(config.fixed_threshold),
			n_boot=int(config.ci_bootstrap_iterations),
			ci_level=float(config.ci_level),
			seed=int(config.ci_seed + fold),
		)

		save_fold_artifacts(
			fold_dir=fold_dir,
			config=config,
			history=history,
			best_epoch=best_epoch,
			best_val_loss=best_val_loss,
			metrics=metrics,
			metrics_ci=metrics_ci,
			y_true=y_te,
			y_score=y_score,
		)

		upsert_fold_result(
			all_fold_results,
			{
				"fold": fold,
				"best_epoch": best_epoch,
				"best_val_loss": best_val_loss,
				"history": history,
				"y_true": y_te,
				"y_score": y_score,
				"metrics": metrics,
				"metrics_ci": metrics_ci,
			},
		)
		completed_folds.add(fold)

		fold_time = time.perf_counter() - fold_t0
		fold_durations.append(fold_time)

		done = len({int(r["fold"]) for r in all_fold_results})
		remain = max(0, config.n_folds - done)
		avg_fold = float(np.mean(fold_durations)) if fold_durations else fold_time
		eta_all = remain * avg_fold

		logger.info(
			"FOLD %d DONE | best_epoch=%d | AUC=%.4f ACC=%.4f F1=%.4f SENS=%.4f SPEC=%.4f | fold_time=%s eta_all~%s",
			fold,
			best_epoch,
			metrics["auc"],
			metrics["acc"],
			metrics["f1"],
			metrics["sens"],
			metrics["spec"],
			format_seconds(fold_time),
			format_seconds(eta_all),
		)

		write_status(output_dir, run_tag, state="running", fold=fold, message="fold completed")

		del model, x_tr_hog, x_va_hog, x_te_hog, x_tr, x_va, x_te
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	total_elapsed = time.perf_counter() - run_t0
	save_summary(output_dir, config, all_fold_results, total_elapsed)
	save_roc_all_folds_plot(output_dir, all_fold_results, config.model_name, logger)
	write_status(output_dir, run_tag, state="completed", message="all available folds completed")

	logger.info("=" * 90)
	logger.info("TRAINING COMPLETE")
	logger.info("Total elapsed: %s", format_seconds(total_elapsed))
	logger.info("Saved artifacts in: %s", output_dir)


if __name__ == "__main__":
	main()
