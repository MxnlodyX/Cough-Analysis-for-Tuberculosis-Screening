"""
Implementation of Fusion Feature (MFCC + Mel-Spectrogram + Spectral Contrast) based Extra-Trees model
for tuberculosis cough classification.
This implementation is adapted from the architecture described in:
https://arxiv.org/pdf/2310.17675

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
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import auc as sklearn_auc, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


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
	require_coda_status_ok: bool = True
	require_coda_file_exists: bool = True

	# SI full test set
	si_tb_rel: str = "SI_DATA/Cough_sounds_patients_with_ptb"
	si_non_tb_rel: str = "SI_DATA/Cough_sounds_healthy_individuals"

	# Fold style follows Capsule_Network.py
	n_folds: int = 5
	seed: int = 42

	# Notebook-style audio and features
	sr_target: int = 44100
	clip_sec: float = 0.5
	normalize_audio: bool = True

	n_fft: int = 2048
	hop_length: int = 512
	win_length: int = 2048
	n_mels: int = 128
	n_mfcc: int = 40
	spec_contrast_bands: int = 8
	spec_contrast_fmin: float = 50.0
	target_frames: int = 40

	# Notebook-style augmentation
	aug_prob: float = 0.50
	aug_mix_prob_snr: float = 0.50
	aug_mix_prob_ir: float = 0.50
	snr_ratio_choices: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
	use_ir: bool = True
	ir_dir: str = "./impulse_responses_wav"

	# ExtraTrees from notebook
	n_estimators: int = 500
	fixed_threshold: float = 0.5

	ci_bootstrap_iterations: int = 2000
	ci_level: float = 0.95
	ci_seed: int = 20260424

	max_coughs_per_subject: int = 0  # 0 = no cap, use all data (48GB RAM available)

	output_rel_dir: str = "output_extra_tree_audio"
	cache_rel_dir: str = "cache_extra_tree_features"
	run_name: Optional[str] = None

	start_fold: int = 1
	resume_from_existing_run: bool = True
	skip_completed_folds: bool = True
	auto_resume_latest: bool = True


def parse_cli_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"ExtraTrees on CODA wavs with Capsule-style split "
			"(CODA 5-fold train/val + SI full test), plus resume and fold summaries"
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
		help="Start from this fold index",
	)

	parser.add_argument("--run-name", type=str, default=None, help="Existing run folder name to resume")
	parser.add_argument("--n-estimators", type=int, default=None, help="Override n_estimators")
	parser.add_argument("--seed", type=int, default=None, help="Override random seed")
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

	if args.n_estimators is not None:
		config.n_estimators = int(args.n_estimators)

	if args.seed is not None:
		config.seed = int(args.seed)

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
	logger = logging.getLogger("extra_tree_audio")
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

	return datetime.now().strftime("extra_tree_audio_%Y%m%d_%H%M%S")


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
	message: str = "",
) -> None:
	payload = {
		"timestamp": datetime.now().isoformat(timespec="seconds"),
		"run_tag": run_tag,
		"state": state,
		"fold": fold,
		"message": message,
	}
	save_json(output_dir / "run_status.json", payload)


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


def parse_binary_label(v) -> Optional[int]:
	if pd.isna(v):
		return None
	if isinstance(v, (int, np.integer)):
		return int(v > 0)
	if isinstance(v, (float, np.floating)):
		return int(float(v) > 0.0)

	s = str(v).strip().lower()
	pos = {"1", "true", "yes", "tb", "ptb", "positive", "tb+", "infected"}
	neg = {"0", "false", "no", "healthy", "negative", "normal", "tb-", "non-tb", "non_tb"}

	if s in pos:
		return 1
	if s in neg:
		return 0

	try:
		return int(float(s) > 0.0)
	except ValueError:
		return None


def cap_samples_per_subject(df: pd.DataFrame, max_per_subject: int, seed: int, subj_col: str) -> pd.DataFrame:
	if max_per_subject <= 0 or df.empty:
		return df.reset_index(drop=True)
	# Use index-based approach to avoid pandas 3.x groupby().apply() behavior change
	# (pandas 3.x excludes group key columns from apply() result by default)
	rng = np.random.default_rng(int(seed))
	keep_indices: List[int] = []
	for _, grp in df.groupby(subj_col, sort=False):
		n = min(len(grp), max_per_subject)
		chosen = rng.choice(len(grp), size=n, replace=False)
		keep_indices.extend(grp.index[chosen].tolist())
	return df.loc[sorted(keep_indices)].reset_index(drop=True)


def load_coda_wav_df(config: ExperimentConfig, project_root: Path, csv_dir: Path) -> pd.DataFrame:
	csv_path = csv_dir / config.coda_csv_name
	if not csv_path.exists():
		raise FileNotFoundError(f"Missing CSV: {csv_path}")

	df = pd.read_csv(csv_path)

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
	labels: List[int] = []
	kept_idx: List[int] = []
	for i, row in enumerate(df.itertuples(index=False)):
		raw = str(getattr(row, config.coda_audio_col))
		lb = parse_binary_label(getattr(row, config.coda_label_col))
		if lb is None:
			continue
		try:
			rp = resolve_existing_path(raw, base_dir=project_root, project_root=project_root)
		except FileNotFoundError:
			continue
		resolved_paths.append(str(rp))
		labels.append(int(lb))
		kept_idx.append(i)

	df = df.iloc[kept_idx].reset_index(drop=True)
	df["resolved_wav_path"] = resolved_paths
	df[config.coda_participant_col] = df[config.coda_participant_col].astype(str)
	df[config.coda_label_col] = np.asarray(labels, dtype="int8")

	df = cap_samples_per_subject(
		df,
		max_per_subject=config.max_coughs_per_subject,
		seed=config.seed,
		subj_col=config.coda_participant_col,
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


def get_feature_signature(config: ExperimentConfig) -> str:
	payload = {
		"sr_target": config.sr_target,
		"clip_sec": config.clip_sec,
		"normalize_audio": config.normalize_audio,
		"n_fft": config.n_fft,
		"hop_length": config.hop_length,
		"win_length": config.win_length,
		"n_mels": config.n_mels,
		"n_mfcc": config.n_mfcc,
		"spec_contrast_bands": config.spec_contrast_bands,
		"spec_contrast_fmin": config.spec_contrast_fmin,
		"target_frames": config.target_frames,
		"aug_prob": config.aug_prob,
		"aug_mix_prob_snr": config.aug_mix_prob_snr,
		"aug_mix_prob_ir": config.aug_mix_prob_ir,
		"snr_ratio_choices": list(config.snr_ratio_choices),
		"use_ir": config.use_ir,
		"ir_dir": config.ir_dir,
		"librosa_version": getattr(librosa, "__version__", "unknown"),
	}
	raw = json.dumps(payload, sort_keys=True).encode("utf-8")
	return hashlib.sha1(raw).hexdigest()[:12]


def ensure_fixed_len(y: np.ndarray, sr: int, sec: float) -> np.ndarray:
	target_len = int(sr * sec)
	if len(y) >= target_len:
		return y[:target_len]
	return np.pad(y, (0, target_len - len(y)), mode="constant")


def safe_norm(y: np.ndarray, normalize: bool) -> np.ndarray:
	if not normalize:
		return y.astype(np.float32)
	m = np.max(np.abs(y)) + 1e-9
	return (y / m).astype(np.float32)


def pad_or_trim_time(mat: np.ndarray, target_frames: int) -> np.ndarray:
	if mat.shape[1] == target_frames:
		return mat
	if mat.shape[1] > target_frames:
		return mat[:, :target_frames]
	pad = target_frames - mat.shape[1]
	return np.pad(mat, ((0, 0), (0, pad)), mode="edge")


def snr_ratio_mix(y: np.ndarray, rng: np.random.RandomState, ratios: Tuple[float, ...], normalize: bool) -> np.ndarray:
	r = float(rng.choice(np.asarray(ratios, dtype=float)))
	noise = rng.normal(0.0, 1.0, size=y.shape).astype(np.float32)
	y_mix = (r * y.astype(np.float32)) + noise
	return safe_norm(y_mix, normalize)


def ir_convolve(
	y: np.ndarray,
	rng: np.random.RandomState,
	ir_files: List[Path],
	sr_target: int,
	normalize: bool,
) -> np.ndarray:
	if len(ir_files) == 0:
		return y

	ir_path = str(ir_files[int(rng.randint(0, len(ir_files)))])
	ir, _ = librosa.load(ir_path, sr=sr_target, mono=True)
	if ir is None or len(ir) == 0:
		return y

	ir = ir / (np.max(np.abs(ir)) + 1e-9)
	n = len(y)
	conv = np.fft.irfft(np.fft.rfft(y, n=n) * np.fft.rfft(ir, n=n), n=n).astype(np.float32)
	return safe_norm(conv, normalize)


def maybe_augment_train_only(
	y: np.ndarray,
	rng: np.random.RandomState,
	train_mode: bool,
	config: ExperimentConfig,
	ir_files: List[Path],
) -> np.ndarray:
	if not train_mode:
		return y
	if rng.rand() >= config.aug_prob:
		return y

	p_snr = float(config.aug_mix_prob_snr)
	p_ir = float(config.aug_mix_prob_ir)
	total = p_snr + p_ir
	if total <= 0:
		return y

	p_snr = p_snr / total
	if rng.rand() < p_snr:
		return snr_ratio_mix(y, rng, config.snr_ratio_choices, config.normalize_audio)

	y2 = ir_convolve(y, rng, ir_files=ir_files, sr_target=config.sr_target, normalize=config.normalize_audio)
	if np.allclose(y2, y):
		return snr_ratio_mix(y, rng, config.snr_ratio_choices, config.normalize_audio)
	return y2


def feature_cache_path(
	wav_path: str,
	cache_dir: Path,
	feature_sig: str,
	train_mode: bool,
	seed: int,
) -> Path:
	mode = "train" if train_mode else "eval"
	stable = hashlib.sha1(
		f"{feature_sig}|{Path(wav_path).resolve()}|{mode}|{seed if train_mode else 0}".encode("utf-8")
	).hexdigest()
	return cache_dir / f"{stable}.npy"


def extract_feature_vector(
	wav_path: str,
	config: ExperimentConfig,
	train_mode: bool,
	seed: int,
	cache_dir: Path,
	feature_sig: str,
	ir_files: List[Path],
) -> np.ndarray:
	cpath = feature_cache_path(wav_path, cache_dir, feature_sig, train_mode, seed)
	if cpath.exists():
		return np.load(cpath).astype(np.float32)

	rng = np.random.RandomState(seed)

	y, _ = librosa.load(wav_path, sr=config.sr_target, mono=True)
	if y is None or len(y) == 0:
		mfcc = np.zeros((config.n_mfcc, config.target_frames), dtype=np.float32)
		mel = np.zeros((config.n_mels, config.target_frames), dtype=np.float32)
		contrast = np.zeros((config.spec_contrast_bands + 1, config.target_frames), dtype=np.float32)
	else:
		y = ensure_fixed_len(y, config.sr_target, config.clip_sec)
		y = safe_norm(y, config.normalize_audio)
		y = maybe_augment_train_only(y, rng=rng, train_mode=train_mode, config=config, ir_files=ir_files)

		mel_power = librosa.feature.melspectrogram(
			y=y,
			sr=config.sr_target,
			n_fft=config.n_fft,
			hop_length=config.hop_length,
			win_length=config.win_length,
			n_mels=config.n_mels,
			power=2.0,
		)
		mel = librosa.power_to_db(mel_power, ref=np.max).astype(np.float32)
		mel = pad_or_trim_time(mel, config.target_frames)

		mfcc = librosa.feature.mfcc(S=mel, n_mfcc=config.n_mfcc).astype(np.float32)
		mfcc = pad_or_trim_time(mfcc, config.target_frames)

		stft_power = np.abs(
			librosa.stft(
				y,
				n_fft=config.n_fft,
				hop_length=config.hop_length,
				win_length=config.win_length,
			)
		) ** 2

		nyquist = 0.5 * config.sr_target
		n_bands = int(config.spec_contrast_bands)
		fmin_cfg = float(config.spec_contrast_fmin)
		max_fmin = (nyquist * 0.99) / (2 ** n_bands)
		fmin_use = float(min(fmin_cfg, max_fmin)) if max_fmin > 0 else 1.0
		if fmin_use <= 0:
			fmin_use = 1.0

		contrast = librosa.feature.spectral_contrast(
			S=stft_power,
			sr=config.sr_target,
			n_bands=n_bands,
			fmin=fmin_use,
		).astype(np.float32)
		contrast = pad_or_trim_time(contrast, config.target_frames)

	vec = np.concatenate([
		np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
		np.mean(mel, axis=1), np.std(mel, axis=1),
		np.mean(contrast, axis=1), np.std(contrast, axis=1)
	]).astype(np.float32)

	np.save(cpath, vec)
	return vec


def df_to_feature_matrix(
	df_part: pd.DataFrame,
	path_col: str,
	label_col: str,
	cache_dir: Path,
	feature_sig: str,
	config: ExperimentConfig,
	logger: logging.Logger,
	log_prefix: str,
	train_mode: bool,
	seed_base: int,
	ir_files: List[Path],
) -> Tuple[np.ndarray, np.ndarray]:
	x_list: List[np.ndarray] = []
	y_list: List[int] = []

	n = len(df_part)
	for i, row in enumerate(df_part.itertuples(index=False), start=1):
		wav_path = str(getattr(row, path_col))
		y = int(getattr(row, label_col))
		feat = extract_feature_vector(
			wav_path=wav_path,
			config=config,
			train_mode=train_mode,
			seed=int(seed_base + i),
			cache_dir=cache_dir,
			feature_sig=feature_sig,
			ir_files=ir_files,
		)
		x_list.append(feat)
		y_list.append(y)

		if i == 1 or i % 500 == 0 or i == n:
			logger.info("%s feature %d/%d", log_prefix, i, n)

	x = np.stack(x_list, axis=0).astype(np.float32)
	y = np.asarray(y_list, dtype=np.int64)
	return x, y


def build_model(config: ExperimentConfig) -> ExtraTreesClassifier:
	return ExtraTreesClassifier(
		n_estimators=int(config.n_estimators),
		random_state=int(config.seed),
		n_jobs=-1,
		class_weight="balanced",
	)


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


def fold_is_completed(fold_dir: Path) -> bool:
	required = [
		fold_dir / "metrics.csv",
		fold_dir / "test_predictions.npz",
		fold_dir / "fold_done.json",
	]
	return all(p.exists() for p in required)


def save_fold_artifacts(
	fold_dir: Path,
	config: ExperimentConfig,
	metrics: Dict[str, float],
	metrics_ci: Dict,
	y_true: np.ndarray,
	y_score: np.ndarray,
	val_metrics: Dict[str, float],
) -> None:
	pd.DataFrame([metrics]).to_csv(fold_dir / "metrics.csv", index=False)
	pd.DataFrame([val_metrics]).to_csv(fold_dir / "val_metrics.csv", index=False)

	np.savez_compressed(
		fold_dir / "test_predictions.npz",
		y_true=np.asarray(y_true, dtype=np.int32),
		y_score=np.asarray(y_score, dtype=np.float32),
	)

	save_json(
		fold_dir / "fold_done.json",
		{
			"fold": int(fold_dir.name.split("_")[-1]),
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
		metrics_df = pd.read_csv(fold_dir / "metrics.csv")
		metrics_ci_path = fold_dir / "metrics_with_ci.json"
		metrics_ci = load_json(metrics_ci_path) if metrics_ci_path.exists() else None

		with np.load(fold_dir / "test_predictions.npz") as npz:
			y_true = np.asarray(npz["y_true"]).astype(int)
			y_score = np.asarray(npz["y_score"]).astype(float)

		if metrics_df.empty:
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

		return {
			"fold": int(fold),
			"y_true": y_true,
			"y_score": y_score,
			"metrics": metrics,
			"metrics_ci": metrics_ci,
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


def save_summary(output_dir: Path, config: ExperimentConfig, results: List[Dict], run_elapsed_sec: float) -> None:
	rows: List[Dict] = []
	target_metrics = ["auc", "sensitivity", "specificity", "ppv", "npv", "f1"]
	fold_ci_rows: List[Dict] = []

	for r in sorted(results, key=lambda x: x["fold"]):
		m = r["metrics"]
		rows.append(
			{
				"fold": int(r["fold"]),
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

	legacy_rows: List[Dict] = []
	for metric in ["acc", "sensitivity", "specificity", "ppv", "npv", "f1", "auc"]:
		arr = np.asarray([float(r["metrics"][metric]) for r in results], dtype=float)
		arr = arr[np.isfinite(arr)]
		if len(arr) == 0:
			mean_val = np.nan
			sd_val = np.nan
		elif len(arr) == 1:
			mean_val = float(arr.mean())
			sd_val = 0.0
		else:
			mean_val = float(arr.mean())
			sd_val = float(arr.std(ddof=1))
		legacy_rows.append(
			{
				"metric": metric.upper(),
				"mean": mean_val,
				"sd": sd_val,
				"mean_sd": f"{mean_val:.4f} +/- {sd_val:.4f}",
			}
		)

	pd.DataFrame(legacy_rows).to_csv(output_dir / "summary_mean_sd.csv", index=False)

	with open(output_dir / "run_summary.txt", "w", encoding="utf-8") as f:
		f.write(f"total_elapsed={format_seconds(run_elapsed_sec)}\n")
		f.write(f"threshold={config.fixed_threshold}\n")
		f.write("model_name=extratrees\n")
		f.write(f"n_estimators={config.n_estimators}\n")
		f.write("ci_method_per_fold=bootstrap percentile over prediction-level samples with replacement\n")
		f.write("ci_method_mean=bootstrap percentile over fold-level means with replacement\n")
		f.write(f"ci_bootstrap_iterations={config.ci_bootstrap_iterations}\n")
		f.write(f"ci_level={config.ci_level}\n")
		f.write("\nblock_metrics:\n")
		f.write(df_block.to_string(index=False))
		f.write("\n\nfold_metrics_ci:\n")
		f.write(df_fold_ci.to_string(index=False))
		f.write("\n\nsummary_mean_ci:\n")
		f.write(df_summary.to_string(index=False))


def save_roc_all_folds_plot(output_dir: Path, results: List[Dict], model_name: str, logger: logging.Logger) -> None:
	if not results:
		return
	mean_fpr = np.linspace(0, 1, 200)
	fold_tprs: List[np.ndarray] = []
	fold_aucs: List[float] = []
	colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(results))))
	fig, ax = plt.subplots(figsize=(9, 7))
	for i, r in enumerate(sorted(results, key=lambda x: x["fold"])):
		yt = np.asarray(r.get("y_true", [])).astype(int)
		ys = np.asarray(r.get("y_score", [])).astype(float)
		if len(yt) == 0 or len(np.unique(yt)) < 2:
			continue
		fpr, tpr, _ = roc_curve(yt, ys)
		fold_auc = float(sklearn_auc(fpr, tpr))
		interp = np.interp(mean_fpr, fpr, tpr); interp[0] = 0.0; interp[-1] = 1.0
		fold_tprs.append(interp); fold_aucs.append(fold_auc)
		ax.plot(fpr, tpr, linestyle="--", linewidth=1.4, color=colors[i % len(colors)],
				label=f"Fold {r['fold']} (AUC={fold_auc:.3f})")
	if not fold_tprs:
		plt.close(fig); return
	mean_tpr = np.mean(fold_tprs, axis=0); mean_tpr[0] = 0.0; mean_tpr[-1] = 1.0
	mean_auc = float(sklearn_auc(mean_fpr, mean_tpr))
	std_auc = float(np.std(fold_aucs)) if len(fold_aucs) > 1 else 0.0
	ax.plot(mean_fpr, mean_tpr, color="navy", linewidth=2.4,
			label=f"Mean ROC (AUC={mean_auc:.3f} +/- {std_auc:.3f})")
	ax.plot([0,1],[0,1],"k--",linewidth=1.0,alpha=0.6,label="Chance")
	ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
	ax.set_xlim([0,1]); ax.set_ylim([0,1.02]); ax.grid(True,alpha=0.3)
	ax.set_title(f"ROC Across Folds ({model_name})"); ax.legend(loc="lower right", fontsize=9)
	fig.tight_layout()
	out_path = output_dir / "roc_all_folds_mean.png"
	fig.savefig(out_path, dpi=200, bbox_inches="tight")
	plt.close(fig)
	logger.info("Saved ROC plot: %s", out_path)


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

	ir_root = Path(config.ir_dir)
	if not ir_root.is_absolute():
		ir_root = (project_root / ir_root).resolve()
	ir_files = sorted(list(ir_root.rglob("*.wav"))) if (config.use_ir and ir_root.exists()) else []

	logger = configure_logger(output_dir)
	logger.info("PROJECT_ROOT=%s", project_root)
	logger.info("OUTPUT_DIR=%s", output_dir)
	logger.info("CACHE_DIR=%s", cache_dir)
	logger.info("feature_sig=%s", feature_sig)
	logger.info("IR files=%d", len(ir_files))
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

		x_tr_raw, y_tr = df_to_feature_matrix(
			fold_info["df_train"],
			path_col="resolved_wav_path",
			label_col=config.coda_label_col,
			cache_dir=cache_dir,
			feature_sig=feature_sig,
			config=config,
			logger=logger,
			log_prefix=f"Fold {fold} train",
			train_mode=True,
			seed_base=int(config.seed + fold * 1000 + 11),
			ir_files=ir_files,
		)
		x_va_raw, y_va = df_to_feature_matrix(
			fold_info["df_val"],
			path_col="resolved_wav_path",
			label_col=config.coda_label_col,
			cache_dir=cache_dir,
			feature_sig=feature_sig,
			config=config,
			logger=logger,
			log_prefix=f"Fold {fold} val",
			train_mode=False,
			seed_base=int(config.seed + fold * 1000 + 22),
			ir_files=ir_files,
		)
		x_te_raw, y_te = df_to_feature_matrix(
			fold_info["df_test"],
			path_col="file_path",
			label_col="label",
			cache_dir=cache_dir,
			feature_sig=feature_sig,
			config=config,
			logger=logger,
			log_prefix=f"Fold {fold} test",
			train_mode=False,
			seed_base=int(config.seed + fold * 1000 + 33),
			ir_files=ir_files,
		)

		scaler = StandardScaler()
		x_tr = scaler.fit_transform(x_tr_raw)
		x_va = scaler.transform(x_va_raw)
		x_te = scaler.transform(x_te_raw)

		write_status(output_dir, run_tag, state="running", fold=fold, message="training model")
		model = build_model(config)
		model.fit(x_tr, y_tr)

		y_score_val = model.predict_proba(x_va)[:, 1].astype(np.float32)
		val_metrics = metrics_from_scores(y_va, y_score_val, thr=float(config.fixed_threshold))

		write_status(output_dir, run_tag, state="running", fold=fold, message="evaluating")
		y_score = model.predict_proba(x_te)[:, 1].astype(np.float32)
		metrics = metrics_from_scores(y_te, y_score, thr=float(config.fixed_threshold))
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
			metrics=metrics,
			metrics_ci=metrics_ci,
			y_true=y_te,
			y_score=y_score,
			val_metrics=val_metrics,
		)

		upsert_fold_result(
			all_fold_results,
			{
				"fold": fold,
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
			"FOLD %d DONE | AUC=%.4f ACC=%.4f F1=%.4f SENS=%.4f SPEC=%.4f | fold_time=%s eta_all~%s",
			fold,
			metrics["auc"],
			metrics["acc"],
			metrics["f1"],
			metrics["sens"],
			metrics["spec"],
			format_seconds(fold_time),
			format_seconds(eta_all),
		)

		write_status(output_dir, run_tag, state="running", fold=fold, message="fold completed")

		del model, x_tr_raw, x_va_raw, x_te_raw, x_tr, x_va, x_te
		gc.collect()

	total_elapsed = time.perf_counter() - run_t0
	save_summary(output_dir, config, all_fold_results, total_elapsed)
	save_roc_all_folds_plot(output_dir, all_fold_results, "extratrees", logger)
	write_status(output_dir, run_tag, state="completed", message="all available folds completed")

	logger.info("=" * 90)
	logger.info("TRAINING COMPLETE")
	logger.info("Total elapsed: %s", format_seconds(total_elapsed))
	logger.info("Saved artifacts in: %s", output_dir)


if __name__ == "__main__":
	main()
