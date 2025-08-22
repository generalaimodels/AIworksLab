#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced End-to-End Audio Classification Pipeline (YAML-driven, single-file, Rich-verbose)

This file implements a modular, scalable, and production-lean pipeline for audio classification:
- YAML-only configuration (single entrypoint): python audio_classifications.py path/to/config.yaml
- Robust pre-processing, feature engineering, normalization, and optional augmentations
- Training with automated checkpointing, best-model saving, early stopping, and optional HPO (Optuna/Ray)
- Comprehensive evaluation (accuracy, F1, precision, recall, top-k accuracy, confusion matrix reporting)
- Optional cross-validation (N-fold), reproducible runs, deterministic settings
- Deployment-oriented outputs: saved best model, model card, optional push-to-hub
- Rich-based verbose mode for beautiful logs, tables, and status (toggle via YAML: pipeline.verbose: true)

Everything (explanations, examples, code) lives in this single .py file to keep the experience clean.


USAGE
- Single YAML input only:
  $ python audio_classifications.py ./configs/audio.yaml


"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple, Union

# Third-party imports with graceful fallbacks
try:
    import yaml  # PyYAML
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from e

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
except Exception:  # pragma: no cover
    Console = None
    RichHandler = None
    Table = None
    Panel = None
    box = None

import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
        classification_report as sklearn_classification_report,
        top_k_accuracy_score,
    )
    from sklearn.model_selection import StratifiedKFold, KFold
except Exception:
    # We'll fallback to evaluate for basic metrics if sklearn is unavailable
    StratifiedKFold = None
    KFold = None

import datasets
import evaluate
import transformers
import torch

from datasets import Dataset, DatasetDict, Features, Value, ClassLabel, Audio, load_dataset
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, EvalPrediction
from transformers.data.data_collator import DataCollatorWithPadding


# ----------------------------
# Global logger and console
# ----------------------------
LOGGER = logging.getLogger("audio_classifications")
RICH_CONSOLE: Optional[Console] = Console(width=120, force_terminal=True) if Console else None


# ----------------------------
# Utilities: Logging & YAML
# ----------------------------
def setup_logging(verbose: bool = True) -> None:
    """Configure logging; if rich is available and verbose, use RichHandler."""
    log_level = logging.DEBUG if verbose else logging.INFO
    handlers: List[logging.Handler] = []
    if RichHandler is not None and RICH_CONSOLE is not None and verbose:
        handlers.append(
            RichHandler(
                console=RICH_CONSOLE,
                rich_tracebacks=True,
                show_time=True,
                show_path=False,
                log_time_format="[%X]",
            )
        )
    else:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=log_level,
        format="%(message)s" if RichHandler and verbose else "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=handlers,
    )
    LOGGER.setLevel(log_level)
    transformers.utils.logging.set_verbosity_info() if verbose else transformers.utils.logging.set_verbosity_warning()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def read_yaml_config(yaml_path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """Read YAML and expand environment variables inside."""
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"YAML config not found: {yaml_path}")
    text = path.read_text(encoding="utf-8")
    text = os.path.expandvars(text)  # ${VAR} expansions
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML content must be a mapping/object.")
    return data


def dump_as_json(obj: Any, path: Union[str, os.PathLike]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ----------------------------
# Dataclasses for structured config
# ----------------------------
@dataclass
class PipelineConfig:
    mode: str = "train"  # train | eval | predict | export (export reserved)
    verbose: bool = True
    use_rich: bool = True
    seed: int = 42
    output_dir: str = "./outputs/exp"
    cache_dir: Optional[str] = None
    trust_remote_code: bool = False
    token: Optional[str] = None


@dataclass
class DataConfig:
    input_format: str = "hf"  # "hf" | "csv" | "json" | "folders"
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    train_split_name: str = "train"
    eval_split_name: str = "validation"
    train_file: Optional[str] = None
    eval_file: Optional[str] = None
    data_dir: Optional[str] = None
    audio_column_name: str = "audio"
    label_column_name: str = "label"
    sampling_rate: Optional[int] = None
    max_length_seconds: float = 5.0
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    normalization: str = "peak"  # "off" | "peak" | "rms"
    augment: Dict[str, Any] = field(default_factory=lambda: {
        "time_shift_max_ratio": 0.0,
        "noise_prob": 0.0,
        "noise_snr_db": [15, 30],
    })


@dataclass
class ModelConfig:
    model_name_or_path: str = "facebook/wav2vec2-base"
    config_name: Optional[str] = None
    feature_extractor_name: Optional[str] = None
    freeze_feature_encoder: bool = True
    attention_mask: bool = True
    ignore_mismatched_sizes: bool = False
    cache_dir: Optional[str] = None
    model_revision: str = "main"
    trust_remote_code: bool = False
    token: Optional[str] = None


@dataclass
class EarlyStoppingCfg:
    enabled: bool = False
    patience: int = 3
    threshold: float = 0.0


@dataclass
class EvalConfig:
    compute_accuracy: bool = True
    compute_f1: bool = True
    compute_precision: bool = True
    compute_recall: bool = True
    f_average: str = "macro"  # 'micro' | 'macro' | 'weighted'
    top_k: List[int] = field(default_factory=lambda: [3])
    confusion_matrix: bool = True
    classification_report: bool = True


@dataclass
class HPOConfig:
    enabled: bool = False
    backend: str = "optuna"  # "optuna" | "ray"
    direction: str = "maximize"
    n_trials: int = 10
    timeout: Optional[int] = None
    search_space: Dict[str, Any] = field(default_factory=dict)  # e.g., {"learning_rate": {"low": 1e-5, "high": 5e-4, "log": True}}


@dataclass
class CrossValConfig:
    n_splits: int = 1
    shuffle: bool = True
    stratified: bool = True
    seed: int = 42


@dataclass
class PredictConfig:
    files: List[str] = field(default_factory=list)
    folder: Optional[str] = None
    pattern: str = "*.wav"
    recursive: bool = True
    output_csv: str = "./outputs/predictions.csv"


@dataclass
class DeployConfig:
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_private_repo: bool = False
    token: Optional[str] = None


# ----------------------------
# Config loading helpers
# ----------------------------
def _merge(dst: MutableMapping[str, Any], src: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = v
    return dst


def load_configs(yaml_path: str) -> Tuple[PipelineConfig, DataConfig, ModelConfig, Dict[str, Any], EarlyStoppingCfg, EvalConfig, HPOConfig, CrossValConfig, PredictConfig, DeployConfig]:
    raw = read_yaml_config(yaml_path)

    pipeline = PipelineConfig(**raw.get("pipeline", {}))
    data = DataConfig(**raw.get("data", {}))
    model = ModelConfig(**raw.get("model", {}))
    training_kwargs = raw.get("training", {}) or {}
    early_stopping = EarlyStoppingCfg(**raw.get("early_stopping", {}))
    eval_cfg = EvalConfig(**raw.get("evaluation", {}))
    hpo_cfg = HPOConfig(**raw.get("hpo", {}))
    cv_cfg = CrossValConfig(**raw.get("cross_validation", {}))
    predict_cfg = PredictConfig(**raw.get("predict", {}))
    deploy_cfg = DeployConfig(**raw.get("deploy", {}))

    # ensure output_dir present in training kwargs
    training_kwargs = dict(training_kwargs)
    training_kwargs.setdefault("output_dir", pipeline.output_dir)
    training_kwargs.setdefault("remove_unused_columns", False)

    return pipeline, data, model, training_kwargs, early_stopping, eval_cfg, hpo_cfg, cv_cfg, predict_cfg, deploy_cfg


# ----------------------------
# Reproducibility
# ----------------------------
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)  # transformers
    try:
        # Best-effort determinism for PyTorch
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except Exception:
        pass


# ----------------------------
# Dataset loading and preprocessing
# ----------------------------
def load_dataset_from_config(data: DataConfig, token: Optional[str] = None, trust_remote_code: bool = False) -> DatasetDict:
    """Load data according to input_format."""
    if data.input_format == "hf":
        if not data.dataset_name:
            raise ValueError("data.dataset_name must be set for input_format='hf'")
        raw = DatasetDict()
        raw["train"] = load_dataset(
            data.dataset_name,
            data.dataset_config_name,
            split=data.train_split_name,
            token=token,
            trust_remote_code=trust_remote_code,
        )
        raw["eval"] = load_dataset(
            data.dataset_name,
            data.dataset_config_name,
            split=data.eval_split_name,
            token=token,
            trust_remote_code=trust_remote_code,
        )
        return raw

    elif data.input_format in {"csv", "json"}:
        if not data.train_file or not data.eval_file:
            raise ValueError("data.train_file and data.eval_file must be set for input_format csv/json")
        loader = "csv" if data.input_format == "csv" else "json"
        dsd = load_dataset(
            loader,
            data_files={"train": data.train_file, "eval": data.eval_file},
        )
        # Cast audio column
        if data.audio_column_name not in dsd["train"].column_names:
            raise ValueError(f"audio_column_name '{data.audio_column_name}' not found in CSV/JSON columns: {dsd['train'].column_names}")
        # Ensure label column exists
        if data.label_column_name not in dsd["train"].column_names:
            raise ValueError(f"label_column_name '{data.label_column_name}' not found in CSV/JSON columns: {dsd['train'].column_names}")
        return DatasetDict({"train": dsd["train"], "eval": dsd["eval"]})

    elif data.input_format == "folders":
        # Expect directory structure: data_dir/<label>/*.wav
        if not data.data_dir:
            raise ValueError("data.data_dir must be set for input_format='folders'")
        data_dir = Path(data.data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"data_dir not found: {data.data_dir}")
        paths, labels = [], []
        for label_dir in sorted([p for p in data_dir.iterdir() if p.is_dir()]):
            label = label_dir.name
            for wav in label_dir.rglob("*"):
                if wav.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}:
                    paths.append(str(wav))
                    labels.append(label)
        if not paths:
            raise ValueError(f"No audio files found in hierarchical {data.data_dir}")
        # Split 90/10 train/eval
        idx = np.arange(len(paths))
        np.random.shuffle(idx)
        split = int(0.9 * len(paths))
        tr_idx, ev_idx = idx[:split], idx[split:]
        build = lambda ids: Dataset.from_dict({data.audio_column_name: [paths[i] for i in ids], data.label_column_name: [labels[i] for i in ids]})
        return DatasetDict({"train": build(tr_idx), "eval": build(ev_idx)})

    else:
        raise ValueError(f"Unsupported data.input_format: {data.input_format}")


def build_label_mappings(dsd: DatasetDict, label_column: str) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """Extract label names and mappings. Supports both ClassLabel and string labels."""
    features = dsd["train"].features
    if isinstance(features.get(label_column), ClassLabel):
        names = list(features[label_column].names)
    else:
        # Build from train + eval
        labels = set()
        for split in ["train", "eval"]:
            if split in dsd:
                labels.update(dsd[split][label_column])
        names = sorted(list(labels))
    label2id = {name: i for i, name in enumerate(names)}
    id2label = {i: name for name, i in label2id.items()}
    return names, label2id, id2label


def random_subsample(wav: np.ndarray, max_length: float, sample_rate: int) -> np.ndarray:
    """Randomly sample chunk of `max_length` seconds from wav."""
    sample_len = int(round(sample_rate * max_length))
    if len(wav) <= sample_len:
        return wav
    start = random.randint(0, len(wav) - sample_len)
    return wav[start : start + sample_len]


def normalize_waveform(wav: np.ndarray, mode: str = "peak") -> np.ndarray:
    if mode == "off":
        return wav
    wav = wav.astype(np.float32)
    eps = 1e-8
    if mode == "peak":
        peak = np.max(np.abs(wav)) + eps
        return wav / peak
    if mode == "rms":
        rms = np.sqrt(np.mean(wav ** 2)) + eps
        return wav / rms
    return wav


def maybe_timeshift(wav: np.ndarray, max_ratio: float = 0.0) -> np.ndarray:
    if max_ratio <= 0.0:
        return wav
    shift = int(len(wav) * random.uniform(-max_ratio, max_ratio))
    if shift == 0:
        return wav
    return np.roll(wav, shift)


def maybe_add_noise(wav: np.ndarray, snr_db_range: Tuple[float, float], prob: float = 0.0) -> np.ndarray:
    if prob <= 0.0 or random.random() > prob:
        return wav
    snr_db = random.uniform(*snr_db_range)
    # signal power
    sig_power = np.mean(wav ** 2) + 1e-9
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise = np.random.normal(0.0, math.sqrt(noise_power), size=wav.shape).astype(wav.dtype)
    return wav + noise


def make_transforms(
    data_cfg: DataConfig,
    feat_extractor: Any,
    model_input_name: str,
    training: bool,
) -> Any:
    """Return a function(batch) -> dict for datasets.set_transform."""
    def _apply(batch: Dict[str, Any]) -> Dict[str, Any]:
        wavs: List[np.ndarray] = []
        labels: List[int] = []
        for i, audio in enumerate(batch[data_cfg.audio_column_name]):
            arr = audio["array"] if isinstance(audio, dict) else audio  # supports if column already cast to Audio
            sr = audio.get("sampling_rate", None) if isinstance(audio, dict) else None
            if data_cfg.sampling_rate and sr and sr != data_cfg.sampling_rate:
                # datasets.Audio will handle resampling when casting; this is fallback
                # avoid custom resampling here to keep single-file minimal
                pass
            # normalize
            x = normalize_waveform(arr, mode=data_cfg.normalization)
            if training:
                # data augmentation
                x = random_subsample(x, data_cfg.max_length_seconds, feat_extractor.sampling_rate)
                x = maybe_timeshift(x, max_ratio=float(data_cfg.augment.get("time_shift_max_ratio", 0.0) or 0.0))
                x = maybe_add_noise(
                    x,
                    snr_db_range=tuple(data_cfg.augment.get("noise_snr_db", [15, 30])),
                    prob=float(data_cfg.augment.get("noise_prob", 0.0) or 0.0),
                )
            wavs.append(x)
            if data_cfg.label_column_name in batch:
                labels.append(batch[data_cfg.label_column_name][i])
        inputs = feat_extractor(wavs, sampling_rate=feat_extractor.sampling_rate)
        out = {model_input_name: inputs.get(model_input_name)}
        if "attention_mask" in inputs:
            out["attention_mask"] = inputs["attention_mask"]
        if labels:
            out["labels"] = labels
        return out
    return _apply


def cast_audio_columns(dsd: DatasetDict, audio_col: str, sampling_rate: int) -> DatasetDict:
    if audio_col not in dsd["train"].column_names:
        raise ValueError(f"audio_column_name '{audio_col}' not found. Available: {dsd['train'].column_names}")
    # datasets will load and resample automatically
    return dsd.cast_column(audio_col, Audio(sampling_rate=sampling_rate))


def maybe_subsample(dsd: DatasetDict, subset_sizes: Dict[str, Optional[int]], seed: int) -> DatasetDict:
    out = DatasetDict()
    for split in dsd.keys():
        ds = dsd[split]
        max_samples = subset_sizes.get(f"max_{split}_samples")
        if max_samples:
            ds = ds.shuffle(seed=seed).select(range(min(max_samples, len(ds))))
        out[split] = ds
    return out


# ----------------------------
# Model/Trainer plumbing
# ----------------------------
def load_feature_extractor(model_cfg: ModelConfig) -> Any:
    """Load feature extractor (or processor) for audio model."""
    fe_kwargs = dict(
        return_attention_mask=model_cfg.attention_mask,
        cache_dir=model_cfg.cache_dir,
        revision=model_cfg.model_revision,
        token=model_cfg.token,
        trust_remote_code=model_cfg.trust_remote_code,
    )
    try:
        feat_extractor = AutoFeatureExtractor.from_pretrained(
            model_cfg.feature_extractor_name or model_cfg.model_name_or_path, **fe_kwargs
        )
        return feat_extractor
    except Exception as e:
        LOGGER.warning(f"AutoFeatureExtractor failed with: {e}; attempting AutoProcessor fallback.")
        try:
            from transformers import AutoProcessor
            feat_extractor = AutoProcessor.from_pretrained(
                model_cfg.feature_extractor_name or model_cfg.model_name_or_path, **fe_kwargs
            )
            return feat_extractor
        except Exception as e2:
            raise RuntimeError("Failed to load feature extractor/processor.") from e2


def build_compute_metrics(eval_cfg: EvalConfig, id2label: Dict[int, str]) -> Any:
    acc_metric = evaluate.load("accuracy")
    prec_metric = evaluate.load("precision")
    rec_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    def _compute(eval_pred: EvalPrediction) -> Dict[str, float]:
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
        preds = np.argmax(logits, axis=1)
        metrics: Dict[str, float] = {}

        if eval_cfg.compute_accuracy:
            metrics["accuracy"] = float(acc_metric.compute(predictions=preds, references=labels)["accuracy"])

        if eval_cfg.compute_precision:
            metrics["precision"] = float(prec_metric.compute(predictions=preds, references=labels, average=eval_cfg.f_average)["precision"])

        if eval_cfg.compute_recall:
            metrics["recall"] = float(rec_metric.compute(predictions=preds, references=labels, average=eval_cfg.f_average)["recall"])

        if eval_cfg.compute_f1:
            metrics["f1"] = float(f1_metric.compute(predictions=preds, references=labels, average=eval_cfg.f_average)["f1"])

        # Optional top-k
        if hasattr(np, "argsort") and len(eval_cfg.top_k) > 0:
            # Safe top-k accuracy using sklearn if available
            try:
                for k in eval_cfg.top_k:
                    if k > logits.shape[1]:
                        continue
                    tk = top_k_accuracy_score(labels, logits, k=k)  # type: ignore[call-arg]
                    metrics[f"top_{k}_accuracy"] = float(tk)
            except Exception:
                pass

        return metrics

    return _compute


def create_trainer(
    model: transformers.PreTrainedModel,
    feature_extractor: Any,
    training_args: TrainingArguments,
    train_ds: Optional[Dataset],
    eval_ds: Optional[Dataset],
    compute_metrics_fn: Any,
    data_collator: Optional[Any] = None,
    callbacks: Optional[List[transformers.TrainerCallback]] = None,
) -> Trainer:
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics_fn,
        tokenizer=feature_extractor,  # allows smart padding/collation
        data_collator=data_collator,
        callbacks=callbacks,
    )


def ensure_training_args(training_kwargs: Dict[str, Any], pipeline: PipelineConfig) -> TrainingArguments:
    # Guarantee essential fields
    training_kwargs = dict(training_kwargs)
    training_kwargs.setdefault("output_dir", pipeline.output_dir)
    training_kwargs.setdefault("evaluation_strategy", "epoch")
    training_kwargs.setdefault("save_strategy", "epoch")
    training_kwargs.setdefault("logging_strategy", "steps")
    training_kwargs.setdefault("logging_steps", 50)
    training_kwargs.setdefault("remove_unused_columns", False)
    training_kwargs.setdefault("report_to", [])
    training_kwargs.setdefault("load_best_model_at_end", True)
    training_kwargs.setdefault("metric_for_best_model", "accuracy")
    training_kwargs.setdefault("greater_is_better", True)
    training_kwargs.setdefault("seed", pipeline.seed)
    training_kwargs.setdefault("fp16", torch.cuda.is_available())
    training_kwargs.setdefault("dataloader_num_workers", 4)

    # Set cache and hub args if provided at pipeline-level
    if pipeline.cache_dir is not None:
        training_kwargs.setdefault("cache_dir", pipeline.cache_dir)

    return TrainingArguments(**training_kwargs)


# ----------------------------
# Evaluation helpers
# ----------------------------
def detailed_eval_report(
    logits: np.ndarray,
    labels: np.ndarray,
    id2label: Dict[int, str],
    eval_cfg: EvalConfig,
    out_dir: Union[str, os.PathLike],
) -> None:
    preds = np.argmax(logits, axis=1)
    out: Dict[str, Any] = {}

    if eval_cfg.confusion_matrix:
        try:
            cm = confusion_matrix(labels, preds).tolist()  # type: ignore
            out["confusion_matrix"] = cm
        except Exception:
            pass

    if eval_cfg.classification_report:
        try:
            rep = sklearn_classification_report(labels, preds, target_names=[id2label[i] for i in range(len(id2label))], output_dict=True)  # type: ignore
            out["classification_report"] = rep
        except Exception:
            pass

    if out:
        dump_as_json(out, Path(out_dir) / "eval_reports.json")


# ----------------------------
# Hyperparameter Optimization
# ----------------------------
def _hp_space_from_yaml(trial: Any, search_space: Dict[str, Any]) -> Dict[str, Any]:
    # Compatible with optuna trial API
    # search_space:
    #   learning_rate: {low: 1e-5, high: 5e-4, log: true}
    #   weight_decay: {low: 0.0, high: 0.1, step: 0.01}
    #   per_device_train_batch_size: {choices: [8, 16, 32]}
    hp: Dict[str, Any] = {}
    for name, spec in search_space.items():
        if "choices" in spec:
            hp[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            low = spec.get("low")
            high = spec.get("high")
            step = spec.get("step", None)
            log = spec.get("log", False)
            if log:
                hp[name] = trial.suggest_float(name, low, high, log=True)
            else:
                hp[name] = trial.suggest_float(name, low, high, step=step)
    return hp


def run_hpo(
    base_trainer: Trainer,
    hpo_cfg: HPOConfig,
    search_space: Dict[str, Any],
) -> Optional[transformers.integrations.HPXxxSearchBackend]:
    if not hpo_cfg.enabled:
        return None
    backend = hpo_cfg.backend.lower()
    LOGGER.info(f"Starting HPO with backend='{backend}', trials={hpo_cfg.n_trials}")
    kwargs = dict(direction=hpo_cfg.direction, n_trials=hpo_cfg.n_trials)
    if hpo_cfg.timeout:
        kwargs["timeout"] = hpo_cfg.timeout

    if backend == "optuna":
        try:
            import optuna  # noqa
        except Exception as e:
            raise RuntimeError("Optuna is required for HPO 'optuna'. Install: pip install optuna") from e

        def optuna_hp_space(trial):
            return _hp_space_from_yaml(trial, search_space)

        best_run = base_trainer.hyperparameter_search(
            hp_space=optuna_hp_space,
            direction=hpo_cfg.direction,
            n_trials=hpo_cfg.n_trials,
            backend="optuna",
        )
        LOGGER.info(f"HPO finished. Best run: {best_run}")
        return best_run

    elif backend == "ray":
        try:
            import ray  # noqa
        except Exception as e:
            raise RuntimeError("Ray Tune is required for HPO 'ray'. Install: pip install ray[tune]") from e

        # Trainers internal Ray integration requires a Ray-compatible search space
        def ray_hp_space(trial):
            return search_space

        best_run = base_trainer.hyperparameter_search(
            hp_space=ray_hp_space,
            direction=hpo_cfg.direction,
            n_trials=hpo_cfg.n_trials,
            backend="ray",
        )
        LOGGER.info(f"HPO finished. Best run: {best_run}")
        return best_run

    else:
        raise ValueError(f"Unsupported HPO backend: {hpo_cfg.backend}")


# ----------------------------
# Prediction
# ----------------------------
def collect_prediction_files(pred_cfg: PredictConfig) -> List[str]:
    files = list(pred_cfg.files)
    if pred_cfg.folder:
        base = Path(pred_cfg.folder)
        if pred_cfg.recursive:
            files.extend([str(p) for p in base.rglob(pred_cfg.pattern)])
        else:
            files.extend([str(p) for p in base.glob(pred_cfg.pattern)])
    # Filter audio-like
    files = [f for f in files if Path(f).suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}]
    if not files:
        raise ValueError("No prediction files found. Provide predict.files or predict.folder/pattern.")
    return sorted(list(set(files)))


def build_prediction_dataset(files: List[str], audio_col: str = "audio") -> Dataset:
    return Dataset.from_dict({audio_col: files}).cast_column(audio_col, Audio())


def run_prediction(
    trainer: Trainer,
    feature_extractor: Any,
    data_cfg: DataConfig,
    pred_cfg: PredictConfig,
    model_input_name: str,
    id2label: Dict[int, str],
    out_dir: Union[str, os.PathLike],
) -> str:
    paths = collect_prediction_files(pred_cfg)
    ds = build_prediction_dataset(paths, audio_col=data_cfg.audio_column_name)

    # Construct eval-like transforms (no random crop by default; whole clip)
    def val_transforms(batch):
        wavs = []
        for audio in batch[data_cfg.audio_column_name]:
            arr = audio["array"]
            x = normalize_waveform(arr, mode=data_cfg.normalization)
            wavs.append(x)
        inputs = feature_extractor(wavs, sampling_rate=feature_extractor.sampling_rate)
        out = {model_input_name: inputs.get(model_input_name)}
        if "attention_mask" in inputs:
            out["attention_mask"] = inputs["attention_mask"]
        return out

    ds = ds.with_transform(val_transforms)
    collator = DataCollatorWithPadding(tokenizer=feature_extractor, padding=True)

    preds_output = trainer.predict(ds, metric_key_prefix="predict")
    logits = preds_output.predictions
    pred_ids = np.argmax(logits, axis=1)
    pred_labels = [id2label[int(i)] for i in pred_ids]

    rows = [{"path": p, "pred_id": int(i), "pred_label": l} for p, i, l in zip(paths, pred_ids, pred_labels)]
    out_csv = Path(pred_cfg.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Write CSV manually to avoid extra deps
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("path,pred_id,pred_label\n")
        for r in rows:
            f.write(f"{r['path']},{r['pred_id']},{r['pred_label']}\n")

    LOGGER.info(f"Wrote predictions to: {out_csv}")
    return str(out_csv)


# ----------------------------
# Cross Validation
# ----------------------------
def run_cross_validation(
    dsd: DatasetDict,
    label_col: str,
    cv_cfg: CrossValConfig,
    build_trainer_fn,
    training_args_builder,
    feature_extractor: Any,
    model_builder_fn,
    make_transforms_train,
    make_transforms_eval,
    model_input_name: str,
    eval_cfg: EvalConfig,
    out_dir: Union[str, os.PathLike],
) -> None:
    if cv_cfg.n_splits <= 1:
        LOGGER.info("Cross-validation n_splits <= 1: skipping CV.")
        return

    if cv_cfg.stratified and StratifiedKFold is None:
        raise RuntimeError("scikit-learn is required for stratified cross-validation. Install: pip install scikit-learn")

    train_ds = dsd["train"]
    y = np.array(train_ds[label_col])
    indices = np.arange(len(train_ds))

    if cv_cfg.stratified:
        splitter = StratifiedKFold(n_splits=cv_cfg.n_splits, shuffle=cv_cfg.shuffle, random_state=cv_cfg.seed)
        splits = splitter.split(indices, y)
    else:
        splitter = KFold(n_splits=cv_cfg.n_splits, shuffle=cv_cfg.shuffle, random_state=cv_cfg.seed) if KFold else None
        if splitter is None:
            raise RuntimeError("scikit-learn is required for K-Fold cross-validation. Install: pip install scikit-learn")
        splits = splitter.split(indices)

    fold_metrics: List[Dict[str, float]] = []
    for fold_id, (tr_idx, val_idx) in enumerate(splits):
        LOGGER.info(f"CV Fold {fold_id+1}/{cv_cfg.n_splits}: train={len(tr_idx)}, val={len(val_idx)}")

        ds_train = Dataset.from_dict(train_ds.select(tr_idx).to_dict())
        ds_eval = Dataset.from_dict(train_ds.select(val_idx).to_dict())
        ds_train = ds_train.cast_column(dsd["train"].column_names[0], dsd["train"].features[dsd["train"].column_names[0]]) if isinstance(dsd["train"].features.get(ds_train.column_names[0]), Audio) else ds_train  # safe cast
        # Cast audio col to Audio with sampling rate using feature_extractor
        ds_train = DatasetDict({"train": ds_train})["train"]
        ds_eval = DatasetDict({"eval": ds_eval})["eval"]

        ds_train = ds_train.with_transform(make_transforms_train)
        ds_eval = ds_eval.with_transform(make_transforms_eval)

        # Build new model per fold
        model = model_builder_fn()
        # TrainingArguments: unique output dir per fold
        ta = training_args_builder(output_dir=str(Path(out_dir) / f"cv_fold_{fold_id}"))
        collator = DataCollatorWithPadding(tokenizer=feature_extractor, padding=True)
        trainer = build_trainer_fn(model, feature_extractor, ta, ds_train, ds_eval, collator)

        # Train
        trainer.train()
        metrics = trainer.evaluate()
        fold_metrics.append(metrics)
        trainer.save_metrics(f"cv_eval_fold_{fold_id}", metrics)

    # Aggregate
    if fold_metrics:
        keys = sorted({k for m in fold_metrics for k in m.keys() if isinstance(m[k], (int, float))})
        agg = {}
        for k in keys:
            vals = [float(m.get(k, float("nan"))) for m in fold_metrics]
            vals = [v for v in vals if not math.isnan(v)]
            if vals:
                agg[k + "_mean"] = float(np.mean(vals))
                agg[k + "_std"] = float(np.std(vals))
        dump_as_json({"folds": fold_metrics, "aggregate": agg}, Path(out_dir) / "cv_metrics.json")
        LOGGER.info(f"CV complete. Aggregate metrics: {agg}")


# ----------------------------
# Main pipeline execution
# ----------------------------
def main():
    # Enforce YAML-only CLI
    if len(sys.argv) != 2 or not sys.argv[1].lower().endswith((".yml", ".yaml")):
        print("Usage: python audio_classifications.py path/to/config.yaml", file=sys.stderr)
        sys.exit(2)

    # Load configurations
    pipeline_cfg, data_cfg, model_cfg, training_kwargs, early_stopping_cfg, eval_cfg, hpo_cfg, cv_cfg, predict_cfg, deploy_cfg = load_configs(
        sys.argv[1]
    )

    # Setup logging
    setup_logging(verbose=pipeline_cfg.verbose)
    if RICH_CONSOLE and pipeline_cfg.verbose:
        RICH_CONSOLE.rule("[bold green]Audio Classification Pipeline")

    # Reproducibility
    set_global_seed(pipeline_cfg.seed)

    # Detect last checkpoint if resuming
    last_checkpoint = None
    out_dir = Path(pipeline_cfg.output_dir)
    if out_dir.exists():
        try:
            last_checkpoint = get_last_checkpoint(str(out_dir))
        except Exception:
            last_checkpoint = None

    # Load feature extractor
    feature_extractor = load_feature_extractor(model_cfg)
    if data_cfg.sampling_rate is None:
        data_cfg.sampling_rate = int(feature_extractor.sampling_rate)

    # Load dataset
    raw_datasets = load_dataset_from_config(data_cfg, token=model_cfg.token or pipeline_cfg.token, trust_remote_code=(model_cfg.trust_remote_code or pipeline_cfg.trust_remote_code))

    # Ensure required columns
    if data_cfg.audio_column_name not in raw_datasets["train"].column_names:
        raise ValueError(f"audio_column_name '{data_cfg.audio_column_name}' not found in dataset columns: {raw_datasets['train'].column_names}")
    if data_cfg.label_column_name not in raw_datasets["train"].column_names:
        raise ValueError(f"label_column_name '{data_cfg.label_column_name}' not found in dataset columns: {raw_datasets['train'].column_names}")

    # Cast audio columns
    raw_datasets = cast_audio_columns(raw_datasets, data_cfg.audio_column_name, data_cfg.sampling_rate or feature_extractor.sampling_rate)

    # Label mappings
    label_names, label2id, id2label = build_label_mappings(raw_datasets, data_cfg.label_column_name)
    if RICH_CONSOLE and pipeline_cfg.verbose:
        table = Table(title="Label Mapping", box=box.SIMPLE)
        table.add_column("id", justify="right")
        table.add_column("label", justify="left")
        for i, n in enumerate(label_names):
            table.add_row(str(i), n)
        RICH_CONSOLE.print(table)

    # Subsample for quicker runs (optional)
    raw_datasets = maybe_subsample(
        raw_datasets,
        {"max_train_samples": data_cfg.max_train_samples, "max_eval_samples": data_cfg.max_eval_samples},
        seed=pipeline_cfg.seed,
    )

    # Model + Config
    config = AutoConfig.from_pretrained(
        model_cfg.config_name or model_cfg.model_name_or_path,
        num_labels=len(label_names),
        label2id={k: int(v) for k, v in label2id.items()},
        id2label={int(k): v for k, v in id2label.items()},
        finetuning_task="audio-classification",
        cache_dir=model_cfg.cache_dir or pipeline_cfg.cache_dir,
        revision=model_cfg.model_revision,
        token=model_cfg.token or pipeline_cfg.token,
        trust_remote_code=model_cfg.trust_remote_code or pipeline_cfg.trust_remote_code,
    )
    model = AutoModelForAudioClassification.from_pretrained(
        model_cfg.model_name_or_path,
        from_tf=bool(".ckpt" in model_cfg.model_name_or_path),
        config=config,
        cache_dir=model_cfg.cache_dir or pipeline_cfg.cache_dir,
        revision=model_cfg.model_revision,
        token=model_cfg.token or pipeline_cfg.token,
        trust_remote_code=model_cfg.trust_remote_code or pipeline_cfg.trust_remote_code,
        ignore_mismatched_sizes=model_cfg.ignore_mismatched_sizes,
    )
    if model_cfg.freeze_feature_encoder and hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()

    # Transforms
    model_input_name = feature_extractor.model_input_names[0]
    train_transforms = make_transforms(data_cfg, feature_extractor, model_input_name, training=True)
    val_transforms = make_transforms(data_cfg, feature_extractor, model_input_name, training=False)

    # Apply transforms
    if pipeline_cfg.mode in {"train", "eval"}:
        if pipeline_cfg.mode == "train":
            raw_datasets["train"].set_transform(train_transforms, output_all_columns=False)
        raw_datasets["eval"].set_transform(val_transforms, output_all_columns=False)

    # Compute metrics
    compute_metrics_fn = build_compute_metrics(eval_cfg, id2label)

    # TrainingArguments
    training_args = ensure_training_args(training_kwargs, pipeline_cfg)
    # Determine resume
    resume_from_checkpoint = training_kwargs.get("resume_from_checkpoint")
    if resume_from_checkpoint is None and last_checkpoint is not None:
        LOGGER.info(f"Detected last checkpoint at {last_checkpoint}, will resume automatically.")
        resume_from_checkpoint = last_checkpoint

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=feature_extractor, padding=True)

    # Callbacks
    callbacks: List[transformers.TrainerCallback] = []
    if early_stopping_cfg.enabled:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_cfg.patience, early_stopping_threshold=early_stopping_cfg.threshold))

    # Trainer builder to reuse in CV/HPO
    def trainer_builder(
        mdl: transformers.PreTrainedModel,
        feat_extr: Any,
        args: TrainingArguments,
        tr_ds: Optional[Dataset],
        ev_ds: Optional[Dataset],
        collator: Any,
    ) -> Trainer:
        return create_trainer(
            model=mdl,
            feature_extractor=feat_extr,
            training_args=args,
            train_ds=tr_ds,
            eval_ds=ev_ds,
            compute_metrics_fn=compute_metrics_fn,
            data_collator=collator,
            callbacks=callbacks,
        )

    # Model builder (for CV folds)
    def model_builder_fn() -> transformers.PreTrainedModel:
        m = AutoModelForAudioClassification.from_pretrained(
            model_cfg.model_name_or_path,
            from_tf=bool(".ckpt" in model_cfg.model_name_or_path),
            config=config,
            cache_dir=model_cfg.cache_dir or pipeline_cfg.cache_dir,
            revision=model_cfg.model_revision,
            token=model_cfg.token or pipeline_cfg.token,
            trust_remote_code=model_cfg.trust_remote_code or pipeline_cfg.trust_remote_code,
            ignore_mismatched_sizes=model_cfg.ignore_mismatched_sizes,
        )
        if model_cfg.freeze_feature_encoder and hasattr(m, "freeze_feature_encoder"):
            m.freeze_feature_encoder()
        return m

    # TrainingArguments builder for folds
    def ta_builder(output_dir: Optional[str] = None) -> TrainingArguments:
        kwargs = dict(dataclasses.asdict(training_args))
        # dataclasses.asdict(TrainingArguments) is not directly supported; re-create from provided training_kwargs
        # So we rebuild from original training_kwargs and override output_dir if needed.
        k = dict(training_kwargs)
        if output_dir:
            k["output_dir"] = output_dir
        k.setdefault("remove_unused_columns", False)
        return TrainingArguments(**k)

    # Main modes
    trainer: Optional[Trainer] = None

    if pipeline_cfg.mode == "train":
        # HPO or CV or plain training
        if cv_cfg.n_splits > 1:
            # Cross-validation training on "train" split only
            run_cross_validation(
                dsd=raw_datasets,
                label_col=data_cfg.label_column_name,
                cv_cfg=cv_cfg,
                build_trainer_fn=lambda m, f, a, tr, ev, col: trainer_builder(m, f, a, tr, ev, col),
                training_args_builder=ta_builder,
                feature_extractor=feature_extractor,
                model_builder_fn=model_builder_fn,
                make_transforms_train=train_transforms,
                make_transforms_eval=val_transforms,
                model_input_name=model_input_name,
                eval_cfg=eval_cfg,
                out_dir=pipeline_cfg.output_dir,
            )
            # Optionally, after CV, you could train final model on full train split or best fold only.
            LOGGER.info("Cross-validation complete.")
            return

        # Single-run trainer
        trainer = trainer_builder(
            mdl=model,
            feat_extr=feature_extractor,
            args=training_args,
            tr_ds=raw_datasets["train"],
            ev_ds=raw_datasets["eval"],
            collator=data_collator,
        )

        # HPO
        if hpo_cfg.enabled and hpo_cfg.search_space:
            best_run = run_hpo(trainer, hpo_cfg, hpo_cfg.search_space)
            if best_run is not None and hasattr(best_run, "hyperparameters"):
                # Update training_args with best hyperparameters and re-instantiate trainer
                LOGGER.info(f"Rebuilding trainer with best hyperparameters: {best_run.hyperparameters}")
                # Merge returned hyperparameters into training_kwargs
                training_kwargs.update(best_run.hyperparameters)
                training_args = ensure_training_args(training_kwargs, pipeline_cfg)
                trainer = trainer_builder(model_builder_fn(), feature_extractor, training_args, raw_datasets["train"], raw_datasets["eval"], data_collator)

        # Train
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

        # Evaluate after training
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        # Detailed reports
        try:
            eval_preds = trainer.predict(raw_datasets["eval"])
            detailed_eval_report(eval_preds.predictions, eval_preds.label_ids, id2label, eval_cfg, pipeline_cfg.output_dir)
        except Exception as e:
            LOGGER.warning(f"Could not produce detailed eval report: {e}")

        # Write model card and optionally push
        card_kwargs = {
            "finetuned_from": model_cfg.model_name_or_path,
            "tasks": "audio-classification",
            "dataset": data_cfg.dataset_name or data_cfg.input_format,
            "tags": ["audio-classification"],
        }
        if deploy_cfg.push_to_hub:
            trainer.push_to_hub(
                hub_model_id=deploy_cfg.hub_model_id,
                private=deploy_cfg.hub_private_repo,
                token=(deploy_cfg.token or pipeline_cfg.token or model_cfg.token),
                **card_kwargs,
            )
        else:
            trainer.create_model_card(**card_kwargs)

    elif pipeline_cfg.mode == "eval":
        # Build trainer without training
        trainer = trainer_builder(
            mdl=model,
            feat_extr=feature_extractor,
            args=training_args,
            tr_ds=None,
            ev_ds=raw_datasets["eval"],
            collator=data_collator,
        )
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        try:
            eval_preds = trainer.predict(raw_datasets["eval"])
            detailed_eval_report(eval_preds.predictions, eval_preds.label_ids, id2label, eval_cfg, pipeline_cfg.output_dir)
        except Exception as e:
            LOGGER.warning(f"Could not produce detailed eval report: {e}")

    elif pipeline_cfg.mode == "predict":
        # Build trainer; eval-mode dataset not required
        trainer = trainer_builder(
            mdl=model,
            feat_extr=feature_extractor,
            args=training_args,
            tr_ds=None,
            ev_ds=None,
            collator=data_collator,
        )
        run_prediction(trainer, feature_extractor, data_cfg, predict_cfg, model_input_name, id2label, pipeline_cfg.output_dir)

    else:
        raise ValueError(f"Unsupported pipeline.mode: {pipeline_cfg.mode}")


# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    # Minimal protection for dependency versions
    try:
        # If available, validate minimal versions gracefully (avoid dev pin failures)
        from transformers.utils.versions import require_version
        require_version("datasets>=1.14.0", "To fix: pip install -U datasets")
        # A conservative floor for transformers that supports current Trainer APIs
        require_version("transformers>=4.28.0", "To fix: pip install -U transformers")
    except Exception as _verr:
        warnings.warn(str(_verr))

    try:
        main()
    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        LOGGER.exception(f"Fatal error: {e}")
        sys.exit(1)