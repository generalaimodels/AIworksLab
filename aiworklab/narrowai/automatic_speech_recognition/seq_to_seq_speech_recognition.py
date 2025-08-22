
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
sequence_to_sequence_speech_recognition.py

Purpose
-------
A rigorously engineered, end-to-end (E2E) sequence-to-sequence (seq2seq) speech recognition pipeline
built on top of Hugging Face Transformers and Datasets. It supports:
- YAML-first configuration (the ONLY CLI argument is the path to a .yaml file).
- Modular data pre-processing, feature extraction, and normalization.
- Robust training with checkpointing, best-model saving, and optional early stopping.
- Evaluation with WER (and optional CER) for reproducibility and reliability.
- Optional hyperparameter optimization (Optuna/Ray Tune) with proper metric tracking.
- Polished, developer-grade logging and rich, verbose visualization via the `rich` module.

Philosophy
----------
- Input contract: The only user input is a single YAML file. All configuration lives there.
- Non-invasive defaults: Only pass arguments to frameworks if explicitly provided.
- Safety & clarity: Defensive programming, explicit typing, and careful error handling.
- Reproducibility: Stable seeds, deterministic flags where feasible, and thorough logging.

Usage
-----
$ python sequence_to_sequence_speech_recognition.py path/to/config.yaml

YAML Schema (minimal skeleton)
------------------------------
model_args:
  model_name_or_path: "openai/whisper-small"     # REQUIRED
  # Optional keys below; only passed if present in YAML:
  # config_name: null
  # tokenizer_name: null
  # feature_extractor_name: null
  # cache_dir: null
  # use_fast_tokenizer: true
  # model_revision: "main"
  # token: null
  # trust_remote_code: false
  # freeze_feature_encoder: true
  # freeze_encoder: false
  # forced_decoder_ids: null
  # suppress_tokens: null
  # apply_spec_augment: false

data_args:
  dataset_name: "mozilla-foundation/common_voice_11_0"  # REQUIRED
  dataset_config_name: "en"                              # Common Voice example
  train_split_name: "train"
  eval_split_name: "validation"
  audio_column_name: "audio"
  text_column_name: "sentence"
  max_duration_in_seconds: 20.0
  min_duration_in_seconds: 0.0
  preprocessing_num_workers: 4
  do_lower_case: true
  language: null    # e.g., "en", "hi", "ar" ... only for multilingual models like Whisper
  task: "transcribe"
  # max_train_samples: null
  # max_eval_samples: null
  # overwrite_cache: false
  # preprocessing_only: false

training_args:
  output_dir: "./outputs/whisper-small-cv11-en"         # REQUIRED
  overwrite_output_dir: false
  do_train: true
  do_eval: true
  do_predict: false
  per_device_train_batch_size: 8
  per_device_eval_batch_size: 8
  gradient_accumulation_steps: 2
  learning_rate: 1.0e-4
  num_train_epochs: 3
  warmup_steps: 500
  fp16: true
  evaluation_strategy: "steps"
  eval_steps: 500
  save_strategy: "steps"
  save_steps: 500
  save_total_limit: 2
  logging_strategy: "steps"
  logging_steps: 50
  predict_with_generate: true
  generation_max_length: 225
  generation_num_beams: 1
  load_best_model_at_end: true
  metric_for_best_model: "wer"
  greater_is_better: false
  seed: 42
  # push_to_hub: false
  # report_to: ["tensorboard"]   # ["none"], ["wandb"], ["mlflow"], etc.

runtime:
  verbose: true                  # toggles rich, colorful logs and tables
  early_stopping_patience: 5     # optional EarlyStoppingCallback; only set if provided

hpo:
  enabled: false                 # enable hyperparameter search (Optuna/Ray)
  backend: "optuna"              # "optuna" or "ray" if installed; fallback to optuna if both present
  n_trials: 10
  direction: "minimize"          # "minimize" for WER
  metric: "eval_wer"             # metric to optimize (e.g., "eval_wer")
  # Optional search space: only pass keys you want to search
  space:
    learning_rate:
      low: 1.0e-5
      high: 5.0e-4
      log: true
    per_device_train_batch_size:
      values: [4, 8, 16]
    gradient_accumulation_steps:
      values: [1, 2, 4]

inference:
  # Optional convenience block to run predictions after training (or evaluation-only)
  # If `do_predict` is true in training_args, we’ll run predictions on the eval dataset.
  save_predictions: true
  predictions_filename: "predictions.txt"

More Example YAMLs
------------------
1) Whisper multilingual (fine-tuning in Hindi on Common Voice 11.0 Hindi):
model_args:
  model_name_or_path: "openai/whisper-small"
data_args:
  dataset_name: "mozilla-foundation/common_voice_11_0"
  dataset_config_name: "hi"
  train_split_name: "train"
  eval_split_name: "validation"
  audio_column_name: "audio"
  text_column_name: "sentence"
  language: "hi"
  task: "transcribe"
training_args:
  output_dir: "./outputs/whisper-small-cv11-hi"
  do_train: true
  do_eval: true
  predict_with_generate: true
  per_device_train_batch_size: 8
  per_device_eval_batch_size: 8
  num_train_epochs: 2
  learning_rate: 2.0e-4
  evaluation_strategy: "steps"
  eval_steps: 400
  save_strategy: "steps"
  save_steps: 400
  load_best_model_at_end: true
  metric_for_best_model: "wer"
  greater_is_better: false
runtime:
  verbose: true

2) Lightweight debugging run (subset) on English Common Voice:
model_args:
  model_name_or_path: "openai/whisper-tiny"
data_args:
  dataset_name: "mozilla-foundation/common_voice_11_0"
  dataset_config_name: "en"
  max_train_samples: 200
  max_eval_samples: 50
  train_split_name: "train"
  eval_split_name: "validation"
  audio_column_name: "audio"
  text_column_name: "sentence"
training_args:
  output_dir: "./outputs/debug-whisper-tiny"
  do_train: true
  do_eval: true
  predict_with_generate: true
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 16
  num_train_epochs: 1
  learning_rate: 3.0e-4
runtime:
  verbose: true

Notes & Exceptions
------------------
- The script only reads a single YAML file; no other CLI flags are supported here.
- We strictly pass optional arguments only if they are present in the YAML (no brittle, generalized defaults).
- For multilingual Whisper checkpoints, set data_args.language and data_args.task; doing so on an English-only
  checkpoint will raise a ValueError (to avoid silent misconfigurations).
- If you enable HPO, ensure that the specified backend is installed (optuna or ray[tune]). Otherwise, HPO falls back
  to an informative error or a graceful no-op depending on your configuration.
- If preprocessing_only is true, we will preprocess and cache datasets, then exit early (no training).
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

# Rich console for human-first, high-signal logging.
try:
    from rich import print as rprint
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.table import Table
    from rich.panel import Panel
    from rich.traceback import install as rich_install
except Exception:
    # No hard dependency: we can still run headless if Rich isn't available.
    rprint = print  # type: ignore
    Console = None  # type: ignore
    RichHandler = None  # type: ignore
    Table = None  # type: ignore
    Panel = None  # type: ignore

# YAML parsing (prefer ruamel.yaml for round-trip safety; fallback to PyYAML if needed).
_YAML_ERR = None
try:
    from ruamel.yaml import YAML

    _yaml_loader = "ruamel"
except Exception as _e:
    _YAML_ERR = _e
    try:
        import yaml  # type: ignore

        _yaml_loader = "pyyaml"
    except Exception as _e2:
        _YAML_ERR = (_e, _e2)
        _yaml_loader = None

import datasets
import evaluate
import torch
from datasets import DatasetDict, load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    EarlyStoppingCallback,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import send_example_telemetry


# Try to check versions but do not hard fail unless it's fundamentally broken.
try:
    from transformers.utils import check_min_version
    check_min_version("4.41.0")
except Exception as _v_e:
    warnings.warn(
        f"Transformers version check failed or older version detected: {_v_e}. "
        "You may proceed, but some arguments or features could be unavailable."
    )

try:
    from transformers.utils.versions import require_version
    require_version("datasets>=1.18.0", "To fix: pip install -U 'datasets>=1.18.0'")
except Exception as _ds_e:
    warnings.warn(
        f"Datasets version check failed (or version too old): {_ds_e}. "
        "The script might still run, but upgrade is recommended."
    )

logger = logging.getLogger("seq2seq_asr")


# --------------------------------------------------------------------------------------
# Dataclasses for args (we strictly pass only the keys present in YAML).
# --------------------------------------------------------------------------------------
@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "HF path or local path to a pretrained speech seq2seq model."})
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    feature_extractor_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    token: Optional[str] = field(default=None)
    trust_remote_code: bool = field(default=False)
    freeze_feature_encoder: bool = field(default=True)
    freeze_encoder: bool = field(default=False)
    forced_decoder_ids: Optional[List[List[int]]] = field(default=None)
    suppress_tokens: Optional[List[int]] = field(default=None)
    apply_spec_augment: bool = field(default=False)


@dataclass
class DataTrainingArguments:
    dataset_name: str = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    overwrite_cache: bool = field(default=False)
    preprocessing_num_workers: Optional[int] = field(default=None)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    audio_column_name: str = field(default="audio")
    text_column_name: str = field(default="text")
    max_duration_in_seconds: float = field(default=20.0)
    min_duration_in_seconds: float = field(default=0.0)
    preprocessing_only: bool = field(default=False)
    train_split_name: str = field(default="train")
    eval_split_name: str = field(default="test")
    do_lower_case: bool = field(default=True)
    language: Optional[str] = field(default=None)
    task: str = field(default="transcribe")


# Optional runtime args (not parsed by HfArgumentParser to avoid CLI collisions)
@dataclass
class RuntimeArguments:
    verbose: bool = True
    early_stopping_patience: Optional[int] = None


@dataclass
class HPOArguments:
    enabled: bool = False
    backend: str = "optuna"  # "optuna" or "ray"
    n_trials: int = 10
    direction: str = "minimize"  # minimize WER
    metric: str = "eval_wer"
    space: Optional[Dict[str, Any]] = None  # A dict describing search space; see notes below.


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator for dynamic padding of inputs and labels for seq2seq speech models.

    Core behaviors:
    - Separate padding strategies for encoder inputs and decoder labels.
    - Replace pad tokens with -100 for labels so they can be ignored by the loss.
    - If BOS was previously added, remove it here (HF Trainer handles BOS).
    - Optionally forward attention_mask for SpecAugment-ready models (e.g., Whisper).
    """
    processor: Any
    decoder_start_token_id: int
    forward_attention_mask: bool

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        model_input_name = self.processor.model_input_names[0]
        input_features = [{model_input_name: f[model_input_name]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        if self.forward_attention_mask:
            batch["attention_mask"] = torch.LongTensor([f["attention_mask"] for f in features])

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Drop BOS if already present; Trainer/Model will re-add it as needed
        if labels.size(1) > 0 and (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def _setup_logging(verbose: bool) -> logging.Logger:
    log_level = logging.INFO if verbose else logging.WARN
    if RichHandler is not None and verbose:
        # Pretty, rich logging for dev ergonomics
        logging.basicConfig(
            level=log_level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True, markup=True)],
        )
        if callable(rich_install):
            rich_install(show_locals=False, suppress=["torch", "transformers", "datasets"])
    else:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
    return logging.getLogger("seq2seq_asr")


def _read_yaml(yaml_path: str) -> Dict[str, Any]:
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"YAML config not found at: {yaml_path}")

    if _yaml_loader is None:
        raise RuntimeError(
            f"Cannot parse YAML: no YAML library found. Tried ruamel.yaml and PyYAML. Errors: {_YAML_ERR}"
        )

    if _yaml_loader == "ruamel":
        yaml_parser = YAML(typ="safe")
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml_parser.load(f) or {}
    else:
        # PyYAML
        import yaml  # type: ignore
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}


def _filter_kwargs(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a shallow copy of d with None values removed."""
    if not d:
        return {}
    return {k: v for k, v in d.items() if v is not None}


def _maybe_table(title: str, data: Dict[str, Any], show: bool) -> None:
    if not show:
        return
    try:
        if Table is None:
            rprint(f"[bold cyan]{title}[/bold cyan]")
            rprint(json.dumps(data, indent=2, ensure_ascii=False))
            return
        table = Table(title=title, show_lines=False)
        table.add_column("Key", style="bold cyan", no_wrap=True)
        table.add_column("Value", style="white")
        for k, v in data.items():
            table.add_row(str(k), json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v))
        console = Console()
        console.print(table)
    except Exception:
        # Fallback printing if rich is faulty
        rprint(f"[bold cyan]{title}[/bold cyan]")
        rprint(json.dumps(data, indent=2, ensure_ascii=False))


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _build_args_from_yaml(cfg: Dict[str, Any]) -> Tuple[ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, RuntimeArguments, HPOArguments]:
    # Pull sub-dicts (or {})
    model_cfg = _filter_kwargs(cfg.get("model_args", {}))
    data_cfg = _filter_kwargs(cfg.get("data_args", {}))
    train_cfg = _filter_kwargs(cfg.get("training_args", {}))
    runtime_cfg = _filter_kwargs(cfg.get("runtime", {}))
    hpo_cfg = _filter_kwargs(cfg.get("hpo", {}))

    # Validate mandatory keys for HF Args
    if "model_name_or_path" not in model_cfg:
        raise ValueError("model_args.model_name_or_path is required in the YAML.")
    if "output_dir" not in train_cfg:
        raise ValueError("training_args.output_dir is required in the YAML.")
    if "dataset_name" not in data_cfg:
        raise ValueError("data_args.dataset_name is required in the YAML.")

    # Parse into HF dataclasses
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_dict({
        **model_cfg,
        **{f"data_{k}": v for k, v in {}.items()},  # reserved if needed
    }, allow_extra_keys=True)  # allow extras to avoid surprises

    # HfArgumentParser cannot parse nested dicts directly, so we parse them separately
    # Approach: parse_dict supports mapping by dataclass field names, so provide each dict individually.
    model_args = parser.parse_dict(model_cfg, allow_extra_keys=True)[0]  # type: ignore
    data_args = parser.parse_dict(data_cfg, allow_extra_keys=True)[1]   # type: ignore
    training_args = parser.parse_dict(train_cfg, allow_extra_keys=True)[2]  # type: ignore

    runtime_args = RuntimeArguments(**runtime_cfg) if runtime_cfg else RuntimeArguments()
    hpo_args = HPOArguments(**hpo_cfg) if hpo_cfg else HPOArguments()

    return model_args, data_args, training_args, runtime_args, hpo_args


def _detect_processing_param_name() -> str:
    """
    Transformers evolving API: Seq2SeqTrainer recently added 'processing_class' but historically used 'tokenizer'.
    We'll detect and return the appropriate kwarg name to pass processor/feature_extractor.
    """
    sig = inspect.signature(Seq2SeqTrainer.__init__)
    if "processing_class" in sig.parameters:
        return "processing_class"
    return "tokenizer"


def _ensure_language_task_for_multilingual(model, tokenizer, data_args: DataTrainingArguments):
    is_multilingual = bool(_safe_getattr(model, "generation_config", None) and getattr(model.generation_config, "is_multilingual", False))
    if is_multilingual:
        # Whisper-style multilingual: enforce language and task presence
        if data_args.language is None:
            raise ValueError("Multilingual checkpoint detected. Please set data_args.language in the YAML.")
        if data_args.task not in {"transcribe", "translate"}:
            raise ValueError("data_args.task must be either 'transcribe' or 'translate' for multilingual checkpoints.")
        # Note: WhisperTokenizer provides set_prefix_tokens; safely check for it
        if hasattr(tokenizer, "set_prefix_tokens"):
            tokenizer.set_prefix_tokens(language=data_args.language, task=data_args.task)
        model.generation_config.language = data_args.language
        model.generation_config.task = data_args.task
    else:
        if data_args.language is not None:
            raise ValueError(
                "Setting language for an English-only checkpoint is not allowed. "
                "Remove data_args.language unless you're fine-tuning a multilingual checkpoint."
            )


def _maybe_add_early_stopping(trainer: Seq2SeqTrainer, patience: Optional[int]) -> None:
    if patience is None:
        return
    try:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=patience))
        logger.info(f"Early stopping enabled with patience={patience}.")
    except Exception as e:
        logger.warning(f"Could not add EarlyStoppingCallback: {e}")


def _maybe_run_hpo(
    trainer: Seq2SeqTrainer,
    hpo_args: HPOArguments,
    training_args: Seq2SeqTrainingArguments,
    verbose: bool,
) -> Optional[Dict[str, Any]]:
    """
    Optional hyperparameter optimization.
    We support Optuna by default; if 'ray' is set, we try Ray Tune if available.

    Search space specification (hpo.space):
    - For numeric continuous: {"low": float, "high": float, "log": bool}
    - For categorical discrete: {"values": [ ... ]}
    Example:
      space:
        learning_rate: {low: 1.0e-5, high: 5.0e-4, log: true}
        per_device_train_batch_size: {values: [4, 8, 16]}
    """
    if not hpo_args.enabled:
        return None

    metric = hpo_args.metric
    direction = hpo_args.direction.lower()
    n_trials = hpo_args.n_trials
    space = hpo_args.space or {}

    # Validate metric
    if not metric:
        raise ValueError("hpo.metric must be provided when HPO is enabled (e.g., 'eval_wer').")

    # Detect backend
    backend = hpo_args.backend.lower()
    if backend not in {"optuna", "ray"}:
        warnings.warn(f"Unsupported HPO backend '{backend}'. Falling back to Optuna.")
        backend = "optuna"

    # Define search space functions depending on backend
    if backend == "optuna":
        try:
            import optuna  # noqa: F401
        except Exception as e:
            raise RuntimeError(f"Optuna not installed but 'optuna' backend selected: {e}") from e

        def hp_space_optuna(trial):
            hp = {}
            for name, spec in space.items():
                if "values" in spec:
                    hp[name] = trial.suggest_categorical(name, spec["values"])
                else:
                    low = float(spec["low"])
                    high = float(spec["high"])
                    log = bool(spec.get("log", False))
                    if log:
                        hp[name] = trial.suggest_float(name, low, high, log=True)
                    else:
                        hp[name] = trial.suggest_float(name, low, high)
            return hp

        logger.info(f"Starting Optuna HPO: n_trials={n_trials}, direction={direction}, metric={metric}")
        best_run = trainer.hyperparameter_search(
            direction=direction,
            backend="optuna",
            hp_space=hp_space_optuna,
            n_trials=n_trials,
            compute_objective=lambda metrics: metrics.get(metric, float("inf")),
        )
        # Apply best params to trainer args
        trainer.args = trainer.args.copy(update=best_run.hyperparameters)
        logger.info(f"Best hyperparameters: {best_run.hyperparameters}")
        return {"best_run": best_run.hyperparameters, "best_score": best_run.objective}

    # Ray Tune
    try:
        import ray  # noqa: F401
        from ray import tune  # noqa: F401
    except Exception as e:
        raise RuntimeError(f"Ray Tune not installed but 'ray' backend selected: {e}") from e

    def hp_space_ray(_trial):
        hp = {}
        for name, spec in space.items():
            if "values" in spec:
                hp[name] = tune.choice(spec["values"])  # type: ignore
            else:
                low = float(spec["low"])
                high = float(spec["high"])
                log = bool(spec.get("log", False))
                if log:
                    # Ray doesn't have log float by default; emulate with sampling in log space
                    hp[name] = tune.loguniform(low, high)  # type: ignore
                else:
                    hp[name] = tune.uniform(low, high)  # type: ignore
        return hp

    logger.info(f"Starting Ray Tune HPO: n_trials={n_trials}, direction={direction}, metric={metric}")
    best_run = trainer.hyperparameter_search(
        direction=direction,
        backend="ray",
        hp_space=hp_space_ray,
        n_trials=n_trials,
        compute_objective=lambda metrics: metrics.get(metric, float("inf")),
    )
    trainer.args = trainer.args.copy(update=best_run.hyperparameters)
    logger.info(f"Best hyperparameters: {best_run.hyperparameters}")
    return {"best_run": best_run.hyperparameters, "best_score": best_run.objective}


def main() -> Optional[Dict[str, Any]]:
    # ----------------------------------------------------------------------------------
    # 0) YAML-only input contract
    # ----------------------------------------------------------------------------------
    if not (len(sys.argv) == 2 and sys.argv[1].lower().endswith((".yaml", ".yml"))):
        print(
            "Usage: python sequence_to_sequence_speech_recognition.py path/to/config.yaml",
            file=sys.stderr,
        )
        sys.exit(2)

    cfg_path = sys.argv[1]
    cfg = _read_yaml(cfg_path)

    # ----------------------------------------------------------------------------------
    # 1) Parse args from YAML (strict, explicit)
    # ----------------------------------------------------------------------------------
    model_args, data_args, training_args, runtime_args, hpo_args = _build_args_from_yaml(cfg)

    # ----------------------------------------------------------------------------------
    # 2) Logging & telemetry
    # ----------------------------------------------------------------------------------
    global logger
    logger = _setup_logging(verbose=bool(runtime_args.verbose))
    if runtime_args.verbose:
        _maybe_table("Model Arguments", asdict(model_args), show=True)
        _maybe_table("Data Arguments", asdict(data_args), show=True)
        _maybe_table("Training Arguments (key subset)", {
            "output_dir": training_args.output_dir,
            "do_train": training_args.do_train,
            "do_eval": training_args.do_eval,
            "do_predict": training_args.do_predict,
            "learning_rate": training_args.learning_rate,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "per_device_eval_batch_size": training_args.per_device_eval_batch_size,
            "num_train_epochs": training_args.num_train_epochs,
            "fp16": training_args.fp16,
            "evaluation_strategy": training_args.evaluation_strategy,
            "save_strategy": training_args.save_strategy,
            "predict_with_generate": training_args.predict_with_generate,
            "load_best_model_at_end": training_args.load_best_model_at_end,
            "metric_for_best_model": training_args.metric_for_best_model,
            "greater_is_better": training_args.greater_is_better,
            "seed": training_args.seed,
        }, show=True)
        _maybe_table("Runtime", asdict(runtime_args), show=True)
        _maybe_table("HPO", asdict(hpo_args), show=True)

    # Track usage (HF telemetry) - non-invasive
    try:
        send_example_telemetry("sequence_to_sequence_speech_recognition", model_args, data_args)
    except Exception:
        pass

    # Adjust framework loggers
    datasets.utils.logging.set_verbosity(training_args.get_process_log_level())
    transformers.utils.logging.set_verbosity(training_args.get_process_log_level())
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # ----------------------------------------------------------------------------------
    # 3) Check for existing checkpoints for resume
    # ----------------------------------------------------------------------------------
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) exists and is not empty. "
                "Set training_args.overwrite_output_dir=true to train from scratch."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Detected checkpoint at {last_checkpoint}. Training will resume from this checkpoint.")

    # ----------------------------------------------------------------------------------
    # 4) Seed for reproducibility
    # ----------------------------------------------------------------------------------
    if training_args.seed is not None:
        set_seed(training_args.seed)

    # ----------------------------------------------------------------------------------
    # 5) Load dataset(s) dynamically (pass only what exists)
    # ----------------------------------------------------------------------------------
    raw_datasets: DatasetDict = DatasetDict()

    def _load_split(split_name: str):
        # Respect "only pass if provided" ethos
        ds_kwargs = {
            "cache_dir": model_args.cache_dir,
            "token": model_args.token,
            "trust_remote_code": model_args.trust_remote_code,
            "split": split_name,
        }
        ds_kwargs = _filter_kwargs(ds_kwargs)
        # load_dataset requires a path + optional config
        if data_args.dataset_config_name is not None:
            return load_dataset(data_args.dataset_name, data_args.dataset_config_name, **ds_kwargs)
        return load_dataset(data_args.dataset_name, **ds_kwargs)

    if training_args.do_train:
        raw_datasets["train"] = _load_split(data_args.train_split_name)
    if training_args.do_eval:
        raw_datasets["eval"] = _load_split(data_args.eval_split_name)

    # Validate columns
    sample_columns = next(iter(raw_datasets.values())).column_names
    if data_args.audio_column_name not in sample_columns:
        raise ValueError(
            f"audio_column_name='{data_args.audio_column_name}' not in dataset columns: {sample_columns}"
        )
    if data_args.text_column_name not in sample_columns:
        raise ValueError(
            f"text_column_name='{data_args.text_column_name}' not in dataset columns: {sample_columns}"
        )

    # ----------------------------------------------------------------------------------
    # 6) Load pretrained model, tokenizer, feature extractor (pass only what exists)
    # ----------------------------------------------------------------------------------
    common_kwargs = _filter_kwargs({
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    })

    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        **common_kwargs,
    )
    # SpecAugment toggle for whisper-like models
    if getattr(config, "model_type", None) == "whisper":
        config.update({"apply_spec_augment": model_args.apply_spec_augment})

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name or model_args.model_name_or_path,
        **common_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        **common_kwargs,
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        **common_kwargs,
    )

    if _safe_getattr(model.config, "decoder_start_token_id", None) is None:
        raise ValueError("Model config requires a valid decoder_start_token_id.")

    if model_args.freeze_feature_encoder:
        if hasattr(model, "freeze_feature_encoder"):
            model.freeze_feature_encoder()
    if model_args.freeze_encoder:
        if hasattr(model, "freeze_encoder"):
            model.freeze_encoder()
        # Avoid gradient checkpointing on a frozen encoder for safety
        try:
            model.model.encoder.gradient_checkpointing = False  # type: ignore
        except Exception:
            pass

    # Language/task setup for multilingual checkpoints
    _ensure_language_task_for_multilingual(model, tokenizer, data_args)

    # Legacy/deprecated forced ids and suppress tokens (if explicitly provided)
    if model_args.forced_decoder_ids is not None:
        logger.warning(
            "Using forced_decoder_ids from YAML. This is deprecated; prefer setting language/task where relevant."
        )
        model.generation_config.forced_decoder_ids = model_args.forced_decoder_ids
    else:
        model.generation_config.forced_decoder_ids = None
        model.config.forced_decoder_ids = None

    if model_args.suppress_tokens is not None:
        logger.warning(
            "Using suppress_tokens from YAML. Consider updating training script if this is necessary."
        )
        model.generation_config.suppress_tokens = model_args.suppress_tokens

    # ----------------------------------------------------------------------------------
    # 7) Resample dataset if needed
    # ----------------------------------------------------------------------------------
    dataset_sampling_rate = next(iter(raw_datasets.values())).features[data_args.audio_column_name].sampling_rate
    if dataset_sampling_rate != feature_extractor.sampling_rate:
        raw_datasets = raw_datasets.cast_column(
            data_args.audio_column_name, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
        )

    # ----------------------------------------------------------------------------------
    # 8) Preprocess datasets
    # ----------------------------------------------------------------------------------
    max_input_length = int(data_args.max_duration_in_seconds * feature_extractor.sampling_rate)
    min_input_length = int(data_args.min_duration_in_seconds * feature_extractor.sampling_rate)

    audio_column = data_args.audio_column_name
    text_column = data_args.text_column_name
    model_input_name = feature_extractor.model_input_names[0]
    do_lower_case = bool(data_args.do_lower_case)

    # If SpecAugment is active on Whisper, we want to forward attention_mask along time axis
    forward_attention_mask = (
        getattr(config, "model_type", None) == "whisper"
        and getattr(config, "apply_spec_augment", False)
        and getattr(config, "mask_time_prob", 0.0) > 0.0
    )

    if data_args.max_train_samples is not None and "train" in raw_datasets:
        raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))
    if data_args.max_eval_samples is not None and "eval" in raw_datasets:
        raw_datasets["eval"] = raw_datasets["eval"].select(range(data_args.max_eval_samples))

    def prepare_dataset(batch: Dict[str, Any]) -> Dict[str, Any]:
        sample = batch[audio_column]
        inputs = feature_extractor(
            sample["array"], sampling_rate=sample["sampling_rate"], return_attention_mask=forward_attention_mask
        )
        batch[model_input_name] = inputs.get(model_input_name)[0]
        batch["input_length"] = len(sample["array"])
        if forward_attention_mask:
            batch["attention_mask"] = inputs.get("attention_mask")[0]

        text = batch[text_column]
        text = text.lower() if do_lower_case and isinstance(text, str) else text
        batch["labels"] = tokenizer(text).input_ids
        return batch

    with training_args.main_process_first(desc="dataset map pre-processing"):
        vectorized_datasets = raw_datasets.map(
            prepare_dataset,
            remove_columns=next(iter(raw_datasets.values())).column_names,
            num_proc=data_args.preprocessing_num_workers,
            desc="Preprocessing dataset",
        )

    def is_audio_in_length_range(length: int) -> bool:
        return (length > min_input_length) and (length < max_input_length)

    vectorized_datasets = vectorized_datasets.filter(
        is_audio_in_length_range,
        num_proc=data_args.preprocessing_num_workers,
        input_columns=["input_length"],
    )

    if data_args.preprocessing_only:
        # Exit early after caching
        cache_paths = {k: v.cache_files for k, v in vectorized_datasets.items()}
        logger.info(f"Preprocessing-only mode: cached files -> {cache_paths}")
        return {"cache_files": cache_paths}

    # ----------------------------------------------------------------------------------
    # 9) Metrics
    # ----------------------------------------------------------------------------------
    wer_metric = evaluate.load("wer", cache_dir=model_args.cache_dir)
    cer_metric = None
    try:
        cer_metric = evaluate.load("cer", cache_dir=model_args.cache_dir)
    except Exception:
        cer_metric = None  # Optional

    def compute_metrics(pred) -> Dict[str, float]:
        pred_ids = pred.predictions
        # Unmask labels (-100 -> tokenizer.pad_token_id) for decoding
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = float(wer_metric.compute(predictions=pred_str, references=label_str))
        results = {"wer": wer}
        if cer_metric is not None:
            try:
                cer = float(cer_metric.compute(predictions=pred_str, references=label_str))
                results["cer"] = cer
            except Exception:
                pass
        return results

    # ----------------------------------------------------------------------------------
    # 10) Save processor parts (ensures downstream loading consistency)
    # ----------------------------------------------------------------------------------
    with training_args.main_process_first():
        if is_main_process(training_args.local_rank):
            feature_extractor.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            config.save_pretrained(training_args.output_dir)

    processor = AutoProcessor.from_pretrained(training_args.output_dir)

    # ----------------------------------------------------------------------------------
    # 11) Data collator + Trainer
    # ----------------------------------------------------------------------------------
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        forward_attention_mask=forward_attention_mask,
    )

    processing_kwarg_name = _detect_processing_param_name()
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=vectorized_datasets.get("train"),
        eval_dataset=vectorized_datasets.get("eval"),
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )
    # Respect Trainer signature differences
    if processing_kwarg_name == "processing_class":
        trainer_kwargs["processing_class"] = feature_extractor
    else:
        trainer_kwargs["tokenizer"] = processor  # historical API expects 'tokenizer'

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    # Optional early stopping
    _maybe_add_early_stopping(trainer, runtime_args.early_stopping_patience)

    # ----------------------------------------------------------------------------------
    # 12) Optional HPO
    # ----------------------------------------------------------------------------------
    hpo_summary = _maybe_run_hpo(trainer, hpo_args, training_args, runtime_args.verbose)

    # ----------------------------------------------------------------------------------
    # 13) Train
    # ----------------------------------------------------------------------------------
    results: Dict[str, Any] = {}
    if training_args.do_train:
        resume_ckpt = training_args.resume_from_checkpoint or last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
        trainer.save_model()  # also saves processor components

        metrics = train_result.metrics
        train_count = len(vectorized_datasets["train"]) if "train" in vectorized_datasets else 0
        if data_args.max_train_samples is not None:
            train_count = min(train_count, data_args.max_train_samples)
        metrics["train_samples"] = train_count

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        results.update({"train_metrics": metrics})

    # ----------------------------------------------------------------------------------
    # 14) Evaluate
    # ----------------------------------------------------------------------------------
    if training_args.do_eval and "eval" in vectorized_datasets:
        logger.info("*** Evaluate ***")
        eval_metrics = trainer.evaluate(
            metric_key_prefix="eval",
            max_length=training_args.generation_max_length,
            num_beams=training_args.generation_num_beams,
        )
        eval_count = len(vectorized_datasets["eval"])
        if data_args.max_eval_samples is not None:
            eval_count = min(eval_count, data_args.max_eval_samples)
        eval_metrics["eval_samples"] = eval_count

        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        results.update({"eval_metrics": eval_metrics})

    # ----------------------------------------------------------------------------------
    # 15) Predict (optional)
    # ----------------------------------------------------------------------------------
    if training_args.do_predict and "eval" in vectorized_datasets:
        logger.info("*** Predict on eval dataset ***")
        predictions = trainer.predict(
            test_dataset=vectorized_datasets["eval"],
            metric_key_prefix="predict",
            max_length=training_args.generation_max_length,
            num_beams=training_args.generation_num_beams,
        )
        pred_ids = predictions.predictions
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        # Save predictions if asked
        infer_cfg = cfg.get("inference", {}) or {}
        if infer_cfg.get("save_predictions", True):
            fname = infer_cfg.get("predictions_filename", "predictions.txt")
            save_path = os.path.join(training_args.output_dir, fname)
            with open(save_path, "w", encoding="utf-8") as f:
                for line in pred_str:
                    f.write(line.strip() + "\n")
            logger.info(f"Saved predictions to: {save_path}")

        # Save metrics
        if predictions.metrics:
            trainer.log_metrics("predict", predictions.metrics)
            trainer.save_metrics("predict", predictions.metrics)
            results.update({"predict_metrics": predictions.metrics})

    # ----------------------------------------------------------------------------------
    # 16) Model card / hub
    # ----------------------------------------------------------------------------------
    card_kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "automatic-speech-recognition"}
    if data_args.dataset_name is not None:
        card_kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            card_kwargs["dataset_args"] = data_args.dataset_config_name
            card_kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            card_kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**card_kwargs)
    else:
        trainer.create_model_card(**card_kwargs)

    # ----------------------------------------------------------------------------------
    # 17) Summary
    # ----------------------------------------------------------------------------------
    summary = {
        "output_dir": training_args.output_dir,
        "hpo": hpo_summary,
        **results,
    }
    if runtime_args.verbose:
        _maybe_table("Run Summary", summary, show=True)
    return summary


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.error("Interrupted by user. Exiting gracefully.")
        sys.exit(130)
    except Exception as e:
        # Pretty traceback if Rich is available
        if Console is not None:
            console = Console()
            console.print_exception()
        else:
            raise
