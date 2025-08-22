```python

YAML CONFIG EXAMPLES
================================================================================
1) Minimal training on an HF dataset (superb/ks: keyword spotting)

pipeline:
  mode: train
  verbose: true            # uses rich for beautiful logs
  seed: 42
  output_dir: ./outputs/wav2vec2-superb-ks
  cache_dir: null
  use_rich: true
  trust_remote_code: false
  token: null              # or set HF token here if needed

data:
  input_format: hf         # 'hf' | 'csv' | 'json' | 'folders'
  dataset_name: superb
  dataset_config_name: ks
  train_split_name: train
  eval_split_name: validation
  audio_column_name: audio
  label_column_name: label
  sampling_rate: null      # if null, auto from feature extractor
  max_length_seconds: 5.0  # random crop length during training
  max_train_samples: null
  max_eval_samples: null
  normalization: "peak"    # 'off' | 'peak' | 'rms'
  augment:
    time_shift_max_ratio: 0.1     # fraction of audio length
    noise_prob: 0.2               # 0..1 probability to add noise
    noise_snr_db: [15, 30]        # SNR range for injected white noise

model:
  model_name_or_path: facebook/wav2vec2-base
  config_name: null
  feature_extractor_name: null
  freeze_feature_encoder: true
  attention_mask: true
  ignore_mismatched_sizes: false
  cache_dir: null
  model_revision: main
  trust_remote_code: false
  token: null

training:
  # These are TrainingArguments-compatible keys
  learning_rate: 3.0e-5
  weight_decay: 0.01
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 16
  gradient_accumulation_steps: 1
  num_train_epochs: 10
  warmup_ratio: 0.1
  evaluation_strategy: "epoch"
  save_strategy: "epoch"
  logging_strategy: "steps"
  logging_steps: 50
  report_to: []  # [] or ["wandb"] or ["tensorboard"]
  fp16: true
  bf16: false
  gradient_checkpointing: false
  lr_scheduler_type: "linear"
  load_best_model_at_end: true
  metric_for_best_model: "accuracy"
  greater_is_better: true
  save_total_limit: 2
  dataloader_num_workers: 4
  remove_unused_columns: false
  optim: "adamw_torch"
  label_smoothing_factor: 0.0
  # Optional: resume_from_checkpoint: "path/to/checkpoint"

early_stopping:
  enabled: true
  patience: 3
  threshold: 0.0

evaluation:
  compute_accuracy: true
  compute_f1: true
  compute_precision: true
  compute_recall: true
  f_average: "macro"    # 'micro' | 'macro' | 'weighted'
  top_k: [3, 5]
  confusion_matrix: true
  classification_report: true

hpo:
  enabled: false
  backend: "optuna"      # 'optuna' | 'ray' (if available)
  direction: "maximize"
  n_trials: 10
  timeout: null
  # A simple search space (if enabled)
  search_space:
    learning_rate: {low: 1.0e-5, high: 5.0e-4, log: true}
    weight_decay: {low: 0.0, high: 0.1, step: 0.01}
    per_device_train_batch_size: {choices: [8, 16, 32]}
    num_train_epochs: {low: 3, high: 12, step: 3}

cross_validation:
  n_splits: 1            # >1 enables Stratified K-Fold
  shuffle: true
  stratified: true
  seed: 42

predict:
  files: []              # list of file paths OR:
  folder: null           # a directory to scan (recursively)
  pattern: "*.wav"
  recursive: true
  output_csv: ./outputs/predictions.csv

deploy:
  push_to_hub: false
  hub_model_id: null
  hub_private_repo: false
  token: null
  # Always writes local model card; pushing is optional.


2) Training from local CSV files (two-column CSV: audio,label)
- audio: path to file
- label: class name (string)

pipeline:
  mode: train
  verbose: true
  output_dir: ./outputs/local-csv
  seed: 1337

data:
  input_format: csv
  train_file: ./data/train.csv
  eval_file: ./data/valid.csv
  audio_column_name: audio
  label_column_name: label
  normalization: "rms"
  augment:
    time_shift_max_ratio: 0.05
    noise_prob: 0.1
    noise_snr_db: [10, 20]

model:
  model_name_or_path: microsoft/wavlm-base
training:
  per_device_train_batch_size: 8
  per_device_eval_batch_size: 8
  num_train_epochs: 20
  evaluation_strategy: "steps"
  save_strategy: "steps"
  save_steps: 500
  logging_steps: 50
  load_best_model_at_end: true
  metric_for_best_model: "f1"
  greater_is_better: true
  learning_rate: 2.0e-5
  weight_decay: 0.01


3) Hyperparameter Optimization on HF dataset (Optuna)
pipeline: { mode: train, verbose: true, output_dir: ./outputs/hpo }
data: { input_format: hf, dataset_name: superb, dataset_config_name: ks }
model: { model_name_or_path: facebook/wav2vec2-base }
training: { num_train_epochs: 10, per_device_train_batch_size: 16, per_device_eval_batch_size: 16 }
hpo:
  enabled: true
  direction: "maximize"
  n_trials: 15
  search_space:
    learning_rate: {low: 1.0e-5, high: 5.0e-4, log: true}
    weight_decay: {low: 0.0, high: 0.1, step: 0.02}
    warmup_ratio: {low: 0.0, high: 0.2, step: 0.05}


4) Predict-only (after training)
pipeline: { mode: predict, verbose: true, output_dir: ./outputs/wav2vec2-superb-ks }
predict:
  folder: ./samples
  pattern: "*.wav"
  recursive: true
  output_csv: ./outputs/predictions.csv
================================================================================

```