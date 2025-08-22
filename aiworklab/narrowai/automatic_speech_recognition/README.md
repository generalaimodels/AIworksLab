```python

Example 1: Whisper-small on Common Voice (English), train+eval with best-model saving
-------------------------------------------------------------------------------------
model:
  model_name_or_path: openai/whisper-small
  cache_dir: null
  use_fast_tokenizer: true
  model_revision: main
  trust_remote_code: false
  token: null
  freeze_feature_encoder: true
  freeze_encoder: false
  forced_decoder_ids: null
  suppress_tokens: null
  apply_spec_augment: true

data:
  dataset_name: mozilla-foundation/common_voice_11_0
  dataset_config_name: en
  train_split_name: train+validation
  eval_split_name: test
  audio_column_name: audio
  text_column_name: sentence
  preprocessing_num_workers: 4
  max_train_samples: null
  max_eval_samples: null
  max_duration_in_seconds: 30.0
  min_duration_in_seconds: 0.5
  preprocessing_only: false
  do_lower_case: true
  language: en
  task: transcribe

training:
  output_dir: ./checkpoints/whisper-small-en
  overwrite_output_dir: false
  do_train: true
  do_eval: true
  do_predict: false
  per_device_train_batch_size: 8
  per_device_eval_batch_size: 8
  gradient_accumulation_steps: 1
  evaluation_strategy: epoch
  save_strategy: epoch
  logging_strategy: steps
  logging_steps: 25
  num_train_epochs: 3
  learning_rate: 1.0e-4
  lr_scheduler_type: cosine_with_restarts
  warmup_ratio: 0.1
  weight_decay: 0.0
  predict_with_generate: true
  generation_max_length: 225
  generation_num_beams: 1
  fp16: false
  bf16: true
  load_best_model_at_end: true
  metric_for_best_model: wer
  greater_is_better: false
  save_total_limit: 2
  seed: 42
  report_to: ["tensorboard"]
  dataloader_num_workers: 4
  remove_unused_columns: true
  gradient_checkpointing: false
  resume_from_checkpoint: null
  push_to_hub: false

pipeline:
  verbose: true
  export_torchscript: false
  export_onnx: false
  inference_examples:
    - ./samples/sample1.wav
    - ./samples/sample2.flac
  hpo:
    enabled: false
    n_trials: 10
    direction: minimize
    timeout: null
    seed: 71
    search_space:
      learning_rate:
        type: float
        low: 1.0e-5
        high: 5.0e-4
        log: true
      num_train_epochs:
        type: int
        low: 2
        high: 4
      per_device_train_batch_size:
        type: categorical
        choices: [4, 8, 16]

Example 2: Wav2Vec2 on LibriSpeech ASR, evaluation only on a subset
--------------------------------------------------------------------
model:
  model_name_or_path: facebook/wav2vec2-base-960h
  cache_dir: null
  use_fast_tokenizer: true
  model_revision: main
  trust_remote_code: false
  token: null
  freeze_feature_encoder: true
  freeze_encoder: false
  forced_decoder_ids: null
  suppress_tokens: null
  apply_spec_augment: false

data:
  dataset_name: librispeech_asr
  dataset_config_name: clean
  train_split_name: train.clean.100
  eval_split_name: validation.clean
  audio_column_name: audio
  text_column_name: text
  preprocessing_num_workers: 4
  max_train_samples: 1000
  max_eval_samples: 500
  max_duration_in_seconds: 20.0
  min_duration_in_seconds: 0.0
  preprocessing_only: false
  do_lower_case: true
  language: null
  task: transcribe

training:
  output_dir: ./checkpoints/w2v2-librispeech
  overwrite_output_dir: false
  do_train: false
  do_eval: true
  do_predict: false
  per_device_train_batch_size: 8
  per_device_eval_batch_size: 8
  gradient_accumulation_steps: 1
  evaluation_strategy: epoch
  save_strategy: epoch
  logging_strategy: steps
  logging_steps: 25
  num_train_epochs: 1
  learning_rate: 5.0e-5
  lr_scheduler_type: linear
  warmup_ratio: 0.0
  weight_decay: 0.0
  predict_with_generate: true
  generation_max_length: 256
  generation_num_beams: 1
  fp16: false
  bf16: true
  load_best_model_at_end: false
  metric_for_best_model: wer
  greater_is_better: false
  save_total_limit: 1
  seed: 42
  report_to: ["none"]
  dataloader_num_workers: 4
  remove_unused_columns: true
  gradient_checkpointing: false
  resume_from_checkpoint: null
  push_to_hub: false

pipeline:
  verbose: true
  export_torchscript: false
  export_onnx: false
  inference_examples: []
  hpo:
    enabled: false
    n_trials: 5
    direction: minimize
    timeout: null
    seed: 11
    search_space: {}

Example 3: Hyperparameter Optimization enabled (Optuna)
------------------------------------------------------
...
pipeline:
  verbose: true
  hpo:
    enabled: true
    n_trials: 20
    direction: minimize
    seed: 123
    search_space:
      learning_rate:
        type: float
        low: 1.0e-5
        high: 2.0e-4
        log: true
      per_device_train_batch_size:
        type: categorical
        choices: [8, 16]
      num_train_epochs:
        type: int
        low: 2
        high: 5


```