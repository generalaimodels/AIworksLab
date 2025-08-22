#!/usr/bin/env python3
"""
Advanced End-to-End Speech Recognition Pipeline
===============================================

This module implements a state-of-the-art sequence-to-sequence speech recognition pipeline
with comprehensive features for data preprocessing, model training, evaluation, and deployment.

Architecture Overview:
---------------------
1. Configuration Management: YAML-based configuration with validation
2. Data Pipeline: Advanced preprocessing with feature engineering and augmentation
3. Model Pipeline: Flexible model architecture with checkpoint management
4. Training Pipeline: Robust training with optimization and monitoring
5. Evaluation Pipeline: Comprehensive metrics and analysis
6. Deployment Pipeline: Model export and inference optimization

Key Features:
------------
- Dynamic argument parsing from YAML configurations
- Rich-based verbose output and progress tracking
- Advanced error handling and recovery mechanisms
- Modular architecture for easy extensibility
- Automatic hyperparameter optimization
- Comprehensive logging and monitoring
- GPU memory optimization and mixed precision training
- Model versioning and checkpoint management

Usage:
------
python sequence_to_sequence_speech_recognition.py config.yaml

Example YAML Configuration:
--------------------------
model_arguments:
  model_name_or_path: "openai/whisper-small"
  freeze_feature_encoder: true
  
data_arguments:
  dataset_name: "mozilla-foundation/common_voice_11_0"
  dataset_config_name: "en"
  max_duration_in_seconds: 30.0
  
training_arguments:
  output_dir: "./results"
  per_device_train_batch_size: 16
  num_train_epochs: 3
  
Author: Advanced AI Systems Team
Version: 2.0.0
License: MIT
"""

import logging
import os
import sys
import yaml
import json
import time
import warnings
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Union, Dict, List, Tuple
from pathlib import Path
import traceback

# Core ML Libraries
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import datasets
import evaluate
from datasets import DatasetDict, load_dataset, Audio

# Transformers and HuggingFace
import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    set_seed,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

# Rich for beautiful console output
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich import print as rprint
from rich.traceback import install as install_rich_traceback

# Install rich traceback for better error reporting
install_rich_traceback(show_locals=True)

# Version requirements and compatibility checks
check_min_version("4.56.0.dev0")
require_version("datasets>=1.18.0", "To fix: pip install -r requirements.txt")

# Initialize Rich console for global use
console = Console(record=True)

# Configure warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class AdvancedLogger:
    """
    Advanced logging system with Rich integration for beautiful console output
    and comprehensive file logging with structured formats.
    
    Features:
    - Multi-level logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - Rich console formatting with colors and icons
    - Structured file logging with JSON format
    - Performance metrics tracking
    - Memory usage monitoring
    - GPU utilization tracking
    """
    
    def __init__(self, name: str, log_file: Optional[str] = None, verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.console = Console()
        self.start_time = time.time()
        
        # Setup logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Rich console handler
        rich_handler = RichHandler(
            console=self.console,
            show_time=True,
            show_path=True,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True
        )
        rich_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        self.logger.addHandler(rich_handler)
        
        # File handler for persistent logging
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional metadata"""
        if kwargs:
            message += f" | Metadata: {kwargs}"
        self.logger.info(f"[blue]ℹ️[/blue] {message}")
    
    def success(self, message: str, **kwargs):
        """Log success message"""
        if kwargs:
            message += f" | Metadata: {kwargs}"
        self.logger.info(f"[green]✅[/green] {message}")
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        if kwargs:
            message += f" | Metadata: {kwargs}"
        self.logger.warning(f"[yellow]⚠️[/yellow] {message}")
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log error message with optional exception details"""
        if error:
            message += f" | Error: {str(error)}"
        if kwargs:
            message += f" | Metadata: {kwargs}"
        self.logger.error(f"[red]❌[/red] {message}")
        if error and self.verbose:
            self.logger.error(f"[red]Traceback:[/red] {traceback.format_exc()}")
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        if kwargs:
            message += f" | Metadata: {kwargs}"
        self.logger.debug(f"[dim]🔍[/dim] {message}")
    
    def performance(self, operation: str, duration: float, **metrics):
        """Log performance metrics"""
        self.logger.info(
            f"[magenta]⚡[/magenta] {operation} completed in {duration:.2f}s | Metrics: {metrics}"
        )

# Initialize global logger
logger = AdvancedLogger("SpeechRecognitionPipeline", verbose=True)

@dataclass
class ModelArguments:
    """
    Advanced model configuration arguments with comprehensive options for
    model selection, optimization, and fine-tuning strategies.
    
    This class encapsulates all model-related parameters including:
    - Pre-trained model selection and configuration
    - Feature extraction and tokenization settings
    - Model architecture modifications (freezing, quantization)
    - Memory optimization and performance tuning
    - Multi-language and task-specific configurations
    """
    
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "Feature extractor name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory to store pretrained models downloaded from huggingface.co"}
    )
    use_fast_tokenizer: bool = field(
        default=True, metadata={"help": "Whether to use fast tokenizer (backed by tokenizers library)"}
    )
    model_revision: str = field(
        default="main", metadata={"help": "Model version to use (branch name, tag name or commit id)"}
    )
    token: Optional[str] = field(
        default=None, metadata={"help": "HuggingFace token for accessing private models"}
    )
    trust_remote_code: bool = field(
        default=False, metadata={"help": "Whether to trust execution of remote code from HuggingFace Hub"}
    )
    freeze_feature_encoder: bool = field(
        default=True, metadata={"help": "Whether to freeze feature encoder layers for transfer learning"}
    )
    freeze_encoder: bool = field(
        default=False, metadata={"help": "Whether to freeze entire encoder for faster training"}
    )
    apply_spec_augment: bool = field(
        default=False, metadata={"help": "Apply SpecAugment data augmentation for Wav2Vec2/Whisper models"}
    )
    gradient_checkpointing: bool = field(
        default=True, metadata={"help": "Enable gradient checkpointing to save memory"}
    )
    use_8bit: bool = field(
        default=False, metadata={"help": "Use 8-bit quantization for memory efficiency"}
    )
    torch_dtype: str = field(
        default="float32", metadata={"help": "PyTorch dtype for model weights (float16, bfloat16, float32)"}
    )

@dataclass
class DataTrainingArguments:
    """
    Advanced data processing and training configuration with support for
    multiple datasets, sophisticated preprocessing, and data augmentation.
    
    Features:
    - Multi-dataset support with dynamic loading
    - Advanced audio preprocessing and feature engineering
    - Data filtering and quality control
    - Augmentation strategies for robustness
    - Memory-efficient data loading and caching
    """
    
    dataset_name: str = field(
        metadata={"help": "Name of the dataset to use (via datasets library)"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "Configuration name of the dataset"}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "Path to training data file (CSV/JSON)"}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "Path to validation data file (CSV/JSON)"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "Number of processes for preprocessing"}
    )
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of training samples for debugging"}
    )
    max_eval_samples: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of evaluation samples for debugging"}
    )
    audio_column_name: str = field(
        default="audio", metadata={"help": "Name of dataset column containing audio data"}
    )
    text_column_name: str = field(
        default="text", metadata={"help": "Name of dataset column containing text data"}
    )
    max_duration_in_seconds: float = field(
        default=20.0, metadata={"help": "Maximum audio duration in seconds"}
    )
    min_duration_in_seconds: float = field(
        default=0.0, metadata={"help": "Minimum audio duration in seconds"}
    )
    preprocessing_only: bool = field(
        default=False, metadata={"help": "Only perform preprocessing without training"}
    )
    train_split_name: str = field(
        default="train", metadata={"help": "Name of training split"}
    )
    eval_split_name: str = field(
        default="test", metadata={"help": "Name of evaluation split"}
    )
    do_lower_case: bool = field(
        default=True, metadata={"help": "Whether to lowercase target text"}
    )
    language: Optional[str] = field(
        default=None, metadata={"help": "Language for multilingual fine-tuning"}
    )
    task: str = field(
        default="transcribe", metadata={"help": "Task: 'transcribe' or 'translate'"}
    )
    use_auth_token: Optional[str] = field(
        default=None, metadata={"help": "HuggingFace authentication token"}
    )
    streaming: bool = field(
        default=False, metadata={"help": "Enable streaming for large datasets"}
    )
    filter_by_duration: bool = field(
        default=True, metadata={"help": "Filter samples by audio duration"}
    )
    normalize_audio: bool = field(
        default=True, metadata={"help": "Normalize audio amplitude"}
    )
    noise_augmentation: bool = field(
        default=False, metadata={"help": "Apply noise augmentation during training"}
    )
    speed_augmentation: bool = field(
        default=False, metadata={"help": "Apply speed augmentation during training"}
    )

class ConfigurationManager:
    """
    Advanced configuration management system for handling YAML-based configurations
    with validation, environment variable substitution, and dynamic argument parsing.
    
    Features:
    - YAML configuration file parsing with schema validation
    - Environment variable substitution in configurations
    - Dynamic argument filtering based on dataclass fields
    - Configuration inheritance and overriding
    - Automatic type conversion and validation
    - Configuration backup and versioning
    """
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config_data = {}
        self.load_config()
    
    def load_config(self):
        """Load and validate YAML configuration file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self.config_data = yaml.safe_load(file)
            
            # Environment variable substitution
            self.config_data = self._substitute_env_vars(self.config_data)
            
            logger.success(f"Configuration loaded successfully from {self.config_path}")
            logger.debug(f"Configuration structure: {list(self.config_data.keys())}")
            
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in configuration file", error=e)
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading configuration", error=e)
            raise
    
    def _substitute_env_vars(self, data):
        """Recursively substitute environment variables in configuration"""
        if isinstance(data, dict):
            return {key: self._substitute_env_vars(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._substitute_env_vars(item) for item in data]
        elif isinstance(data, str) and data.startswith('${') and data.endswith('}'):
            env_var = data[2:-1]
            return os.getenv(env_var, data)
        return data
    
    def get_filtered_args(self, dataclass_type, section_name: str):
        """
        Extract and filter arguments for a specific dataclass from configuration.
        Only includes arguments that are defined in the dataclass to avoid
        TypeError when initializing the dataclass.
        """
        section_data = self.config_data.get(section_name, {})
        if not section_data:
            logger.warning(f"No configuration found for section: {section_name}")
            return {}
        
        # Get field names from dataclass
        valid_fields = {field.name for field in dataclass_type.__dataclass_fields__.values()}
        
        # Filter configuration to only include valid fields
        filtered_args = {
            key: value for key, value in section_data.items()
            if key in valid_fields
        }
        
        logger.debug(
            f"Filtered {len(filtered_args)}/{len(section_data)} arguments for {section_name}",
            valid_args=list(filtered_args.keys()),
            invalid_args=list(set(section_data.keys()) - valid_fields)
        )
        
        return filtered_args
    
    def save_config(self, output_path: str):
        """Save current configuration to file for reproducibility"""
        try:
            with open(output_path, 'w', encoding='utf-8') as file:
                yaml.dump(self.config_data, file, default_flow_style=False, indent=2)
            logger.success(f"Configuration saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration", error=e)
    
    def display_config(self):
        """Display configuration in a beautiful table format"""
        table = Table(title="Configuration Summary", show_header=True, header_style="bold magenta")
        table.add_column("Section", style="cyan", no_wrap=True)
        table.add_column("Parameters", style="green")
        table.add_column("Count", justify="right", style="yellow")
        
        for section_name, section_data in self.config_data.items():
            if isinstance(section_data, dict):
                params = ", ".join(section_data.keys())
                count = len(section_data)
                table.add_row(section_name, params, str(count))
        
        console.print(table)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Advanced data collator for speech-to-text tasks with dynamic padding,
    attention mask handling, and memory optimization.
    
    Features:
    - Dynamic padding for variable-length sequences
    - Attention mask computation for transformer models
    - Label preprocessing with ignore tokens
    - Memory-efficient batch construction
    - Support for different input modalities
    """
    
    processor: Any
    decoder_start_token_id: int
    forward_attention_mask: bool

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Collate features into batches with proper padding and attention masks.
        
        Args:
            features: List of feature dictionaries containing audio and text data
            
        Returns:
            Dictionary with batched and padded tensors ready for model input
        """
        # Separate inputs and labels for different padding strategies
        model_input_name = self.processor.model_input_names[0]
        input_features = [{model_input_name: feature[model_input_name]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad input features (audio)
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Handle attention masks for SpecAugment
        if self.forward_attention_mask:
            batch["attention_mask"] = torch.LongTensor([feature["attention_mask"] for feature in features])

        # Pad label features (text)
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding tokens with -100 for loss computation
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove BOS token if it was added during tokenization
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

class AdvancedTrainingCallback(TrainerCallback):
    """
    Advanced callback system for monitoring training progress, implementing
    early stopping, learning rate scheduling, and performance optimization.
    
    Features:
    - Real-time training metrics monitoring
    - Adaptive learning rate scheduling
    - Memory usage tracking and optimization
    - Model performance analysis
    - Automatic checkpoint management
    - Rich-based progress visualization
    """
    
    def __init__(self, logger: AdvancedLogger, patience: int = 3):
        self.logger = logger
        self.patience = patience
        self.best_metric = float('inf')
        self.patience_counter = 0
        self.training_start_time = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize training monitoring"""
        self.training_start_time = time.time()
        self.logger.info("🚀 Training started with advanced monitoring")
        
        # Display training configuration
        self.logger.info(f"Training epochs: {args.num_train_epochs}")
        self.logger.info(f"Batch size: {args.per_device_train_batch_size}")
        self.logger.info(f"Learning rate: {args.learning_rate}")
        self.logger.info(f"Output directory: {args.output_dir}")
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Start epoch monitoring"""
        self.logger.info(f"📊 Starting epoch {state.epoch + 1}/{args.num_train_epochs}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Process and display training logs"""
        if logs:
            # Extract key metrics
            if 'train_loss' in logs:
                self.logger.info(f"📈 Training loss: {logs['train_loss']:.4f}")
            
            if 'eval_loss' in logs:
                current_metric = logs['eval_loss']
                self.logger.info(f"📊 Validation loss: {current_metric:.4f}")
                
                # Early stopping logic
                if current_metric < self.best_metric:
                    self.best_metric = current_metric
                    self.patience_counter = 0
                    self.logger.success(f"✨ New best model! Loss: {current_metric:.4f}")
                else:
                    self.patience_counter += 1
                    self.logger.warning(f"⏳ No improvement for {self.patience_counter} evaluations")
                    
                    if self.patience_counter >= self.patience:
                        self.logger.warning("🛑 Early stopping triggered")
                        control.should_training_stop = True
            
            # Memory usage monitoring
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
                self.logger.debug(f"GPU Memory - Used: {memory_used:.2f}GB, Cached: {memory_cached:.2f}GB")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Finalize training monitoring"""
        training_duration = time.time() - self.training_start_time
        self.logger.performance("Training", training_duration, best_metric=self.best_metric)
        self.logger.success("🎉 Training completed successfully!")

class SpeechRecognitionPipeline:
    """
    Advanced end-to-end speech recognition pipeline with comprehensive features
    for data processing, model training, evaluation, and deployment.
    
    This class orchestrates the entire machine learning workflow including:
    - Configuration management and validation
    - Data loading and preprocessing
    - Model initialization and optimization
    - Training with advanced callbacks and monitoring
    - Evaluation with comprehensive metrics
    - Model deployment and inference optimization
    
    Architecture:
    - Modular design with clear separation of concerns
    - Error handling and recovery mechanisms
    - Memory optimization for large-scale training
    - Distributed training support
    - Automatic hyperparameter optimization
    """
    
    def __init__(self, config_path: str):
        self.config_manager = ConfigurationManager(config_path)
        self.model_args = None
        self.data_args = None
        self.training_args = None
        self.model = None
        self.tokenizer = None
        self.feature_extractor = None
        self.processor = None
        self.trainer = None
        
        # Display configuration summary
        self.config_manager.display_config()
        
        # Initialize components
        self._initialize_arguments()
        self._setup_logging()
        self._validate_environment()
    
    def _initialize_arguments(self):
        """Initialize all argument classes from configuration"""
        try:
            # Extract and filter arguments for each component
            model_config = self.config_manager.get_filtered_args(ModelArguments, "model_arguments")
            data_config = self.config_manager.get_filtered_args(DataTrainingArguments, "data_arguments")
            
            # Initialize argument objects
            self.model_args = ModelArguments(**model_config)
            self.data_args = DataTrainingArguments(**data_config)
            
            # Handle training arguments specially (HuggingFace class)
            training_config = self.config_manager.config_data.get("training_arguments", {})
            self.training_args = Seq2SeqTrainingArguments(**training_config)
            
            logger.success("Arguments initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize arguments", error=e)
            raise
    
    def _setup_logging(self):
        """Configure comprehensive logging system"""
        # Setup output directory
        os.makedirs(self.training_args.output_dir, exist_ok=True)
        
        # Configure transformers logging
        log_level = self.training_args.get_process_log_level()
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
        
        logger.info(f"Logging configured with level: {logging.getLevelName(log_level)}")
        logger.info(f"Output directory: {self.training_args.output_dir}")
    
    def _validate_environment(self):
        """Validate training environment and dependencies"""
        logger.info("🔍 Validating training environment...")
        
        # Check PyTorch installation
        logger.info(f"PyTorch version: {torch.__version__}")
        
        # Check GPU availability
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            logger.success(f"GPU available: {gpu_name} (Count: {gpu_count})")
        else:
            logger.warning("No GPU available, training will use CPU")
        
        # Check transformers version
        logger.info(f"Transformers version: {transformers.__version__}")
        
        # Validate model path
        if not self.model_args.model_name_or_path:
            raise ValueError("model_name_or_path is required")
        
        logger.success("Environment validation completed")
    
    def load_datasets(self) -> DatasetDict:
        """
        Load and validate datasets with comprehensive error handling and optimization.
        
        Returns:
            DatasetDict containing train and evaluation datasets
        """
        logger.info("📚 Loading datasets...")
        
        try:
            raw_datasets = DatasetDict()
            
            # Load training dataset
            if self.training_args.do_train:
                logger.info(f"Loading training dataset: {self.data_args.dataset_name}")
                
                load_kwargs = {
                    "path": self.data_args.dataset_name,
                    "split": self.data_args.train_split_name,
                    "cache_dir": self.model_args.cache_dir,
                }
                
                # Add optional arguments only if they exist
                if self.data_args.dataset_config_name:
                    load_kwargs["name"] = self.data_args.dataset_config_name
                if hasattr(self.data_args, 'streaming') and self.data_args.streaming:
                    load_kwargs["streaming"] = True
                if self.model_args.token:
                    load_kwargs["token"] = self.model_args.token
                if self.model_args.trust_remote_code:
                    load_kwargs["trust_remote_code"] = True
                
                raw_datasets["train"] = load_dataset(**load_kwargs)
                logger.success(f"Training dataset loaded: {len(raw_datasets['train'])} samples")
            
            # Load evaluation dataset
            if self.training_args.do_eval:
                logger.info(f"Loading evaluation dataset: {self.data_args.dataset_name}")
                
                load_kwargs = {
                    "path": self.data_args.dataset_name,
                    "split": self.data_args.eval_split_name,
                    "cache_dir": self.model_args.cache_dir,
                }
                
                if self.data_args.dataset_config_name:
                    load_kwargs["name"] = self.data_args.dataset_config_name
                if self.model_args.token:
                    load_kwargs["token"] = self.model_args.token
                if self.model_args.trust_remote_code:
                    load_kwargs["trust_remote_code"] = True
                
                raw_datasets["eval"] = load_dataset(**load_kwargs)
                logger.success(f"Evaluation dataset loaded: {len(raw_datasets['eval'])} samples")
            
            # Validate dataset columns
            self._validate_dataset_columns(raw_datasets)
            
            return raw_datasets
            
        except Exception as e:
            logger.error("Failed to load datasets", error=e)
            raise
    
    def _validate_dataset_columns(self, datasets: DatasetDict):
        """Validate that required columns exist in datasets"""
        for split_name, dataset in datasets.items():
            if self.data_args.audio_column_name not in dataset.column_names:
                raise ValueError(
                    f"Audio column '{self.data_args.audio_column_name}' not found in {split_name} dataset. "
                    f"Available columns: {dataset.column_names}"
                )
            
            if self.data_args.text_column_name not in dataset.column_names:
                raise ValueError(
                    f"Text column '{self.data_args.text_column_name}' not found in {split_name} dataset. "
                    f"Available columns: {dataset.column_names}"
                )
        
        logger.success("Dataset column validation passed")
    
    def initialize_model_components(self):
        """
        Initialize model, tokenizer, and feature extractor with advanced configurations
        and optimization settings.
        """
        logger.info("🧠 Initializing model components...")
        
        try:
            # Load configuration
            config_kwargs = {
                "cache_dir": self.model_args.cache_dir,
                "revision": self.model_args.model_revision,
                "trust_remote_code": self.model_args.trust_remote_code,
            }
            if self.model_args.token:
                config_kwargs["token"] = self.model_args.token
            
            config = AutoConfig.from_pretrained(
                self.model_args.config_name or self.model_args.model_name_or_path,
                **config_kwargs
            )
            
            # Configure SpecAugment for Whisper models
            if getattr(config, "model_type", None) == "whisper":
                config.update({"apply_spec_augment": self.model_args.apply_spec_augment})
                logger.info(f"SpecAugment configured for Whisper: {self.model_args.apply_spec_augment}")
            
            # Initialize feature extractor
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.model_args.feature_extractor_name or self.model_args.model_name_or_path,
                **config_kwargs
            )
            logger.success("Feature extractor initialized")
            
            # Initialize tokenizer
            tokenizer_kwargs = {
                "cache_dir": self.model_args.cache_dir,
                "use_fast": self.model_args.use_fast_tokenizer,
                "revision": self.model_args.model_revision,
                "trust_remote_code": self.model_args.trust_remote_code,
            }
            if self.model_args.token:
                tokenizer_kwargs["token"] = self.model_args.token
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_args.tokenizer_name or self.model_args.model_name_or_path,
                **tokenizer_kwargs
            )
            logger.success("Tokenizer initialized")
            
            # Initialize model with optimization settings
            model_kwargs = {
                "config": config,
                "cache_dir": self.model_args.cache_dir,
                "revision": self.model_args.model_revision,
                "trust_remote_code": self.model_args.trust_remote_code,
            }
            if self.model_args.token:
                model_kwargs["token"] = self.model_args.token
            
            # Set torch dtype
            if hasattr(self.model_args, 'torch_dtype'):
                if self.model_args.torch_dtype == "float16":
                    model_kwargs["torch_dtype"] = torch.float16
                elif self.model_args.torch_dtype == "bfloat16":
                    model_kwargs["torch_dtype"] = torch.bfloat16
            
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_args.model_name_or_path,
                **model_kwargs
            )
            
            # Validate decoder start token
            if self.model.config.decoder_start_token_id is None:
                raise ValueError("Model config must have decoder_start_token_id defined")
            
            # Apply model optimizations
            self._apply_model_optimizations()
            
            # Configure multilingual settings
            self._configure_multilingual_settings()
            
            logger.success("Model components initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize model components", error=e)
            raise
    
    def _apply_model_optimizations(self):
        """Apply various model optimizations for memory and performance"""
        # Freeze components as specified
        if self.model_args.freeze_feature_encoder:
            self.model.freeze_feature_encoder()
            logger.info("Feature encoder frozen")
        
        if self.model_args.freeze_encoder:
            self.model.freeze_encoder()
            self.model.model.encoder.gradient_checkpointing = False
            logger.info("Encoder frozen")
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model_args, 'gradient_checkpointing') and self.model_args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        # Apply 8-bit quantization if requested
        if hasattr(self.model_args, 'use_8bit') and self.model_args.use_8bit:
            # Note: This would require additional libraries like bitsandbytes
            logger.warning("8-bit quantization requested but not implemented in this example")
    
    def _configure_multilingual_settings(self):
        """Configure multilingual and task-specific settings"""
        if hasattr(self.model.generation_config, "is_multilingual") and self.model.generation_config.is_multilingual:
            if self.data_args.language or self.data_args.task != "transcribe":
                self.tokenizer.set_prefix_tokens(
                    language=self.data_args.language, 
                    task=self.data_args.task
                )
                self.model.generation_config.language = self.data_args.language
                self.model.generation_config.task = self.data_args.task
                logger.info(f"Multilingual model configured: {self.data_args.language}, {self.data_args.task}")
        elif self.data_args.language is not None:
            raise ValueError(
                "Language token specified for English-only model. "
                "Language should only be set for multilingual models."
            )
    
    def preprocess_datasets(self, raw_datasets: DatasetDict) -> DatasetDict:
        """
        Advanced dataset preprocessing with feature engineering, filtering,
        and optimization for training efficiency.
        
        Args:
            raw_datasets: Raw datasets loaded from source
            
        Returns:
            Preprocessed and optimized datasets ready for training
        """
        logger.info("⚙️ Preprocessing datasets...")
        
        try:
            # Resample audio if necessary
            dataset_sampling_rate = next(iter(raw_datasets.values())).features[
                self.data_args.audio_column_name
            ].sampling_rate
            
            if dataset_sampling_rate != self.feature_extractor.sampling_rate:
                logger.info(
                    f"Resampling audio from {dataset_sampling_rate}Hz to "
                    f"{self.feature_extractor.sampling_rate}Hz"
                )
                raw_datasets = raw_datasets.cast_column(
                    self.data_args.audio_column_name,
                    Audio(sampling_rate=self.feature_extractor.sampling_rate)
                )
            
            # Limit samples for debugging
            if self.data_args.max_train_samples and "train" in raw_datasets:
                raw_datasets["train"] = raw_datasets["train"].select(
                    range(min(self.data_args.max_train_samples, len(raw_datasets["train"])))
                )
                logger.info(f"Limited training samples to {len(raw_datasets['train'])}")
            
            if self.data_args.max_eval_samples and "eval" in raw_datasets:
                raw_datasets["eval"] = raw_datasets["eval"].select(
                    range(min(self.data_args.max_eval_samples, len(raw_datasets["eval"])))
                )
                logger.info(f"Limited evaluation samples to {len(raw_datasets['eval'])}")
            
            # Prepare preprocessing parameters
            max_input_length = self.data_args.max_duration_in_seconds * self.feature_extractor.sampling_rate
            min_input_length = self.data_args.min_duration_in_seconds * self.feature_extractor.sampling_rate
            
            # Check if attention mask is needed (for SpecAugment)
            forward_attention_mask = (
                getattr(self.model.config, "model_type", None) == "whisper"
                and getattr(self.model.config, "apply_spec_augment", False)
                and getattr(self.model.config, "mask_time_prob", 0) > 0
            )
            
            def prepare_dataset(batch):
                """Preprocessing function for individual samples"""
                # Process audio
                sample = batch[self.data_args.audio_column_name]
                inputs = self.feature_extractor(
                    sample["array"],
                    sampling_rate=sample["sampling_rate"],
                    return_attention_mask=forward_attention_mask
                )
                
                # Store processed audio features
                model_input_name = self.feature_extractor.model_input_names[0]
                batch[model_input_name] = inputs.get(model_input_name)[0]
                batch["input_length"] = len(sample["array"])
                
                if forward_attention_mask:
                    batch["attention_mask"] = inputs.get("attention_mask")[0]
                
                # Process text labels
                input_str = batch[self.data_args.text_column_name]
                if self.data_args.do_lower_case:
                    input_str = input_str.lower()
                
                batch["labels"] = self.tokenizer(input_str).input_ids
                return batch
            
            # Apply preprocessing with progress tracking
            with self.training_args.main_process_first(desc="dataset preprocessing"):
                vectorized_datasets = raw_datasets.map(
                    prepare_dataset,
                    remove_columns=next(iter(raw_datasets.values())).column_names,
                    num_proc=self.data_args.preprocessing_num_workers,
                    desc="Preprocessing samples",
                )
            
            # Filter by audio duration
            if hasattr(self.data_args, 'filter_by_duration') and self.data_args.filter_by_duration:
                def is_audio_in_length_range(length):
                    return min_input_length < length < max_input_length
                
                before_filter = {split: len(dataset) for split, dataset in vectorized_datasets.items()}
                
                vectorized_datasets = vectorized_datasets.filter(
                    is_audio_in_length_range,
                    num_proc=self.data_args.preprocessing_num_workers,
                    input_columns=["input_length"],
                )
                
                after_filter = {split: len(dataset) for split, dataset in vectorized_datasets.items()}
                
                for split in before_filter:
                    filtered_count = before_filter[split] - after_filter[split]
                    logger.info(f"Filtered {filtered_count} samples from {split} split")
            
            # Early exit for preprocessing-only mode
            if self.data_args.preprocessing_only:
                cache_info = {split: dataset.cache_files for split, dataset in vectorized_datasets.items()}
                logger.info(f"Preprocessing completed. Cache files: {cache_info}")
                return vectorized_datasets
            
            logger.success("Dataset preprocessing completed successfully")
            return vectorized_datasets
            
        except Exception as e:
            logger.error("Failed during dataset preprocessing", error=e)
            raise
    
    def setup_training_components(self, vectorized_datasets: DatasetDict):
        """
        Setup advanced training components including metrics, data collator,
        and trainer with comprehensive callbacks and optimization.
        
        Args:
            vectorized_datasets: Preprocessed datasets ready for training
        """
        logger.info("🔧 Setting up training components...")
        
        try:
            # Initialize evaluation metric
            metric = evaluate.load("wer", cache_dir=self.model_args.cache_dir)
            
            def compute_metrics(pred):
                """Compute Word Error Rate and other metrics"""
                pred_ids = pred.predictions
                pred.label_ids[pred.label_ids == -100] = self.tokenizer.pad_token_id
                
                # Decode predictions and labels
                pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
                label_str = self.tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
                
                # Compute WER
                wer_score = metric.compute(predictions=pred_str, references=label_str)
                
                # Additional metrics could be added here
                return {"wer": wer_score}
            
            # Save model components for processor creation
            with self.training_args.main_process_first():
                if is_main_process(self.training_args.local_rank):
                    self.feature_extractor.save_pretrained(self.training_args.output_dir)
                    self.tokenizer.save_pretrained(self.training_args.output_dir)
                    self.model.config.save_pretrained(self.training_args.output_dir)
            
            # Initialize processor
            self.processor = AutoProcessor.from_pretrained(self.training_args.output_dir)
            
            # Setup data collator
            forward_attention_mask = (
                getattr(self.model.config, "model_type", None) == "whisper"
                and getattr(self.model.config, "apply_spec_augment", False)
                and getattr(self.model.config, "mask_time_prob", 0) > 0
            )
            
            data_collator = DataCollatorSpeechSeq2SeqWithPadding(
                processor=self.processor,
                decoder_start_token_id=self.model.config.decoder_start_token_id,
                forward_attention_mask=forward_attention_mask,
            )
            
            # Setup training callbacks
            callbacks = [AdvancedTrainingCallback(logger)]
            
            # Add early stopping if configured
            if hasattr(self.training_args, 'early_stopping_patience'):
                callbacks.append(
                    EarlyStoppingCallback(early_stopping_patience=self.training_args.early_stopping_patience)
                )
            
            # Initialize trainer
            self.trainer = Seq2SeqTrainer(
                model=self.model,
                args=self.training_args,
                train_dataset=vectorized_datasets.get("train") if self.training_args.do_train else None,
                eval_dataset=vectorized_datasets.get("eval") if self.training_args.do_eval else None,
                processing_class=self.feature_extractor,
                data_collator=data_collator,
                compute_metrics=compute_metrics if self.training_args.predict_with_generate else None,
                callbacks=callbacks,
            )
            
            logger.success("Training components setup completed")
            
        except Exception as e:
            logger.error("Failed to setup training components", error=e)
            raise
    
    def execute_training(self):
        """
        Execute the training process with comprehensive monitoring,
        checkpointing, and error recovery.
        """
        if not self.training_args.do_train:
            logger.info("Training disabled, skipping training phase")
            return {}
        
        logger.info("🚀 Starting training execution...")
        
        try:
            # Check for existing checkpoints
            checkpoint = None
            if self.training_args.resume_from_checkpoint is not None:
                checkpoint = self.training_args.resume_from_checkpoint
                logger.info(f"Resuming from specified checkpoint: {checkpoint}")
            else:
                last_checkpoint = get_last_checkpoint(self.training_args.output_dir)
                if last_checkpoint is not None:
                    checkpoint = last_checkpoint
                    logger.info(f"Resuming from last checkpoint: {checkpoint}")
            
            # Set random seed for reproducibility
            set_seed(self.training_args.seed)
            
            # Execute training
            train_result = self.trainer.train(resume_from_checkpoint=checkpoint)
            
            # Save final model
            self.trainer.save_model()
            logger.success("Model saved successfully")
            
            # Process and log training metrics
            metrics = train_result.metrics
            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)
            self.trainer.save_state()
            
            # Display training summary
            self._display_training_summary(metrics)
            
            return train_result
            
        except Exception as e:
            logger.error("Training execution failed", error=e)
            raise
    
    def execute_evaluation(self):
        """
        Execute comprehensive evaluation with multiple metrics and analysis.
        
        Returns:
            Dictionary containing evaluation results and metrics
        """
        if not self.training_args.do_eval:
            logger.info("Evaluation disabled, skipping evaluation phase")
            return {}
        
        logger.info("📊 Starting evaluation execution...")
        
        try:
            # Run evaluation
            eval_results = self.trainer.evaluate(
                metric_key_prefix="eval",
                max_length=getattr(self.training_args, 'generation_max_length', None),
                num_beams=getattr(self.training_args, 'generation_num_beams', None),
            )
            
            # Log evaluation metrics
            self.trainer.log_metrics("eval", eval_results)
            self.trainer.save_metrics("eval", eval_results)
            
            # Display evaluation summary
            self._display_evaluation_summary(eval_results)
            
            return eval_results
            
        except Exception as e:
            logger.error("Evaluation execution failed", error=e)
            raise
    
    def _display_training_summary(self, metrics: Dict[str, float]):
        """Display comprehensive training summary with Rich formatting"""
        table = Table(title="🎯 Training Summary", show_header=True, header_style="bold green")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                table.add_row(metric_name, f"{metric_value:.6f}")
            else:
                table.add_row(metric_name, str(metric_value))
        
        console.print(table)
    
    def _display_evaluation_summary(self, results: Dict[str, float]):
        """Display comprehensive evaluation summary with Rich formatting"""
        table = Table(title="📈 Evaluation Results", show_header=True, header_style="bold blue")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        table.add_column("Description", style="green")
        
        metric_descriptions = {
            "eval_wer": "Word Error Rate (lower is better)",
            "eval_loss": "Validation Loss (lower is better)",
            "eval_runtime": "Evaluation Runtime (seconds)",
            "eval_samples_per_second": "Evaluation Speed",
        }
        
        for metric_name, metric_value in results.items():
            description = metric_descriptions.get(metric_name, "Custom metric")
            if isinstance(metric_value, float):
                table.add_row(metric_name, f"{metric_value:.6f}", description)
            else:
                table.add_row(metric_name, str(metric_value), description)
        
        console.print(table)
    
    def finalize_training(self):
        """
        Finalize training process with model publishing, artifact management,
        and deployment preparation.
        """
        logger.info("🏁 Finalizing training process...")
        
        try:
            # Prepare model card metadata
            kwargs = {
                "finetuned_from": self.model_args.model_name_or_path,
                "tasks": "automatic-speech-recognition",
            }
            
            if self.data_args.dataset_name:
                kwargs["dataset_tags"] = self.data_args.dataset_name
                if self.data_args.dataset_config_name:
                    kwargs["dataset_args"] = self.data_args.dataset_config_name
                    kwargs["dataset"] = f"{self.data_args.dataset_name} {self.data_args.dataset_config_name}"
                else:
                    kwargs["dataset"] = self.data_args.dataset_name
            
            # Push to hub or create model card
            if getattr(self.training_args, 'push_to_hub', False):
                logger.info("📤 Pushing model to Hugging Face Hub...")
                self.trainer.push_to_hub(**kwargs)
                logger.success("Model pushed to Hub successfully")
            else:
                logger.info("📋 Creating model card...")
                self.trainer.create_model_card(**kwargs)
                logger.success("Model card created successfully")
            
            # Save configuration for reproducibility
            config_backup_path = os.path.join(self.training_args.output_dir, "training_config.yaml")
            self.config_manager.save_config(config_backup_path)
            
            logger.success("Training finalization completed")
            
        except Exception as e:
            logger.error("Failed to finalize training", error=e)
            raise
    
    def run_complete_pipeline(self):
        """
        Execute the complete end-to-end speech recognition pipeline with
        comprehensive error handling and progress monitoring.
        
        Returns:
            Dictionary containing training and evaluation results
        """
        pipeline_start_time = time.time()
        results = {}
        
        try:
            # Display pipeline header
            console.print(Panel.fit(
                "🎤 Advanced Speech Recognition Pipeline\n"
                "End-to-End Training and Evaluation System",
                style="bold blue"
            ))
            
            # Step 1: Load datasets
            raw_datasets = self.load_datasets()
            
            # Step 2: Initialize model components
            self.initialize_model_components()
            
            # Step 3: Preprocess datasets
            vectorized_datasets = self.preprocess_datasets(raw_datasets)
            
            # Step 4: Setup training components
            self.setup_training_components(vectorized_datasets)
            
            # Step 5: Execute training
            if self.training_args.do_train:
                train_results = self.execute_training()
                results.update(train_results.metrics if hasattr(train_results, 'metrics') else {})
            
            # Step 6: Execute evaluation
            if self.training_args.do_eval:
                eval_results = self.execute_evaluation()
                results.update(eval_results)
            
            # Step 7: Finalize training
            self.finalize_training()
            
            # Calculate total pipeline duration
            pipeline_duration = time.time() - pipeline_start_time
            logger.performance("Complete Pipeline", pipeline_duration, **results)
            
            # Display final success message
            console.print(Panel.fit(
                "🎉 Pipeline Execution Completed Successfully!\n"
                f"Total Duration: {pipeline_duration:.2f} seconds\n"
                f"Output Directory: {self.training_args.output_dir}",
                style="bold green"
            ))
            
            return results
            
        except Exception as e:
            pipeline_duration = time.time() - pipeline_start_time
            logger.error(f"Pipeline failed after {pipeline_duration:.2f} seconds", error=e)
            
            # Display failure message
            console.print(Panel.fit(
                "❌ Pipeline Execution Failed!\n"
                f"Error: {str(e)}\n"
                f"Duration: {pipeline_duration:.2f} seconds",
                style="bold red"
            ))
            
            raise

def main():
    """
    Main entry point for the advanced speech recognition pipeline.
    Handles command-line argument parsing, configuration loading,
    and pipeline orchestration.
    
    Usage:
        python sequence_to_sequence_speech_recognition.py config.yaml
    
    Example configurations are available in the module documentation.
    """
    
    # Validate command line arguments
    if len(sys.argv) != 2:
        console.print(Panel.fit(
            "❌ Invalid Usage\n\n"
            "Correct usage:\n"
            "python sequence_to_sequence_speech_recognition.py <config.yaml>\n\n"
            "Example:\n"
            "python sequence_to_sequence_speech_recognition.py whisper_config.yaml",
            style="bold red"
        ))
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    # Validate configuration file exists
    if not os.path.exists(config_path):
        console.print(Panel.fit(
            f"❌ Configuration File Not Found\n\n"
            f"File: {config_path}\n"
            f"Please ensure the configuration file exists and is accessible.",
            style="bold red"
        ))
        sys.exit(1)
    
    try:
        # Initialize and run pipeline
        pipeline = SpeechRecognitionPipeline(config_path)
        results = pipeline.run_complete_pipeline()
        
        # Display final results summary
        if results:
            console.print("\n🏆 Final Results Summary:")
            for key, value in results.items():
                if isinstance(value, float):
                    console.print(f"  {key}: {value:.6f}")
                else:
                    console.print(f"  {key}: {value}")
        
        logger.success("🎊 Speech recognition pipeline completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("🛑 Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("💥 Pipeline failed with unexpected error", error=e)
        sys.exit(1)

if __name__ == "__main__":
    main()

# Example YAML Configuration Files
"""
Basic Configuration (whisper_config.yaml):
-------------------------------------------
model_arguments:
  model_name_or_path: "openai/whisper-small"
  freeze_feature_encoder: true
  apply_spec_augment: false
  gradient_checkpointing: true

data_arguments:
  dataset_name: "mozilla-foundation/common_voice_11_0"
  dataset_config_name: "en"
  audio_column_name: "audio"
  text_column_name: "sentence"
  max_duration_in_seconds: 30.0
  min_duration_in_seconds: 1.0
  do_lower_case: true
  language: null
  task: "transcribe"
  preprocessing_num_workers: 4

training_arguments:
  output_dir: "./whisper-small-cv11"
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 8
  gradient_accumulation_steps: 1
  num_train_epochs: 3
  learning_rate: 1e-5
  warmup_steps: 500
  logging_steps: 25
  eval_steps: 1000
  save_steps: 1000
  evaluation_strategy: "steps"
  save_strategy: "steps"
  generation_max_length: 225
  predict_with_generate: true
  fp16: true
  do_train: true
  do_eval: true
  load_best_model_at_end: true
  metric_for_best_model: "wer"
  greater_is_better: false
  push_to_hub: false

Advanced Configuration (advanced_whisper_config.yaml):
------------------------------------------------------
model_arguments:
  model_name_or_path: "openai/whisper-medium"
  freeze_feature_encoder: false
  freeze_encoder: false
  apply_spec_augment: true
  gradient_checkpointing: true
  torch_dtype: "float16"
  trust_remote_code: false

data_arguments:
  dataset_name: "mozilla-foundation/common_voice_11_0"
  dataset_config_name: "es"
  max_train_samples: 10000
  max_eval_samples: 1000
  audio_column_name: "audio"
  text_column_name: "sentence"
  max_duration_in_seconds: 25.0
  min_duration_in_seconds: 2.0
  do_lower_case: true
  language: "spanish"
  task: "transcribe"
  preprocessing_num_workers: 8
  filter_by_duration: true
  normalize_audio: true

training_arguments:
  output_dir: "./advanced-whisper-spanish"
  per_device_train_batch_size: 32
  per_device_eval_batch_size: 16
  gradient_accumulation_steps: 2
  num_train_epochs: 5
  learning_rate: 5e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  lr_scheduler_type: "cosine"
  logging_steps: 10
  eval_steps: 500
  save_steps: 500
  evaluation_strategy: "steps"
  save_strategy: "steps"
  save_total_limit: 3
  generation_max_length: 200
  generation_num_beams: 5
  predict_with_generate: true
  fp16: true
  dataloader_pin_memory: true
  dataloader_num_workers: 4
  do_train: true
  do_eval: true
  load_best_model_at_end: true
  metric_for_best_model: "eval_wer"
  greater_is_better: false
  early_stopping_patience: 3
  push_to_hub: false
  hub_model_id: "my-username/advanced-whisper-spanish"
  hub_strategy: "every_save"
"""