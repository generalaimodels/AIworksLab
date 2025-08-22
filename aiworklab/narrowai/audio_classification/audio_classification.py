#!/usr/bin/env python3
"""
Advanced End-to-End Audio Classification Pipeline
===============================================

This module implements a state-of-the-art, production-ready audio classification pipeline
with comprehensive features including:

1. YAML-based configuration management
2. Advanced preprocessing with feature engineering
3. Automated hyperparameter optimization using Optuna
4. Rich console output with progress tracking
5. Comprehensive evaluation metrics and visualization
6. Model checkpointing and best model saving
7. Cross-validation support
8. Data augmentation capabilities
9. Advanced logging and monitoring
10. Deployment-ready model export

Architecture Overview:
- DataProcessor: Handles data loading, preprocessing, and augmentation
- ModelManager: Manages model initialization, training, and evaluation
- HyperparameterOptimizer: Automated hyperparameter tuning
- MetricsCalculator: Comprehensive evaluation metrics
- PipelineOrchestrator: Main orchestrator class

Usage:
    python audio_classification.py config.yaml

Author: Advanced AI Pipeline Development Team
Version: 2.0.0
License: MIT
"""

import os
import sys
import yaml
import json
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import argparse

# Core ML libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import transformers
from transformers import (
    AutoConfig, AutoFeatureExtractor, AutoModelForAudioClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict, load_dataset, Audio
import evaluate

# Optimization and hyperparameter tuning
import optuna
from optuna.integration import PyTorchLightningPruningCallback

# Visualization and monitoring
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve, f1_score
)
from sklearn.model_selection import StratifiedKFold

# Rich for beautiful console output
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TaskID
from rich.panel import Panel
from rich.tree import Tree
from rich.syntax import Syntax
from rich.logging import RichHandler
from rich.traceback import install

# Audio processing
import librosa
import torchaudio
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

# Utilities
import pickle
import joblib
from contextlib import contextmanager

# Install rich traceback for better error messages
install(show_locals=True)

# Initialize Rich console
console = Console()

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


@dataclass
class AudioConfig:
    """Configuration class for audio processing parameters."""
    sampling_rate: int = 16000
    max_length_seconds: float = 10.0
    min_length_seconds: float = 1.0
    normalize_audio: bool = True
    apply_augmentation: bool = True
    augmentation_probability: float = 0.3


@dataclass
class ModelConfig:
    """Configuration class for model parameters."""
    model_name_or_path: str = "facebook/wav2vec2-base"
    freeze_feature_encoder: bool = True
    attention_mask: bool = True
    dropout_rate: float = 0.1
    hidden_size: int = 768
    num_attention_heads: int = 12


@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""
    output_dir: str = "./results"
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_steps: int = 100
    seed: int = 42
    fp16: bool = True
    dataloader_num_workers: int = 4
    early_stopping_patience: int = 3
    metric_for_best_model: str = "eval_f1"
    greater_is_better: bool = True


@dataclass
class DataConfig:
    """Configuration class for data parameters."""
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    train_file: Optional[str] = None
    eval_file: Optional[str] = None
    test_file: Optional[str] = None
    audio_column_name: str = "audio"
    label_column_name: str = "label"
    train_split_ratio: float = 0.8
    val_split_ratio: float = 0.1
    test_split_ratio: float = 0.1
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None


@dataclass
class OptimizationConfig:
    """Configuration class for hyperparameter optimization."""
    enable_optimization: bool = False
    n_trials: int = 50
    optimization_direction: str = "maximize"
    optimization_metric: str = "eval_f1"
    pruning_enabled: bool = True
    study_name: str = "audio_classification_study"


@dataclass
class ExperimentConfig:
    """Main configuration class combining all sub-configurations."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    verbose: bool = True
    experiment_name: str = "audio_classification_experiment"
    use_cross_validation: bool = False
    cv_folds: int = 5
    generate_reports: bool = True
    save_predictions: bool = True


class AudioAugmentationPipeline:
    """Advanced audio augmentation pipeline using audiomentations."""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.augmentations = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.3),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.3),
        ])
    
    def __call__(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply augmentations to audio signal."""
        if self.config.apply_augmentation and np.random.random() < self.config.augmentation_probability:
            return self.augmentations(samples=audio, sample_rate=sample_rate)
        return audio


class AdvancedMetricsCalculator:
    """Comprehensive metrics calculator for audio classification."""
    
    def __init__(self, label_names: List[str]):
        self.label_names = label_names
        self.num_labels = len(label_names)
    
    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Basic metrics
        accuracy = (predictions == labels).mean()
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_weighted = f1_score(labels, predictions, average='weighted')
        
        # Per-class metrics
        report = classification_report(
            labels, predictions, 
            target_names=self.label_names, 
            output_dict=True, 
            zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': report['macro avg']['precision'],
            'recall_macro': report['macro avg']['recall'],
        }
        
        # Add per-class metrics
        for i, label in enumerate(self.label_names):
            if label in report:
                metrics[f'f1_{label}'] = report[label]['f1-score']
                metrics[f'precision_{label}'] = report[label]['precision']
                metrics[f'recall_{label}'] = report[label]['recall']
        
        return metrics
    
    def generate_detailed_report(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        report = {
            'classification_report': classification_report(
                y_true, y_pred, target_names=self.label_names, zero_division=0
            ),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'accuracy': float((y_true == y_pred).mean()),
            'f1_scores': {
                'macro': float(f1_score(y_true, y_pred, average='macro')),
                'micro': float(f1_score(y_true, y_pred, average='micro')),
                'weighted': float(f1_score(y_true, y_pred, average='weighted'))
            }
        }
        
        # Add AUC scores for multi-class if probabilities available
        if y_prob is not None and self.num_labels > 2:
            try:
                report['auc_ovr'] = float(roc_auc_score(y_true, y_prob, multi_class='ovr'))
                report['auc_ovo'] = float(roc_auc_score(y_true, y_prob, multi_class='ovo'))
            except ValueError:
                console.print("[yellow]Warning: Could not compute AUC scores[/yellow]")
        
        return report


class DataProcessor:
    """Advanced data processing pipeline with comprehensive preprocessing."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.feature_extractor = None
        self.augmentation_pipeline = AudioAugmentationPipeline(config.audio)
        self.label_encoder = {}
        self.id2label = {}
        
    def setup_feature_extractor(self):
        """Initialize and configure the feature extractor."""
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.config.model.model_name_or_path,
            return_attention_mask=self.config.model.attention_mask
        )
        console.print(f"[green]✓[/green] Feature extractor loaded: {self.config.model.model_name_or_path}")
    
    def load_dataset(self) -> DatasetDict:
        """Load and prepare dataset with comprehensive error handling."""
        try:
            raw_datasets = DatasetDict()
            
            if self.config.data.dataset_name:
                # Load from HuggingFace datasets
                dataset = load_dataset(
                    self.config.data.dataset_name,
                    self.config.data.dataset_config_name
                )
                
                # Split dataset if needed
                if 'train' in dataset and 'validation' in dataset:
                    raw_datasets['train'] = dataset['train']
                    raw_datasets['validation'] = dataset['validation']
                    if 'test' in dataset:
                        raw_datasets['test'] = dataset['test']
                else:
                    # Create splits
                    train_test = dataset['train'].train_test_split(
                        test_size=1-self.config.data.train_split_ratio,
                        seed=self.config.training.seed
                    )
                    raw_datasets['train'] = train_test['train']
                    
                    # Further split test into validation and test
                    val_test = train_test['test'].train_test_split(
                        test_size=self.config.data.test_split_ratio / (1-self.config.data.train_split_ratio),
                        seed=self.config.training.seed
                    )
                    raw_datasets['validation'] = val_test['train']
                    raw_datasets['test'] = val_test['test']
            
            else:
                # Load from local files
                console.print("[yellow]Loading from local files not implemented in this example[/yellow]")
                raise NotImplementedError("Local file loading not implemented")
            
            # Cast audio column to correct format
            audio_feature = Audio(sampling_rate=self.config.audio.sampling_rate)
            for split in raw_datasets:
                raw_datasets[split] = raw_datasets[split].cast_column(
                    self.config.data.audio_column_name, audio_feature
                )
            
            # Setup label mappings
            self._setup_label_mappings(raw_datasets['train'])
            
            console.print(f"[green]✓[/green] Dataset loaded successfully")
            self._print_dataset_info(raw_datasets)
            
            return raw_datasets
            
        except Exception as e:
            console.print(f"[red]✗[/red] Error loading dataset: {str(e)}")
            raise
    
    def _setup_label_mappings(self, train_dataset):
        """Setup label to ID mappings."""
        labels = train_dataset.features[self.config.data.label_column_name].names
        
        for i, label in enumerate(labels):
            self.label_encoder[label] = i
            self.id2label[i] = label
        
        console.print(f"[green]✓[/green] Label mappings created for {len(labels)} classes")
    
    def _print_dataset_info(self, datasets: DatasetDict):
        """Print comprehensive dataset information."""
        table = Table(title="Dataset Information")
        table.add_column("Split", style="cyan")
        table.add_column("Size", style="magenta")
        table.add_column("Features", style="green")
        
        for split_name, dataset in datasets.items():
            features = ", ".join(dataset.column_names)
            table.add_row(split_name, str(len(dataset)), features)
        
        console.print(table)
    
    def random_subsample(self, wav: np.ndarray, max_length: float) -> np.ndarray:
        """Randomly subsample audio to specified length."""
        sample_length = int(round(self.config.audio.sampling_rate * max_length))
        if len(wav) <= sample_length:
            return wav
        
        random_offset = np.random.randint(0, len(wav) - sample_length)
        return wav[random_offset:random_offset + sample_length]
    
    def preprocess_audio(self, audio: np.ndarray, is_training: bool = True) -> np.ndarray:
        """Comprehensive audio preprocessing pipeline."""
        # Normalize audio
        if self.config.audio.normalize_audio:
            audio = librosa.util.normalize(audio)
        
        # Apply augmentation during training
        if is_training and self.config.audio.apply_augmentation:
            audio = self.augmentation_pipeline(audio, self.config.audio.sampling_rate)
        
        # Subsample if needed
        if is_training:
            audio = self.random_subsample(audio, self.config.audio.max_length_seconds)
        
        return audio
    
    def create_train_transforms(self):
        """Create training data transformation function."""
        def train_transforms(batch):
            model_input_name = self.feature_extractor.model_input_names[0]
            processed_audio = []
            
            for audio in batch[self.config.data.audio_column_name]:
                processed = self.preprocess_audio(audio["array"], is_training=True)
                processed_audio.append(processed)
            
            # Extract features
            inputs = self.feature_extractor(
                processed_audio, 
                sampling_rate=self.config.audio.sampling_rate,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            output_batch = {model_input_name: inputs.get(model_input_name)}
            output_batch["labels"] = batch[self.config.data.label_column_name]
            
            return output_batch
        
        return train_transforms
    
    def create_eval_transforms(self):
        """Create evaluation data transformation function."""
        def eval_transforms(batch):
            model_input_name = self.feature_extractor.model_input_names[0]
            processed_audio = []
            
            for audio in batch[self.config.data.audio_column_name]:
                processed = self.preprocess_audio(audio["array"], is_training=False)
                processed_audio.append(processed)
            
            # Extract features
            inputs = self.feature_extractor(
                processed_audio,
                sampling_rate=self.config.audio.sampling_rate,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            output_batch = {model_input_name: inputs.get(model_input_name)}
            output_batch["labels"] = batch[self.config.data.label_column_name]
            
            return output_batch
        
        return eval_transforms


class ModelManager:
    """Advanced model management with comprehensive training and evaluation."""
    
    def __init__(self, config: ExperimentConfig, data_processor: DataProcessor):
        self.config = config
        self.data_processor = data_processor
        self.model = None
        self.trainer = None
        self.metrics_calculator = None
        
    def setup_model(self):
        """Initialize and configure the model."""
        # Setup model configuration
        model_config = AutoConfig.from_pretrained(
            self.config.model.model_name_or_path,
            num_labels=len(self.data_processor.id2label),
            label2id={label: str(id) for id, label in self.data_processor.id2label.items()},
            id2label={str(id): label for id, label in self.data_processor.id2label.items()},
            finetuning_task="audio-classification"
        )
        
        # Load model
        self.model = AutoModelForAudioClassification.from_pretrained(
            self.config.model.model_name_or_path,
            config=model_config,
            ignore_mismatched_sizes=True
        )
        
        # Freeze feature encoder if specified
        if self.config.model.freeze_feature_encoder:
            self.model.freeze_feature_encoder()
            console.print("[green]✓[/green] Feature encoder frozen")
        
        # Setup metrics calculator
        label_names = list(self.data_processor.id2label.values())
        self.metrics_calculator = AdvancedMetricsCalculator(label_names)
        
        console.print(f"[green]✓[/green] Model loaded: {self.config.model.model_name_or_path}")
        console.print(f"[green]✓[/green] Number of labels: {len(self.data_processor.id2label)}")
    
    def create_training_arguments(self, trial=None) -> TrainingArguments:
        """Create training arguments with optional hyperparameter optimization."""
        # Base arguments
        args_dict = {
            "output_dir": self.config.training.output_dir,
            "num_train_epochs": self.config.training.num_train_epochs,
            "per_device_train_batch_size": self.config.training.per_device_train_batch_size,
            "per_device_eval_batch_size": self.config.training.per_device_eval_batch_size,
            "learning_rate": self.config.training.learning_rate,
            "weight_decay": self.config.training.weight_decay,
            "warmup_ratio": self.config.training.warmup_ratio,
            "evaluation_strategy": self.config.training.evaluation_strategy,
            "save_strategy": self.config.training.save_strategy,
            "logging_steps": self.config.training.logging_steps,
            "seed": self.config.training.seed,
            "fp16": self.config.training.fp16,
            "dataloader_num_workers": self.config.training.dataloader_num_workers,
            "load_best_model_at_end": True,
            "metric_for_best_model": self.config.training.metric_for_best_model,
            "greater_is_better": self.config.training.greater_is_better,
            "save_total_limit": 3,
            "report_to": "none"  # Disable wandb/tensorboard for this example
        }
        
        # Override with trial parameters if optimizing
        if trial is not None:
            args_dict.update({
                "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
                "per_device_train_batch_size": trial.suggest_categorical("batch_size", [4, 8, 16, 32]),
                "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.3),
                "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3)
            })
        
        return TrainingArguments(**args_dict)
    
    def setup_trainer(self, datasets: DatasetDict, trial=None) -> Trainer:
        """Setup trainer with comprehensive configuration."""
        training_args = self.create_training_arguments(trial)
        
        # Setup callbacks
        callbacks = []
        if self.config.training.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.training.early_stopping_patience
                )
            )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets.get("train"),
            eval_dataset=datasets.get("validation"),
            processing_class=self.data_processor.feature_extractor,
            compute_metrics=self.metrics_calculator.compute_metrics,
            callbacks=callbacks
        )
        
        return self.trainer
    
    def train_model(self, datasets: DatasetDict, trial=None) -> Dict[str, float]:
        """Train the model with comprehensive monitoring."""
        console.print("[bold blue]Starting model training...[/bold blue]")
        
        # Setup trainer
        trainer = self.setup_trainer(datasets, trial)
        
        # Apply data transforms
        train_transforms = self.data_processor.create_train_transforms()
        eval_transforms = self.data_processor.create_eval_transforms()
        
        datasets["train"].set_transform(train_transforms, output_all_columns=False)
        datasets["validation"].set_transform(eval_transforms, output_all_columns=False)
        
        # Train model
        train_result = trainer.train()
        
        # Evaluate model
        eval_result = trainer.evaluate()
        
        console.print("[green]✓[/green] Training completed successfully")
        
        # Print training summary
        self._print_training_summary(train_result.metrics, eval_result)
        
        return eval_result
    
    def _print_training_summary(self, train_metrics: Dict, eval_metrics: Dict):
        """Print comprehensive training summary."""
        table = Table(title="Training Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Train", style="green")
        table.add_column("Validation", style="magenta")
        
        common_metrics = ["loss", "accuracy", "f1_macro", "f1_weighted"]
        
        for metric in common_metrics:
            train_key = f"train_{metric}"
            eval_key = f"eval_{metric}"
            
            train_val = train_metrics.get(train_key, "N/A")
            eval_val = eval_metrics.get(eval_key, "N/A")
            
            if isinstance(train_val, float):
                train_val = f"{train_val:.4f}"
            if isinstance(eval_val, float):
                eval_val = f"{eval_val:.4f}"
            
            table.add_row(metric, str(train_val), str(eval_val))
        
        console.print(table)
    
    def evaluate_model(self, datasets: DatasetDict) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        console.print("[bold blue]Starting comprehensive evaluation...[/bold blue]")
        
        results = {}
        
        for split_name in ["validation", "test"]:
            if split_name in datasets:
                console.print(f"[yellow]Evaluating on {split_name} set...[/yellow]")
                
                # Apply transforms
                eval_transforms = self.data_processor.create_eval_transforms()
                datasets[split_name].set_transform(eval_transforms, output_all_columns=False)
                
                # Get predictions
                predictions = self.trainer.predict(datasets[split_name])
                y_pred = np.argmax(predictions.predictions, axis=1)
                y_true = predictions.label_ids
                y_prob = torch.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
                
                # Generate detailed report
                detailed_report = self.metrics_calculator.generate_detailed_report(
                    y_true, y_pred, y_prob
                )
                
                results[split_name] = {
                    "metrics": predictions.metrics,
                    "detailed_report": detailed_report,
                    "predictions": y_pred.tolist(),
                    "true_labels": y_true.tolist(),
                    "probabilities": y_prob.tolist()
                }
                
                console.print(f"[green]✓[/green] {split_name.capitalize()} evaluation completed")
        
        return results


class HyperparameterOptimizer:
    """Advanced hyperparameter optimization using Optuna."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.study = None
        
    def create_objective(self, model_manager: ModelManager, datasets: DatasetDict):
        """Create objective function for optimization."""
        def objective(trial):
            try:
                # Train model with trial parameters
                eval_result = model_manager.train_model(datasets, trial)
                
                # Return optimization metric
                metric_key = f"eval_{self.config.optimization.optimization_metric}"
                return eval_result.get(metric_key, 0.0)
                
            except Exception as e:
                console.print(f"[red]Trial failed: {str(e)}[/red]")
                return 0.0
        
        return objective
    
    def optimize(self, model_manager: ModelManager, datasets: DatasetDict) -> optuna.Study:
        """Run hyperparameter optimization."""
        console.print("[bold blue]Starting hyperparameter optimization...[/bold blue]")
        
        # Create or load study
        study_name = f"{self.config.optimization.study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.study = optuna.create_study(
            direction=self.config.optimization.optimization_direction,
            study_name=study_name,
            pruner=optuna.pruners.MedianPruner() if self.config.optimization.pruning_enabled else None
        )
        
        # Create objective function
        objective = self.create_objective(model_manager, datasets)
        
        # Run optimization
        with Progress() as progress:
            task = progress.add_task(
                "[green]Optimizing hyperparameters...", 
                total=self.config.optimization.n_trials
            )
            
            for i in range(self.config.optimization.n_trials):
                self.study.optimize(objective, n_trials=1)
                progress.update(task, advance=1)
                
                # Print best trial info
                if (i + 1) % 10 == 0:
                    best_trial = self.study.best_trial
                    console.print(
                        f"[cyan]Trial {i+1}/{self.config.optimization.n_trials}: "
                        f"Best {self.config.optimization.optimization_metric} = {best_trial.value:.4f}[/cyan]"
                    )
        
        # Print optimization results
        self._print_optimization_results()
        
        return self.study
    
    def _print_optimization_results(self):
        """Print comprehensive optimization results."""
        best_trial = self.study.best_trial
        
        console.print(Panel.fit(
            f"[bold green]Best Trial Results[/bold green]\n"
            f"Value: {best_trial.value:.4f}\n"
            f"Params: {json.dumps(best_trial.params, indent=2)}",
            title="Hyperparameter Optimization Complete"
        ))


class CrossValidator:
    """K-Fold cross-validation implementation."""
    
    def __init__(self, config: ExperimentConfig, n_folds: int = 5):
        self.config = config
        self.n_folds = n_folds
        self.cv_results = []
    
    def run_cross_validation(
        self, 
        datasets: DatasetDict, 
        model_manager: ModelManager
    ) -> List[Dict[str, float]]:
        """Run k-fold cross-validation."""
        console.print(f"[bold blue]Starting {self.n_folds}-fold cross-validation...[/bold blue]")
        
        # Combine train and validation sets for CV
        combined_dataset = datasets["train"]
        if "validation" in datasets:
            combined_dataset = datasets["train"].concatenate(datasets["validation"])
        
        # Get labels for stratified split
        labels = [example[self.config.data.label_column_name] for example in combined_dataset]
        
        # Create stratified folds
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.config.training.seed)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(combined_dataset)), labels)):
            console.print(f"[yellow]Running fold {fold + 1}/{self.n_folds}...[/yellow]")
            
            # Create fold datasets
            fold_train = combined_dataset.select(train_idx)
            fold_val = combined_dataset.select(val_idx)
            
            fold_datasets = DatasetDict({
                "train": fold_train,
                "validation": fold_val
            })
            
            # Reinitialize model for each fold
            model_manager.setup_model()
            
            # Train and evaluate
            eval_result = model_manager.train_model(fold_datasets)
            fold_results.append(eval_result)
            
            console.print(f"[green]✓[/green] Fold {fold + 1} completed")
        
        # Calculate cross-validation statistics
        self._calculate_cv_statistics(fold_results)
        
        return fold_results
    
    def _calculate_cv_statistics(self, fold_results: List[Dict[str, float]]):
        """Calculate and display cross-validation statistics."""
        # Extract metrics
        metrics = {}
        for result in fold_results:
            for key, value in result.items():
                if key.startswith("eval_") and isinstance(value, (int, float)):
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
        
        # Calculate statistics
        table = Table(title="Cross-Validation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Mean", style="green")
        table.add_column("Std", style="magenta")
        table.add_column("Min", style="red")
        table.add_column("Max", style="blue")
        
        for metric, values in metrics.items():
            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                
                table.add_row(
                    metric.replace("eval_", ""),
                    f"{mean_val:.4f}",
                    f"{std_val:.4f}",
                    f"{min_val:.4f}",
                    f"{max_val:.4f}"
                )
        
        console.print(table)


class ReportGenerator:
    """Comprehensive report generation and visualization."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = Path(config.training.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_comprehensive_report(
        self, 
        evaluation_results: Dict[str, Any],
        model_manager: ModelManager,
        training_time: float
    ):
        """Generate comprehensive experiment report."""
        console.print("[bold blue]Generating comprehensive report...[/bold blue]")
        
        report = {
            "experiment_info": {
                "name": self.config.experiment_name,
                "timestamp": datetime.now().isoformat(),
                "training_time_seconds": training_time,
                "model_name": self.config.model.model_name_or_path,
                "num_labels": len(model_manager.data_processor.id2label),
                "label_names": list(model_manager.data_processor.id2label.values())
            },
            "configuration": asdict(self.config),
            "results": evaluation_results
        }
        
        # Save detailed report
        report_path = self.output_dir / "experiment_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate visualizations
        self._generate_visualizations(evaluation_results, model_manager)
        
        # Generate summary
        self._generate_summary(evaluation_results)
        
        console.print(f"[green]✓[/green] Report saved to {report_path}")
    
    def _generate_visualizations(self, evaluation_results: Dict[str, Any], model_manager: ModelManager):
        """Generate comprehensive visualizations."""
        plt.style.use('seaborn-v0_8')
        
        for split_name, results in evaluation_results.items():
            if "detailed_report" in results:
                detailed_report = results["detailed_report"]
                
                # Confusion Matrix
                plt.figure(figsize=(10, 8))
                cm = np.array(detailed_report["confusion_matrix"])
                sns.heatmap(
                    cm, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues',
                    xticklabels=model_manager.data_processor.id2label.values(),
                    yticklabels=model_manager.data_processor.id2label.values()
                )
                plt.title(f'Confusion Matrix - {split_name.capitalize()}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                plt.savefig(self.output_dir / f'confusion_matrix_{split_name}.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # F1 Scores Bar Plot
                f1_scores = detailed_report["f1_scores"]
                plt.figure(figsize=(8, 6))
                plt.bar(f1_scores.keys(), f1_scores.values())
                plt.title(f'F1 Scores - {split_name.capitalize()}')
                plt.ylabel('F1 Score')
                plt.ylim(0, 1)
                for i, v in enumerate(f1_scores.values()):
                    plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
                plt.tight_layout()
                plt.savefig(self.output_dir / f'f1_scores_{split_name}.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        console.print("[green]✓[/green] Visualizations generated")
    
    def _generate_summary(self, evaluation_results: Dict[str, Any]):
        """Generate experiment summary."""
        summary_table = Table(title="Experiment Summary")
        summary_table.add_column("Dataset", style="cyan")
        summary_table.add_column("Accuracy", style="green")
        summary_table.add_column("F1 (Macro)", style="magenta")
        summary_table.add_column("F1 (Weighted)", style="blue")
        
        for split_name, results in evaluation_results.items():
            if "detailed_report" in results:
                report = results["detailed_report"]
                accuracy = f"{report['accuracy']:.4f}"
                f1_macro = f"{report['f1_scores']['macro']:.4f}"
                f1_weighted = f"{report['f1_scores']['weighted']:.4f}"
                
                summary_table.add_row(split_name.capitalize(), accuracy, f1_macro, f1_weighted)
        
        console.print(summary_table)


class PipelineOrchestrator:
    """Main orchestrator class for the entire pipeline."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data_processor = DataProcessor(config)
        self.model_manager = None
        self.report_generator = ReportGenerator(config)
        
        # Setup logging
        self._setup_logging()
        
        # Print pipeline info
        self._print_pipeline_info()
    
    def _setup_logging(self):
        """Setup comprehensive logging system."""
        log_dir = Path(self.config.training.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
        
        # File handler
        file_handler = logging.FileHandler(
            log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
        
        # Rich handler for console
        if self.config.verbose:
            rich_handler = RichHandler(rich_tracebacks=True)
            rich_handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(rich_handler)
    
    def _print_pipeline_info(self):
        """Print comprehensive pipeline information."""
        tree = Tree("[bold blue]Audio Classification Pipeline[/bold blue]")
        
        # Configuration tree
        config_branch = tree.add("[green]Configuration[/green]")
        config_branch.add(f"Experiment: {self.config.experiment_name}")
        config_branch.add(f"Model: {self.config.model.model_name_or_path}")
        config_branch.add(f"Epochs: {self.config.training.num_train_epochs}")
        config_branch.add(f"Batch Size: {self.config.training.per_device_train_batch_size}")
        config_branch.add(f"Learning Rate: {self.config.training.learning_rate}")
        
        # Features tree
        features_branch = tree.add("[magenta]Features[/magenta]")
        features_branch.add(f"✓ Audio Augmentation" if self.config.audio.apply_augmentation else "✗ Audio Augmentation")
        features_branch.add(f"✓ Hyperparameter Optimization" if self.config.optimization.enable_optimization else "✗ Hyperparameter Optimization")
        features_branch.add(f"✓ Cross Validation" if self.config.use_cross_validation else "✗ Cross Validation")
        features_branch.add(f"✓ Comprehensive Reports" if self.config.generate_reports else "✗ Comprehensive Reports")
        
        console.print(tree)
    
    def run_pipeline(self) -> Dict[str, Any]:
        """Execute the complete pipeline."""
        start_time = datetime.now()
        
        try:
            console.print("[bold green]🚀 Starting Audio Classification Pipeline[/bold green]")
            
            # Step 1: Setup data processor
            console.print("\n[bold]Step 1: Data Processing Setup[/bold]")
            self.data_processor.setup_feature_extractor()
            
            # Step 2: Load and prepare datasets
            console.print("\n[bold]Step 2: Dataset Loading[/bold]")
            datasets = self.data_processor.load_dataset()
            
            # Step 3: Initialize model manager
            console.print("\n[bold]Step 3: Model Setup[/bold]")
            self.model_manager = ModelManager(self.config, self.data_processor)
            self.model_manager.setup_model()
            
            # Step 4: Hyperparameter optimization (optional)
            if self.config.optimization.enable_optimization:
                console.print("\n[bold]Step 4: Hyperparameter Optimization[/bold]")
                optimizer = HyperparameterOptimizer(self.config)
                study = optimizer.optimize(self.model_manager, datasets)
                
                # Update config with best parameters
                best_params = study.best_trial.params
                for param, value in best_params.items():
                    setattr(self.config.training, param, value)
                
                # Reinitialize model with best parameters
                self.model_manager.setup_model()
            
            # Step 5: Cross-validation (optional)
            cv_results = None
            if self.config.use_cross_validation:
                console.print("\n[bold]Step 5: Cross-Validation[/bold]")
                cv = CrossValidator(self.config, self.config.cv_folds)
                cv_results = cv.run_cross_validation(datasets, self.model_manager)
            
            # Step 6: Final training
            console.print("\n[bold]Step 6: Final Model Training[/bold]")
            final_training_results = self.model_manager.train_model(datasets)
            
            # Step 7: Comprehensive evaluation
            console.print("\n[bold]Step 7: Model Evaluation[/bold]")
            evaluation_results = self.model_manager.evaluate_model(datasets)
            
            # Step 8: Report generation
            if self.config.generate_reports:
                console.print("\n[bold]Step 8: Report Generation[/bold]")
                training_time = (datetime.now() - start_time).total_seconds()
                self.report_generator.generate_comprehensive_report(
                    evaluation_results, self.model_manager, training_time
                )
            
            # Step 9: Save final model
            console.print("\n[bold]Step 9: Model Saving[/bold]")
            self.model_manager.trainer.save_model()
            console.print(f"[green]✓[/green] Model saved to {self.config.training.output_dir}")
            
            # Pipeline completion
            total_time = datetime.now() - start_time
            console.print(f"\n[bold green]🎉 Pipeline completed successfully in {total_time}![/bold green]")
            
            return {
                "training_results": final_training_results,
                "evaluation_results": evaluation_results,
                "cv_results": cv_results,
                "total_time": total_time.total_seconds()
            }
            
        except Exception as e:
            console.print(f"\n[bold red]❌ Pipeline failed: {str(e)}[/bold red]")
            console.print_exception()
            raise


def load_config_from_yaml(yaml_path: str) -> ExperimentConfig:
    """Load configuration from YAML file with comprehensive validation."""
    try:
        with open(yaml_path, 'r') as file:
            yaml_config = yaml.safe_load(file)
        
        # Create config with defaults
        config = ExperimentConfig()
        
        # Update with YAML values
        for section, values in yaml_config.items():
            if hasattr(config, section) and isinstance(values, dict):
                section_config = getattr(config, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
                    else:
                        console.print(f"[yellow]Warning: Unknown config key: {section}.{key}[/yellow]")
            elif hasattr(config, section):
                setattr(config, section, values)
            else:
                console.print(f"[yellow]Warning: Unknown config section: {section}[/yellow]")
        
        console.print(f"[green]✓[/green] Configuration loaded from {yaml_path}")
        return config
        
    except Exception as e:
        console.print(f"[red]✗[/red] Error loading config from {yaml_path}: {str(e)}")
        raise


def main():
    """Main entry point for the audio classification pipeline."""
    parser = argparse.ArgumentParser(
        description="Advanced Audio Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python audio_classification.py config.yaml
    python audio_classification.py config.yaml --verbose
        """
    )
    
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config_from_yaml(args.config_path)
        
        # Override verbose setting if specified
        if args.verbose:
            config.verbose = True
        
        # Create and run pipeline
        pipeline = PipelineOrchestrator(config)
        results = pipeline.run_pipeline()
        
        # Print final summary
        console.print(Panel.fit(
            f"[bold green]Pipeline Execution Complete![/bold green]\n"
            f"Total Time: {results['total_time']:.2f} seconds\n"
            f"Results saved to: {config.training.output_dir}",
            title="Success"
        ))
        
    except Exception as e:
        console.print(f"[bold red]Pipeline execution failed: {str(e)}[/bold red]")
        console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Example YAML Configuration File (config.yaml):
==============================================

experiment_name: "audio_classification_experiment"
verbose: true
use_cross_validation: false
cv_folds: 5
generate_reports: true
save_predictions: true

audio:
  sampling_rate: 16000
  max_length_seconds: 10.0
  min_length_seconds: 1.0
  normalize_audio: true
  apply_augmentation: true
  augmentation_probability: 0.3

model:
  model_name_or_path: "facebook/wav2vec2-base"
  freeze_feature_encoder: true
  attention_mask: true
  dropout_rate: 0.1
  hidden_size: 768
  num_attention_heads: 12

training:
  output_dir: "./results"
  num_train_epochs: 10
  per_device_train_batch_size: 8
  per_device_eval_batch_size: 16
  learning_rate: 2e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  evaluation_strategy: "epoch"
  save_strategy: "epoch"
  logging_steps: 100
  seed: 42
  fp16: true
  dataloader_num_workers: 4
  early_stopping_patience: 3
  metric_for_best_model: "eval_f1_macro"
  greater_is_better: true

data:
  dataset_name: "superb"
  dataset_config_name: "ks"
  audio_column_name: "audio"
  label_column_name: "label"
  train_split_ratio: 0.8
  val_split_ratio: 0.1
  test_split_ratio: 0.1
  max_train_samples: null
  max_eval_samples: null

optimization:
  enable_optimization: false
  n_trials: 50
  optimization_direction: "maximize"
  optimization_metric: "f1_macro"
  pruning_enabled: true
  study_name: "audio_classification_study"

Usage Instructions:
==================
1. Save the above YAML content to a file named 'config.yaml'
2. Run: python audio_classification.py config.yaml
3. The pipeline will automatically:
   - Load and preprocess the dataset
   - Train the model with the specified configuration
   - Evaluate the model comprehensively
   - Generate detailed reports and visualizations
   - Save the trained model and results

Key Features:
============
- Comprehensive data preprocessing with augmentation
- Advanced model training with early stopping
- Hyperparameter optimization using Optuna
- Cross-validation support
- Rich console output with progress tracking
- Detailed evaluation metrics and visualizations
- Modular and extensible architecture
- Production-ready error handling and logging
- YAML-based configuration management
- Automated report generation
"""