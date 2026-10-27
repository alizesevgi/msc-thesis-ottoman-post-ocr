#!/usr/bin/env python3
"""
ByT5 OCR Post-Correction with Better Normalized Data

This script:
1. Loads df_synthetic_better_random.pkl (synthetic data with Better columns)
   - Uses Better_Pred/Better_Ref (normalized punctuation/spacing)
   - Synthetic inputs preserve punctuation (string-level manipulation)
2. Uses existing split from byt5_ungated_baseline/split_info.json
   - Train: includes synthetic augmentation pairs (negative indices)
   - Dev/Test: original pairs only (positive indices)
3. Fine-tunes google/byt5-small on OCR correction
4. Includes identity pairs (clean -> clean) for ORIGINAL pairs only
5. Selects best checkpoint by CER on full dev set
6. Evaluates on all splits and saves predictions

Columns used:
- Input:  Synthetic_Input (OCR output, with punctuation preserved)
- Target: Better_Ref (ground truth, normalized)
"""

import os
import glob
import json
import random
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
)
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Data
    "input_file": "df_synthetic_better_random.pkl",
    "output_dir": "byt5_ungated_cer_better_v2",
    "pred_column": "Synthetic_Input",
    "ref_column": "Better_Ref",

    # Use existing split from baseline model (for synthetic data compatibility)
    "existing_split_file": "byt5_ungated_baseline/split_info.json",

    # Model
    "model_name": "google/byt5-small",

    # Split (only used if existing_split_file is None)
    "test_size": 0.1,
    "dev_size": 0.1,  # 10% of remaining 90% = ~81/9/10
    "random_seed": 42,

    # Training
    "epochs": 10,
    "batch_size": 2,  # Reduced for shared GPU memory
    "learning_rate": 5e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 4,  # Effective batch size = 2 * 4 = 8
    "max_input_length": 128,
    "max_target_length": 128,

    # Early stopping
    "early_stopping_patience": 3,
    "early_stopping_threshold": 0.001,

    # Generation
    "num_beams": 4,
    "no_repeat_ngram_size": 0,  # Disabled - OCR correction needs to preserve repetitive text
    "repetition_penalty": 1.0,  # Disabled - was causing "soluya soluya" -> "soluya soğuya" errors
    "length_penalty": 1.0,

    # Identity pairs
    "include_identity_pairs": True,
    "identity_error_threshold": 0.05,  # Include as identity if error rate < 5%
}


# =============================================================================
# REPRODUCIBILITY - GLOBAL SEEDING
# =============================================================================

def set_global_seed(seed: int):
    """Set seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging to both file and console."""
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger("byt5_baseline")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    # File handler
    log_file = os.path.join(output_dir, "training.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - [%(levelname)s] %(message)s"
    ))
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s - [%(levelname)s] %(message)s"
    ))
    logger.addHandler(console_handler)

    return logger


# =============================================================================
# METRICS
# =============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def compute_cer(prediction: str, reference: str) -> float:
    """Compute Character Error Rate."""
    if len(reference) == 0:
        return 0.0 if len(prediction) == 0 else 1.0
    return levenshtein_distance(prediction, reference) / len(reference)


def compute_wer(prediction: str, reference: str) -> float:
    """Compute Word Error Rate."""
    pred_words = prediction.split()
    ref_words = reference.split()

    if len(ref_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0

    # Word-level Levenshtein
    return levenshtein_distance_words(pred_words, ref_words) / len(ref_words)


def levenshtein_distance_words(s1: List[str], s2: List[str]) -> int:
    """Compute Levenshtein distance at word level."""
    if len(s1) < len(s2):
        return levenshtein_distance_words(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, w1 in enumerate(s1):
        curr_row = [i + 1]
        for j, w2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (w1 != w2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


# =============================================================================
# DATASET
# =============================================================================

class OCRCorrectionDataset(Dataset):
    """Dataset for OCR correction with ByT5."""

    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_input_length: int = 512,
        max_target_length: int = 512,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize input
        input_encoding = self.tokenizer(
            item["input"],
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize target
        target_encoding = self.tokenizer(
            item["target"],
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Replace padding token id with -100 for loss calculation
        labels = target_encoding["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": labels,
        }


# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_and_prepare_data(
    config: Dict,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Load data and split using existing split_info.json OR create new splits.

    For synthetic data (df_synthetic_random.pkl):
    - Uses existing split from byt5_ungated_baseline/split_info.json
    - Dev/test indices refer to original rows (positive indices)
    - Train = all rows NOT in dev/test indices (includes synthetic rows with negative indices)
    """

    logger.info(f"Loading data from {config['input_file']}...")
    df = pd.read_pickle(config["input_file"])
    logger.info(f"Loaded {len(df)} sequences")

    # Verify columns exist
    required_cols = [config["pred_column"], config["ref_column"], "Book", "PageNum"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Books: {df['Book'].unique().tolist()}")

    # Compute error rate per sequence for identity pair selection
    # Use CER between input and target (since ErrorTags not available in Better dataframe)
    def compute_error_rate_from_cer(row):
        pred = row[config["pred_column"]]
        ref = row[config["ref_column"]]
        if not pred or not ref:
            return 0.0
        return compute_cer(pred, ref)

    logger.info("Computing error rates from CER...")
    df["error_rate"] = df.apply(compute_error_rate_from_cer, axis=1)

    # Log data statistics
    logger.info(f"\nData statistics:")
    logger.info(f"  Total sequences: {len(df)}")
    logger.info(f"  Mean error rate: {df['error_rate'].mean():.4f}")
    logger.info(f"  Sequences with 0% errors: {(df['error_rate'] == 0).sum()}")
    logger.info(f"  Sequences with <5% errors: {(df['error_rate'] < 0.05).sum()}")

    # Check if synthetic data columns exist
    if 'is_synthetic' in df.columns:
        n_synthetic = df['is_synthetic'].sum()
        n_original = (~df['is_synthetic']).sum()
        logger.info(f"  Original pairs: {n_original}")
        logger.info(f"  Synthetic pairs: {n_synthetic}")

    # =========================================================================
    # USE EXISTING SPLIT OR CREATE NEW
    # =========================================================================
    existing_split_file = config.get("existing_split_file")

    if existing_split_file and os.path.exists(existing_split_file):
        # Load existing split
        logger.info(f"\nLoading existing split from: {existing_split_file}")
        with open(existing_split_file, 'r') as f:
            existing_split = json.load(f)

        dev_indices = set(existing_split['dev_indices'])
        test_indices = set(existing_split['test_indices'])

        logger.info(f"  Dev indices: {len(dev_indices)}")
        logger.info(f"  Test indices: {len(test_indices)}")

        # Split: train = NOT in dev/test, dev/test = in their respective indices
        df_dev = df[df.index.isin(dev_indices)].copy()
        df_test = df[df.index.isin(test_indices)].copy()
        df_train = df[~df.index.isin(dev_indices | test_indices)].copy()

        logger.info(f"\nSplit sizes (using existing split):")
        logger.info(f"  Train: {len(df_train)} sequences (includes synthetic augmentation)")
        logger.info(f"  Dev:   {len(df_dev)} sequences (original only)")
        logger.info(f"  Test:  {len(df_test)} sequences (original only)")

        # Create split_info for this run
        split_info = {
            "random_seed": config["random_seed"],
            "split_strategy": "existing_split_with_synthetic",
            "existing_split_file": existing_split_file,
            "total_sequences": len(df),
            "split_sizes": {
                "train_sequences": len(df_train),
                "dev_sequences": len(df_dev),
                "test_sequences": len(df_test),
            },
            "train_indices": df_train.index.tolist(),
            "dev_indices": df_dev.index.tolist(),
            "test_indices": df_test.index.tolist(),
        }

    else:
        # Create new page-level splits (original behavior)
        logger.info(f"\nCreating PAGE-LEVEL stratified splits (seed={config['random_seed']})...")
        logger.info("  (Sequences from the same page stay together)")

        # Create unique page identifiers
        df["_page_id"] = df["Book"] + "_page_" + df["PageNum"].astype(str)

        # Get unique pages with their book for stratification
        pages_df = df.groupby("_page_id").agg({
            "Book": "first",
            "error_rate": "mean",
        }).reset_index()

        total_pages = len(pages_df)
        logger.info(f"  Total unique pages: {total_pages}")

        # First split: train+dev vs test (at page level)
        pages_traindev, pages_test = train_test_split(
            pages_df,
            test_size=config["test_size"],
            stratify=pages_df["Book"],
            random_state=config["random_seed"],
        )

        # Second split: train vs dev (at page level)
        adjusted_dev_size = config["dev_size"] / (1 - config["test_size"])
        pages_train, pages_dev = train_test_split(
            pages_traindev,
            test_size=adjusted_dev_size,
            stratify=pages_traindev["Book"],
            random_state=config["random_seed"],
        )

        # Map pages back to sequences
        train_page_ids = set(pages_train["_page_id"])
        dev_page_ids = set(pages_dev["_page_id"])
        test_page_ids = set(pages_test["_page_id"])

        df_train = df[df["_page_id"].isin(train_page_ids)].copy()
        df_dev = df[df["_page_id"].isin(dev_page_ids)].copy()
        df_test = df[df["_page_id"].isin(test_page_ids)].copy()

        # Remove temporary column
        df_train = df_train.drop(columns=["_page_id"])
        df_dev = df_dev.drop(columns=["_page_id"])
        df_test = df_test.drop(columns=["_page_id"])

        # Verify no page overlap
        train_pages = set(zip(df_train["Book"], df_train["PageNum"]))
        dev_pages = set(zip(df_dev["Book"], df_dev["PageNum"]))
        test_pages = set(zip(df_test["Book"], df_test["PageNum"]))

        assert len(train_pages & dev_pages) == 0, "Train/Dev page overlap!"
        assert len(train_pages & test_pages) == 0, "Train/Test page overlap!"
        assert len(dev_pages & test_pages) == 0, "Dev/Test page overlap!"
        logger.info("  Verified: No page overlap between splits")

        # Compile split info
        split_info = {
            "random_seed": config["random_seed"],
            "split_strategy": "page_level",
            "total_sequences": len(df),
            "total_pages": total_pages,
            "split_ratios": {
                "train": len(df_train) / len(df),
                "dev": len(df_dev) / len(df),
                "test": len(df_test) / len(df),
            },
            "split_sizes": {
                "train_sequences": len(df_train),
                "train_pages": len(pages_train),
                "dev_sequences": len(df_dev),
                "dev_pages": len(pages_dev),
                "test_sequences": len(df_test),
                "test_pages": len(pages_test),
            },
            "train_indices": df_train.index.tolist(),
            "dev_indices": df_dev.index.tolist(),
            "test_indices": df_test.index.tolist(),
            "per_book": {},
        }

        logger.info(f"\nSplit sizes:")
        logger.info(f"  Train: {len(df_train)} sequences, {len(pages_train)} pages ({100*len(df_train)/len(df):.1f}%)")
        logger.info(f"  Dev:   {len(df_dev)} sequences, {len(pages_dev)} pages ({100*len(df_dev)/len(df):.1f}%)")
        logger.info(f"  Test:  {len(df_test)} sequences, {len(pages_test)} pages ({100*len(df_test)/len(df):.1f}%)")

    # =========================================================================
    # VERIFY DEV/TEST CONTAIN NO SYNTHETIC ROWS
    # =========================================================================
    # This is critical for fair evaluation - synthetic rows should only be in train
    if 'is_synthetic' in df.columns:
        dev_synthetic = df_dev['is_synthetic'].sum() if 'is_synthetic' in df_dev.columns else 0
        test_synthetic = df_test['is_synthetic'].sum() if 'is_synthetic' in df_test.columns else 0

        if dev_synthetic > 0:
            raise ValueError(f"CRITICAL: Dev set contains {dev_synthetic} synthetic rows! "
                           "Synthetic data should only be in training set.")
        if test_synthetic > 0:
            raise ValueError(f"CRITICAL: Test set contains {test_synthetic} synthetic rows! "
                           "Synthetic data should only be in training set.")

        logger.info(f"\n  Verified: Dev and Test sets contain NO synthetic rows")

    return df_train, df_dev, df_test, split_info


def create_training_pairs(
    df: pd.DataFrame,
    config: Dict,
    logger: logging.Logger,
    include_identity: bool = True,
    split_name: str = "train",
) -> List[Dict]:
    """Create input-target pairs for training."""

    pairs = []
    identity_count = 0
    correction_count = 0

    for idx, row in df.iterrows():
        pred_text = row[config["pred_column"]]
        ref_text = row[config["ref_column"]]
        error_rate = row.get("error_rate", 0.0)

        # Skip empty sequences
        if not pred_text or not ref_text:
            continue

        # Add correction pair
        pairs.append({
            "input": pred_text,
            "target": ref_text,
            "type": "correction",
            "original_idx": idx,
            "error_rate": error_rate,
        })
        correction_count += 1

        # Add identity pair for low-error sequences (ORIGINAL pairs only)
        # Synthetic variants have stale error_rate from original row, so skip them
        is_synthetic = row.get("is_synthetic", False)
        if include_identity and error_rate < config["identity_error_threshold"] and not is_synthetic:
            pairs.append({
                "input": ref_text,
                "target": ref_text,
                "type": "identity",
                "original_idx": idx,
                "error_rate": 0.0,
            })
            identity_count += 1

    logger.info(f"\n{split_name.upper()} pairs created: {len(pairs)}")
    logger.info(f"  Correction pairs: {correction_count}")
    logger.info(f"  Identity pairs:   {identity_count}")

    return pairs


# =============================================================================
# CUSTOM CALLBACK FOR DEV EVALUATION
# =============================================================================

class DevEvaluationCallback(TrainerCallback):
    """Callback to compute CER/WER on FULL dev set after each epoch.

    Uses full dev set for accurate CER estimation during model selection.
    This is critical since we select the best checkpoint by CER.
    """

    def __init__(
        self,
        model,
        tokenizer,
        dev_df: pd.DataFrame,
        config: Dict,
        logger: logging.Logger,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dev_df = dev_df
        self.config = config
        self.logger = logger
        self.epoch_metrics = []

    def on_epoch_end(self, args, state, control, **kwargs):
        """Evaluate on FULL dev set after each epoch for accurate model selection."""
        epoch = int(state.epoch)
        self.logger.info(f"\n--- Epoch {epoch} Dev Evaluation (Full Dev Set) ---")

        # Use FULL dev set for accurate CER estimation (critical for model selection)
        # Previous version used only 100 samples with fixed seed, causing variance issues
        dev_sample = self.dev_df
        sample_size = len(dev_sample)

        self.model.eval()
        device = next(self.model.parameters()).device

        cers = []
        # Use tqdm for progress since we're evaluating the full dev set
        for idx, row in tqdm(dev_sample.iterrows(), total=sample_size, desc=f"Epoch {epoch} dev eval"):
            pred_text = row[self.config["pred_column"]]
            ref_text = row[self.config["ref_column"]]

            if not pred_text:
                continue

            inputs = self.tokenizer(
                pred_text,
                max_length=self.config["max_input_length"],
                truncation=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=min(self.config["max_target_length"], int(len(pred_text) * 1.3) + 10),
                    num_beams=self.config["num_beams"],
                    no_repeat_ngram_size=self.config["no_repeat_ngram_size"],
                    repetition_penalty=self.config["repetition_penalty"],
                    early_stopping=True,
                    do_sample=False,
                )

            corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            baseline_cer = compute_cer(pred_text, ref_text)
            corrected_cer = compute_cer(corrected, ref_text)
            cers.append({
                "baseline": baseline_cer,
                "corrected": corrected_cer,
                "delta": baseline_cer - corrected_cer,
            })

        # Compute stats
        baseline_mean = np.mean([c["baseline"] for c in cers])
        corrected_mean = np.mean([c["corrected"] for c in cers])
        delta_mean = np.mean([c["delta"] for c in cers])
        n_improved = sum(1 for c in cers if c["delta"] > 0)
        n_degraded = sum(1 for c in cers if c["delta"] < 0)

        # Store global_step for explicit epoch-to-checkpoint mapping
        metrics = {
            "epoch": epoch,
            "global_step": state.global_step,  # Track step for checkpoint selection
            "dev_sample_size": sample_size,
            "baseline_cer": baseline_mean,
            "corrected_cer": corrected_mean,
            "delta_cer": delta_mean,
            "n_improved": n_improved,
            "n_degraded": n_degraded,
            "pct_improved": 100 * n_improved / len(cers),
        }
        self.epoch_metrics.append(metrics)

        self.logger.info(f"  Dev set size: {sample_size} (full dev set)")
        self.logger.info(f"  Baseline CER: {baseline_mean:.4f}")
        self.logger.info(f"  Corrected CER: {corrected_mean:.4f}")
        self.logger.info(f"  Delta CER: {delta_mean:.4f} (positive = improvement)")
        self.logger.info(f"  Improved: {n_improved}/{len(cers)} ({100*n_improved/len(cers):.1f}%)")
        self.logger.info(f"  Degraded: {n_degraded}/{len(cers)} ({100*n_degraded/len(cers):.1f}%)")

        self.model.train()
        return control


# =============================================================================
# TRAINING
# =============================================================================

def train_model(
    train_data: List[Dict],
    dev_data: List[Dict],
    dev_df: pd.DataFrame,
    config: Dict,
    logger: logging.Logger,
) -> Tuple[T5ForConditionalGeneration, AutoTokenizer, List[Dict]]:
    """Train ByT5 model. Returns model, tokenizer, and epoch metrics."""

    logger.info(f"\n{'='*60}")
    logger.info("LOADING MODEL")
    logger.info(f"{'='*60}")
    logger.info(f"Model: {config['model_name']}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = T5ForConditionalGeneration.from_pretrained(config["model_name"])

    # Log model info
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {num_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    model = model.to(device)

    # Create datasets
    train_dataset = OCRCorrectionDataset(
        train_data,
        tokenizer,
        max_input_length=config["max_input_length"],
        max_target_length=config["max_target_length"],
    )

    dev_dataset = OCRCorrectionDataset(
        dev_data,
        tokenizer,
        max_input_length=config["max_input_length"],
        max_target_length=config["max_target_length"],
    )

    logger.info(f"\nDataset sizes:")
    logger.info(f"  Train: {len(train_dataset)} examples")
    logger.info(f"  Dev:   {len(dev_dataset)} examples")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(config["output_dir"], "checkpoints"),
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        logging_dir=os.path.join(config["output_dir"], "logs"),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,  # We'll select best by CER manually
        metric_for_best_model="eval_loss",  # Not used since load_best_model_at_end=False
        greater_is_better=False,
        save_total_limit=config["epochs"],  # Keep all epoch checkpoints for CER-based selection
        bf16=True,  # Use BF16 instead of FP16 - ByT5 is unstable with FP16
        fp16=False,
        max_grad_norm=0.5,  # More aggressive gradient clipping for stability
        report_to=[],
        dataloader_num_workers=0,
        seed=config["random_seed"],
        data_seed=config["random_seed"],
    )

    # Callbacks
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=config["early_stopping_patience"],
        early_stopping_threshold=config["early_stopping_threshold"],
    )

    dev_eval_callback = DevEvaluationCallback(
        model=model,
        tokenizer=tokenizer,
        dev_df=dev_df,
        config=config,
        logger=logger,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        callbacks=[early_stopping, dev_eval_callback],
    )

    # Log training config
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING CONFIGURATION")
    logger.info(f"{'='*60}")
    logger.info(f"  Epochs: {config['epochs']}")
    logger.info(f"  Batch size: {config['batch_size']}")
    logger.info(f"  Gradient accumulation: {config['gradient_accumulation_steps']}")
    logger.info(f"  Effective batch size: {config['batch_size'] * config['gradient_accumulation_steps']}")
    logger.info(f"  Learning rate: {config['learning_rate']}")
    logger.info(f"  Warmup ratio: {config['warmup_ratio']}")
    logger.info(f"  Weight decay: {config['weight_decay']}")
    logger.info(f"  Early stopping patience: {config['early_stopping_patience']}")
    logger.info(f"  Max input length: {config['max_input_length']}")
    logger.info(f"  Max target length: {config['max_target_length']}")

    # Train
    logger.info(f"\n{'='*60}")
    logger.info("STARTING TRAINING")
    logger.info(f"{'='*60}")

    train_result = trainer.train()

    logger.info(f"\n{'='*60}")
    logger.info("TRAINING COMPLETED")
    logger.info(f"{'='*60}")
    logger.info(f"  Total steps: {train_result.global_step}")
    logger.info(f"  Final training loss: {train_result.training_loss:.4f}")

    # Get epoch metrics from callback
    epoch_metrics = dev_eval_callback.epoch_metrics

    # Find best epoch by CER (lowest corrected_cer)
    logger.info(f"\n{'='*60}")
    logger.info("SELECTING BEST MODEL BY CER")
    logger.info(f"{'='*60}")

    if epoch_metrics:
        for m in epoch_metrics:
            logger.info(f"  Epoch {m['epoch']} (step {m['global_step']}): CER={m['corrected_cer']:.4f}, "
                       f"delta={m['delta_cer']:.4f}, improved={m['n_improved']}/{m['dev_sample_size']}")

        # Best = lowest corrected_cer
        best_epoch_metrics = min(epoch_metrics, key=lambda x: x['corrected_cer'])
        best_epoch = best_epoch_metrics['epoch']
        best_step = best_epoch_metrics['global_step']
        logger.info(f"\n  Best epoch by CER: {best_epoch} at step {best_step} "
                   f"(CER={best_epoch_metrics['corrected_cer']:.4f})")

        # Load checkpoint from best epoch using EXPLICIT global_step
        # This is more reliable than inferring from epoch index
        checkpoint_dir = os.path.join(config["output_dir"], "checkpoints")
        best_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{best_step}")

        if os.path.exists(best_checkpoint_path):
            logger.info(f"  Loading checkpoint: {best_checkpoint_path}")
            model = T5ForConditionalGeneration.from_pretrained(best_checkpoint_path)
            model = model.to(next(trainer.model.parameters()).device)
        else:
            # Fallback: find closest checkpoint if exact match doesn't exist
            # (could happen if early stopping triggered mid-epoch)
            checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))

            def get_step_num(path):
                try:
                    return int(os.path.basename(path).split("-")[-1])
                except ValueError:
                    return 0

            if checkpoint_paths:
                # Find checkpoint with step closest to (but not exceeding) best_step
                valid_checkpoints = [(p, get_step_num(p)) for p in checkpoint_paths
                                    if get_step_num(p) <= best_step]
                if valid_checkpoints:
                    best_checkpoint = max(valid_checkpoints, key=lambda x: x[1])[0]
                    logger.info(f"  Exact checkpoint not found. Loading closest: {best_checkpoint}")
                    model = T5ForConditionalGeneration.from_pretrained(best_checkpoint)
                    model = model.to(next(trainer.model.parameters()).device)
                else:
                    logger.warning(f"  No valid checkpoint found for step {best_step}, using final model")
                    model = trainer.model
            else:
                logger.warning(f"  No checkpoints found, using final trained model")
                model = trainer.model
    else:
        logger.info("  No epoch metrics available, using final trained model")
        model = trainer.model  # Use the trained model

    # Save best model (the one we selected by CER)
    best_model_path = os.path.join(config["output_dir"], "best_model")
    model.save_pretrained(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    logger.info(f"  Best model (by CER) saved to: {best_model_path}")

    return model, tokenizer, epoch_metrics


# =============================================================================
# INFERENCE
# =============================================================================

def generate_corrections(
    model: T5ForConditionalGeneration,
    tokenizer: AutoTokenizer,
    df: pd.DataFrame,
    config: Dict,
    logger: logging.Logger,
    split_name: str = "Test",
) -> pd.DataFrame:
    """Generate corrections for a dataframe."""

    device = next(model.parameters()).device
    model.eval()

    results = []

    logger.info(f"\nGenerating corrections for {split_name} set ({len(df)} sequences)...")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{split_name} inference"):
        pred_text = row[config["pred_column"]]
        ref_text = row[config["ref_column"]]
        book = row.get("Book", "Unknown")

        # Skip empty
        if not pred_text:
            corrected = ""
        else:
            # Tokenize
            inputs = tokenizer(
                pred_text,
                max_length=config["max_input_length"],
                truncation=True,
                return_tensors="pt",
            ).to(device)

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=min(config["max_target_length"], int(len(pred_text) * 1.3) + 10),
                    min_length=max(1, int(len(pred_text) * 0.5)),
                    num_beams=config["num_beams"],
                    no_repeat_ngram_size=config["no_repeat_ngram_size"],
                    repetition_penalty=config["repetition_penalty"],
                    length_penalty=config["length_penalty"],
                    early_stopping=True,
                    do_sample=False,
                )

            corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Compute metrics
        baseline_cer = compute_cer(pred_text, ref_text)
        corrected_cer = compute_cer(corrected, ref_text)
        baseline_wer = compute_wer(pred_text, ref_text)
        corrected_wer = compute_wer(corrected, ref_text)

        # Determine status
        delta_cer = baseline_cer - corrected_cer
        if delta_cer > 0.001:
            status = "improved"
        elif delta_cer < -0.001:
            status = "degraded"
        else:
            status = "unchanged"

        results.append({
            "original_idx": idx,
            "book": book,
            "pred": pred_text,
            "ref": ref_text,
            "corrected": corrected,
            "baseline_cer": baseline_cer,
            "corrected_cer": corrected_cer,
            "delta_cer": delta_cer,
            "baseline_wer": baseline_wer,
            "corrected_wer": corrected_wer,
            "delta_wer": baseline_wer - corrected_wer,
            "status": status,
        })

    return pd.DataFrame(results)


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_results(
    results_df: pd.DataFrame,
    logger: logging.Logger,
    split_name: str = "Test",
) -> Dict:
    """Compute aggregate metrics from results."""

    n_total = len(results_df)

    metrics = {
        "split": split_name,
        "n_sequences": n_total,

        # CER
        "baseline_cer_mean": float(results_df["baseline_cer"].mean()),
        "baseline_cer_std": float(results_df["baseline_cer"].std()),
        "baseline_cer_median": float(results_df["baseline_cer"].median()),
        "corrected_cer_mean": float(results_df["corrected_cer"].mean()),
        "corrected_cer_std": float(results_df["corrected_cer"].std()),
        "corrected_cer_median": float(results_df["corrected_cer"].median()),
        "delta_cer_mean": float(results_df["delta_cer"].mean()),
        "delta_cer_std": float(results_df["delta_cer"].std()),

        # WER
        "baseline_wer_mean": float(results_df["baseline_wer"].mean()),
        "baseline_wer_std": float(results_df["baseline_wer"].std()),
        "corrected_wer_mean": float(results_df["corrected_wer"].mean()),
        "corrected_wer_std": float(results_df["corrected_wer"].std()),
        "delta_wer_mean": float(results_df["delta_wer"].mean()),

        # Improvement breakdown
        "n_improved": int((results_df["status"] == "improved").sum()),
        "n_degraded": int((results_df["status"] == "degraded").sum()),
        "n_unchanged": int((results_df["status"] == "unchanged").sum()),
    }

    metrics["pct_improved"] = 100 * metrics["n_improved"] / n_total
    metrics["pct_degraded"] = 100 * metrics["n_degraded"] / n_total
    metrics["pct_unchanged"] = 100 * metrics["n_unchanged"] / n_total

    # CER reduction percentage
    if metrics["baseline_cer_mean"] > 0:
        metrics["cer_reduction_pct"] = 100 * (
            metrics["baseline_cer_mean"] - metrics["corrected_cer_mean"]
        ) / metrics["baseline_cer_mean"]
    else:
        metrics["cer_reduction_pct"] = 0.0

    # WER reduction percentage
    if metrics["baseline_wer_mean"] > 0:
        metrics["wer_reduction_pct"] = 100 * (
            metrics["baseline_wer_mean"] - metrics["corrected_wer_mean"]
        ) / metrics["baseline_wer_mean"]
    else:
        metrics["wer_reduction_pct"] = 0.0

    # Per-book metrics
    metrics["per_book"] = {}
    for book in results_df["book"].unique():
        book_df = results_df[results_df["book"] == book]
        metrics["per_book"][book] = {
            "n_sequences": len(book_df),
            "baseline_cer_mean": float(book_df["baseline_cer"].mean()),
            "corrected_cer_mean": float(book_df["corrected_cer"].mean()),
            "delta_cer_mean": float(book_df["delta_cer"].mean()),
            "n_improved": int((book_df["status"] == "improved").sum()),
            "n_degraded": int((book_df["status"] == "degraded").sum()),
            "pct_improved": 100 * (book_df["status"] == "improved").sum() / len(book_df),
        }

    # Log results
    logger.info(f"\n{'='*60}")
    logger.info(f"{split_name.upper()} EVALUATION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Sequences: {metrics['n_sequences']}")

    logger.info(f"\nCharacter Error Rate (CER):")
    logger.info(f"  Baseline:  {metrics['baseline_cer_mean']:.4f} (+/- {metrics['baseline_cer_std']:.4f})")
    logger.info(f"  Corrected: {metrics['corrected_cer_mean']:.4f} (+/- {metrics['corrected_cer_std']:.4f})")
    logger.info(f"  Delta:     {metrics['delta_cer_mean']:.4f} (positive = improvement)")
    logger.info(f"  Reduction: {metrics['cer_reduction_pct']:.2f}%")

    logger.info(f"\nWord Error Rate (WER):")
    logger.info(f"  Baseline:  {metrics['baseline_wer_mean']:.4f} (+/- {metrics['baseline_wer_std']:.4f})")
    logger.info(f"  Corrected: {metrics['corrected_wer_mean']:.4f} (+/- {metrics['corrected_wer_std']:.4f})")
    logger.info(f"  Delta:     {metrics['delta_wer_mean']:.4f}")
    logger.info(f"  Reduction: {metrics['wer_reduction_pct']:.2f}%")

    logger.info(f"\nImprovement Breakdown:")
    logger.info(f"  Improved:  {metrics['n_improved']:>5} ({metrics['pct_improved']:.1f}%)")
    logger.info(f"  Degraded:  {metrics['n_degraded']:>5} ({metrics['pct_degraded']:.1f}%)")
    logger.info(f"  Unchanged: {metrics['n_unchanged']:>5} ({metrics['pct_unchanged']:.1f}%)")

    logger.info(f"\nPer-Book Results:")
    for book, book_metrics in metrics["per_book"].items():
        logger.info(f"  {book}:")
        logger.info(f"    CER: {book_metrics['baseline_cer_mean']:.4f} -> {book_metrics['corrected_cer_mean']:.4f} (delta: {book_metrics['delta_cer_mean']:.4f})")
        logger.info(f"    Improved: {book_metrics['n_improved']}/{book_metrics['n_sequences']} ({book_metrics['pct_improved']:.1f}%)")

    return metrics


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""

    # Set global seed FIRST
    set_global_seed(CONFIG["random_seed"])

    # Setup logging
    logger = setup_logging(CONFIG["output_dir"])

    logger.info("=" * 60)
    logger.info("BYT5 OCR CORRECTION - UNGATED BASELINE")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now()}")
    logger.info(f"Output directory: {CONFIG['output_dir']}")
    logger.info(f"Random seed: {CONFIG['random_seed']}")
    logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Save config
    config_path = os.path.join(CONFIG["output_dir"], "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2)
    logger.info(f"Config saved to: {config_path}")

    # Load and split data
    logger.info(f"\n{'='*60}")
    logger.info("LOADING AND SPLITTING DATA")
    logger.info(f"{'='*60}")

    df_train, df_dev, df_test, split_info = load_and_prepare_data(CONFIG, logger)

    # Save split info
    split_info_path = os.path.join(CONFIG["output_dir"], "split_info.json")
    with open(split_info_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)
    logger.info(f"Split info saved to: {split_info_path}")

    # Create training pairs
    logger.info(f"\n{'='*60}")
    logger.info("CREATING TRAINING PAIRS")
    logger.info(f"{'='*60}")

    train_pairs = create_training_pairs(
        df_train,
        CONFIG,
        logger,
        include_identity=CONFIG["include_identity_pairs"],
        split_name="train",
    )

    dev_pairs = create_training_pairs(
        df_dev,
        CONFIG,
        logger,
        include_identity=False,  # No identity pairs in dev - evaluate on real corrections only
        split_name="dev",
    )

    # Compute baseline metrics
    logger.info(f"\n{'='*60}")
    logger.info("BASELINE METRICS (before training)")
    logger.info(f"{'='*60}")

    baseline_metrics = {}
    for split_name, split_df in [("Train", df_train), ("Dev", df_dev), ("Test", df_test)]:
        # For training set, separate original vs synthetic metrics
        if split_name == "Train" and 'is_synthetic' in split_df.columns:
            # Original training pairs
            df_original = split_df[~split_df['is_synthetic']]
            cers_original = [
                compute_cer(row[CONFIG["pred_column"]], row[CONFIG["ref_column"]])
                for _, row in df_original.iterrows()
            ]
            wers_original = [
                compute_wer(row[CONFIG["pred_column"]], row[CONFIG["ref_column"]])
                for _, row in df_original.iterrows()
            ]

            # Synthetic training pairs
            df_synthetic = split_df[split_df['is_synthetic']]
            cers_synthetic = [
                compute_cer(row[CONFIG["pred_column"]], row[CONFIG["ref_column"]])
                for _, row in df_synthetic.iterrows()
            ]
            wers_synthetic = [
                compute_wer(row[CONFIG["pred_column"]], row[CONFIG["ref_column"]])
                for _, row in df_synthetic.iterrows()
            ]

            baseline_metrics["train_original"] = {
                "n_sequences": len(df_original),
                "cer_mean": np.mean(cers_original),
                "cer_std": np.std(cers_original),
                "wer_mean": np.mean(wers_original),
                "wer_std": np.std(wers_original),
            }
            baseline_metrics["train_synthetic"] = {
                "n_sequences": len(df_synthetic),
                "cer_mean": np.mean(cers_synthetic) if cers_synthetic else 0.0,
                "cer_std": np.std(cers_synthetic) if cers_synthetic else 0.0,
                "wer_mean": np.mean(wers_synthetic) if wers_synthetic else 0.0,
                "wer_std": np.std(wers_synthetic) if wers_synthetic else 0.0,
            }

            logger.info(f"{split_name} (Original pairs: {len(df_original)}):")
            logger.info(f"  CER: {np.mean(cers_original):.4f} (+/- {np.std(cers_original):.4f})")
            logger.info(f"  WER: {np.mean(wers_original):.4f} (+/- {np.std(wers_original):.4f})")
            logger.info(f"{split_name} (Synthetic pairs: {len(df_synthetic)}):")
            if cers_synthetic:
                logger.info(f"  CER: {np.mean(cers_synthetic):.4f} (+/- {np.std(cers_synthetic):.4f})")
                logger.info(f"  WER: {np.mean(wers_synthetic):.4f} (+/- {np.std(wers_synthetic):.4f})")
                logger.info(f"  Note: Synthetic CER is lower because inputs are partially corrected")
            else:
                logger.info(f"  (no synthetic pairs)")

            # Also compute combined for backwards compatibility
            all_cers = cers_original + cers_synthetic
            all_wers = wers_original + wers_synthetic
            baseline_metrics["train"] = {
                "n_sequences": len(split_df),
                "cer_mean": np.mean(all_cers),
                "cer_std": np.std(all_cers),
                "wer_mean": np.mean(all_wers),
                "wer_std": np.std(all_wers),
            }
        else:
            # Dev and Test sets (original pairs only)
            cers = [
                compute_cer(row[CONFIG["pred_column"]], row[CONFIG["ref_column"]])
                for _, row in split_df.iterrows()
            ]
            wers = [
                compute_wer(row[CONFIG["pred_column"]], row[CONFIG["ref_column"]])
                for _, row in split_df.iterrows()
            ]
            baseline_metrics[split_name.lower()] = {
                "n_sequences": len(split_df),
                "cer_mean": np.mean(cers),
                "cer_std": np.std(cers),
                "wer_mean": np.mean(wers),
                "wer_std": np.std(wers),
            }
            logger.info(f"{split_name}:")
            logger.info(f"  CER: {np.mean(cers):.4f} (+/- {np.std(cers):.4f})")
            logger.info(f"  WER: {np.mean(wers):.4f} (+/- {np.std(wers):.4f})")

    # Train model
    model, tokenizer, epoch_metrics = train_model(
        train_pairs, dev_pairs, df_dev, CONFIG, logger
    )

    # Save epoch metrics
    epoch_metrics_path = os.path.join(CONFIG["output_dir"], "epoch_metrics.json")
    with open(epoch_metrics_path, "w", encoding="utf-8") as f:
        json.dump(epoch_metrics, f, indent=2)
    logger.info(f"Epoch metrics saved to: {epoch_metrics_path}")

    # Evaluate on all splits
    all_metrics = {"baseline": baseline_metrics}

    for split_name, split_df in [("Train", df_train), ("Dev", df_dev), ("Test", df_test)]:
        logger.info(f"\n{'='*60}")
        logger.info(f"EVALUATING ON {split_name.upper()}")
        logger.info(f"{'='*60}")

        results_df = generate_corrections(
            model, tokenizer, split_df, CONFIG, logger, split_name
        )

        # Save predictions
        predictions_path = os.path.join(
            CONFIG["output_dir"],
            f"predictions_{split_name.lower()}.csv"
        )
        results_df.to_csv(predictions_path, index=False, encoding="utf-8")
        logger.info(f"Predictions saved to: {predictions_path}")

        # Compute and log metrics
        metrics = evaluate_results(results_df, logger, split_name)
        all_metrics[split_name.lower()] = metrics

    # Save all metrics
    metrics_path = os.path.join(CONFIG["output_dir"], "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"\nAll metrics saved to: {metrics_path}")

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*60}")

    test_metrics = all_metrics["test"]
    logger.info(f"\nTest Set Results:")
    logger.info(f"  CER: {test_metrics['baseline_cer_mean']:.4f} -> {test_metrics['corrected_cer_mean']:.4f}")
    logger.info(f"  CER Reduction: {test_metrics['cer_reduction_pct']:.2f}%")
    logger.info(f"  WER: {test_metrics['baseline_wer_mean']:.4f} -> {test_metrics['corrected_wer_mean']:.4f}")
    logger.info(f"  WER Reduction: {test_metrics['wer_reduction_pct']:.2f}%")
    logger.info(f"\n  Improved: {test_metrics['pct_improved']:.1f}%")
    logger.info(f"  Degraded: {test_metrics['pct_degraded']:.1f}%")
    logger.info(f"  Unchanged: {test_metrics['pct_unchanged']:.1f}%")

    logger.info(f"\nOutput files:")
    for f in sorted(os.listdir(CONFIG["output_dir"])):
        fpath = os.path.join(CONFIG["output_dir"], f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            logger.info(f"  {f} ({size:,} bytes)")

    logger.info(f"\nCompleted at: {datetime.now()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
