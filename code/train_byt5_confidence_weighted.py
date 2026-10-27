"""
ByT5 OCR Correction with Confidence-Weighted Training Loss
VERSION 2: Modified to match ungated baseline more closely

Changes from v1:
- batch_size: 4 → 2 (match ungated)
- warmup: warmup_steps=500 → warmup_ratio=0.1 (match ungated)
- Removed gradient_checkpointing (match ungated)
- Added identity pairs with weight=1.0 (match ungated)
- Generation params: added no_repeat_ngram_size=0, repetition_penalty=1.0, length_penalty=1.0, min_length
- Added dataloader_num_workers=0 (match ungated)
- Keep custom WeightedDataCollator (needed for weights)
- Keep custom WeightedSeq2SeqTrainer (needed for weighted loss)

Key difference from gated approach:
- Gated: uses confidence at INFERENCE time to skip/apply correction
- This: uses confidence at TRAINING time to weight the loss

Weighting strategy:
  - Correction pairs (original): weight = beta_calibrate(min_conf), clipped to [0.2, 1.0]
  - Identity pairs: weight = 1.0 (always learn to preserve)
  - Synthetic pairs: weight = 0.5 (auxiliary data)

Data:
- Training data: df_synthetic_better_random.pkl (Synthetic_Input -> Better_Ref)
- Confidence data: df_tagged_better.pkl (TokenConfidences)
- Uses existing split from alizethesis/split_info.json (same as ungated model)
"""

import os
import json
import logging
import random
import glob
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback
)
from tqdm import tqdm


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = "."  # Directory containing df_synthetic_better_random.pkl, df_tagged_better.pkl and split_info.json
SYNTH_DATA_PATH = os.path.join(BASE_DIR, "df_synthetic_better_random.pkl")
CONFIDENCE_DATA_PATH = os.path.join(BASE_DIR, "df_tagged_better.pkl")
SPLIT_INFO_PATH = os.path.join(BASE_DIR, "split_info.json")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "v2_matched_recal")  # Recalibrated with correct beta params from df_tagged_better.pkl


def set_seed(seed: int):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Match ungated's deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Training hyperparameters - matched to ungated baseline."""
    model_name: str = "google/byt5-small"
    max_input_length: int = 128
    max_target_length: int = 128
    batch_size: int = 2  # CHANGED: was 4, now matches ungated
    gradient_accumulation_steps: int = 4  # Effective batch size = 2 * 4 = 8
    learning_rate: float = 5e-5
    num_epochs: int = 10
    warmup_ratio: float = 0.1  # CHANGED: was warmup_steps=500, now matches ungated
    weight_decay: float = 0.01
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    output_dir: str = OUTPUT_DIR
    seed: int = 42
    # Confidence weighting parameters
    use_beta_calibration: bool = True  # Use beta calibration instead of focal
    beta_a: float = 0.13869323124087574  # Beta calibration parameter (from thresholding_v2/token_level on df_tagged_better.pkl)
    beta_b: float = 0.22760454627596885  # Beta calibration parameter (from thresholding_v2/token_level on df_tagged_better.pkl)
    weight_floor: float = 0.2  # Minimum weight
    weight_ceiling: float = 1.0  # Maximum weight
    synthetic_weight: float = 0.5  # Weight for synthetic examples
    identity_weight: float = 1.0  # Weight for identity pairs
    # Generation parameters (match ungated)
    num_beams: int = 4
    no_repeat_ngram_size: int = 0  # ADDED: match ungated
    repetition_penalty: float = 1.0  # ADDED: match ungated
    length_penalty: float = 1.0  # ADDED: match ungated
    # Identity pairs (match ungated)
    include_identity_pairs: bool = True
    identity_error_threshold: float = 0.05  # Include as identity if CER < 5%


# =============================================================================
# METRICS
# =============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein distance."""
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


def compute_cer(pred: str, ref: str) -> float:
    """Compute Character Error Rate."""
    if len(ref) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return levenshtein_distance(pred, ref) / len(ref)


# =============================================================================
# BETA CALIBRATION
# =============================================================================

def beta_calibrate(conf: float, a: float = 0.13869323124087574, b: float = 0.22760454627596885) -> float:
    """
    Transform raw confidence to calibrated error probability.

    Based on existing calibration from twostage/detection/thresholding/method1/
    p_error = sigmoid(a × logit(1 - conf) + b)
    """
    conf_clipped = np.clip(conf, 1e-6, 1 - 1e-6)
    logit = np.log((1 - conf_clipped) / conf_clipped)
    return 1 / (1 + np.exp(-(a * logit + b)))


# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_merge_data(logger):
    """Load synthetic data and merge confidence columns."""
    logger.info(f"Loading synthetic data from: {SYNTH_DATA_PATH}")
    synth_df = pd.read_pickle(SYNTH_DATA_PATH)
    logger.info(f"Loaded {len(synth_df)} sequences")
    logger.info(f"  - Synthetic: {synth_df['is_synthetic'].sum()}")
    logger.info(f"  - Original: {(~synth_df['is_synthetic']).sum()}")

    logger.info(f"\nLoading confidence data from: {CONFIDENCE_DATA_PATH}")
    conf_df = pd.read_pickle(CONFIDENCE_DATA_PATH)
    logger.info(f"Loaded {len(conf_df)} sequences with confidences")

    # Preprocess confidence data
    def preprocess_conf_row(row):
        result = {}
        tokens_str = row.get('Tokens', '')
        if isinstance(tokens_str, str):
            result['Tokens_Cleaned'] = tokens_str.split()
        else:
            result['Tokens_Cleaned'] = []

        result['TokenConfidences_Cleaned'] = row.get('TokenConfidences', [])
        return result

    conf_lookup = {}
    for idx, row in conf_df.iterrows():
        conf_lookup[idx] = preprocess_conf_row(row)

    # Add confidence columns
    synth_df['Tokens_Cleaned'] = None
    synth_df['TokenConfidences_Cleaned'] = None
    synth_df['min_confidence'] = None

    n_matched = 0
    for i, row in synth_df.iterrows():
        orig_idx = row['original_idx']
        if orig_idx in conf_lookup:
            confs = conf_lookup[orig_idx]['TokenConfidences_Cleaned']
            synth_df.at[i, 'Tokens_Cleaned'] = conf_lookup[orig_idx]['Tokens_Cleaned']
            synth_df.at[i, 'TokenConfidences_Cleaned'] = confs
            if isinstance(confs, list) and len(confs) > 0:
                synth_df.at[i, 'min_confidence'] = min(confs)
            n_matched += 1

    logger.info(f"Matched {n_matched} rows with confidence data")
    return synth_df, conf_df


def load_existing_split(df: pd.DataFrame, split_info_path: str, logger):
    """Load train/dev/test split from existing split_info.json."""
    logger.info(f"Loading split from: {split_info_path}")

    with open(split_info_path, 'r') as f:
        split_info = json.load(f)

    dev_indices = set(split_info['dev_indices'])
    test_indices = set(split_info['test_indices'])

    df_dev = df[df['original_idx'].isin(dev_indices)].copy()
    df_test = df[df['original_idx'].isin(test_indices)].copy()
    df_train = df[~df['original_idx'].isin(dev_indices | test_indices)].copy()

    # Filter synthetic from dev/test
    if 'is_synthetic' in df.columns:
        n_synthetic_train = df_train['is_synthetic'].sum()
        n_original_train = (~df_train['is_synthetic']).sum()
        logger.info(f"Train: {n_synthetic_train} synthetic + {n_original_train} original")

        df_dev = df_dev[~df_dev['is_synthetic']].copy()
        df_test = df_test[~df_test['is_synthetic']].copy()

    logger.info(f"Split: Train={len(df_train)}, Dev={len(df_dev)}, Test={len(df_test)}")
    return df_train, df_dev, df_test


# =============================================================================
# TRAINING PAIRS CREATION WITH WEIGHTS
# =============================================================================

def create_training_pairs_with_weights(
    df: pd.DataFrame,
    config: TrainingConfig,
    logger,
    include_identity: bool = True,
    split_name: str = "train",
) -> Tuple[List[Dict], np.ndarray]:
    """
    Create input-target pairs with confidence-based weights.

    Weight scheme:
    - Correction pairs (original): beta_calibrate(min_conf), clipped to [0.2, 1.0]
    - Identity pairs: 1.0 (always important to learn)
    - Synthetic pairs: 0.5 (auxiliary data)
    """

    pairs = []
    weights = []
    identity_count = 0
    correction_count = 0
    synthetic_count = 0

    for idx, row in df.iterrows():
        pred_text = row['Synthetic_Input']
        ref_text = row['Better_Ref']

        # Skip empty sequences
        if not pred_text or not ref_text:
            continue

        # Compute error rate (CER between input and target)
        error_rate = compute_cer(pred_text, ref_text)
        is_synthetic = row.get("is_synthetic", False)
        min_conf = row.get('min_confidence')

        # Compute weight for this example
        if is_synthetic:
            weight = config.synthetic_weight
            synthetic_count += 1
        else:
            if min_conf is not None and isinstance(min_conf, (int, float)):
                if config.use_beta_calibration:
                    # Beta-calibrated weight
                    error_prob = beta_calibrate(min_conf, config.beta_a, config.beta_b)
                    weight = np.clip(error_prob, config.weight_floor, config.weight_ceiling)
                else:
                    # Fallback: simple linear
                    weight = 1.0 - 0.3 * min_conf
            else:
                weight = config.synthetic_weight  # No confidence, use neutral

        # Add correction pair
        pairs.append({
            "input": pred_text,
            "target": ref_text,
            "type": "correction",
            "original_idx": row['original_idx'],
            "error_rate": error_rate,
        })
        weights.append(weight)
        correction_count += 1

        # Add identity pair for low-error sequences (ORIGINAL pairs only)
        if include_identity and error_rate < config.identity_error_threshold and not is_synthetic:
            pairs.append({
                "input": ref_text,
                "target": ref_text,
                "type": "identity",
                "original_idx": row['original_idx'],
                "error_rate": 0.0,
            })
            # Identity pairs get fixed weight = 1.0
            # They're always correct and we want the model to learn preservation
            weights.append(config.identity_weight)
            identity_count += 1

    weights = np.array(weights)

    logger.info(f"\n{split_name.upper()} pairs created: {len(pairs)}")
    logger.info(f"  Correction pairs: {correction_count}")
    logger.info(f"  Identity pairs:   {identity_count}")
    logger.info(f"  (Synthetic in corrections: {synthetic_count})")

    # Log weight statistics
    logger.info(f"\nWeight statistics for {split_name}:")
    logger.info(f"  Min: {weights.min():.4f}, Max: {weights.max():.4f}")
    logger.info(f"  Mean: {weights.mean():.4f}, Std: {weights.std():.4f}")
    logger.info(f"  Median: {np.median(weights):.4f}")

    # Show weight distribution by pair type
    correction_weights = weights[:correction_count]
    if identity_count > 0:
        identity_weights = weights[correction_count:]
        logger.info(f"  Correction weights: mean={correction_weights.mean():.4f}")
        logger.info(f"  Identity weights: mean={identity_weights.mean():.4f}")

    return pairs, weights


# =============================================================================
# WEIGHTED DATASET
# =============================================================================

class WeightedByT5Dataset(Dataset):
    """ByT5 dataset with per-sample confidence weights."""

    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        weights: np.ndarray,
        max_input_length: int = 128,
        max_target_length: int = 128,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.weights = weights
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        inputs = self.tokenizer(
            item["input"],
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        targets = self.tokenizer(
            item["target"],
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        labels = targets['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels,
            'weight': torch.tensor(self.weights[idx], dtype=torch.float32)
        }


class UnweightedByT5Dataset(Dataset):
    """Standard ByT5 dataset for dev/test (no weights needed)."""

    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_input_length: int = 128,
        max_target_length: int = 128,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        inputs = self.tokenizer(
            item["input"],
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        targets = self.tokenizer(
            item["target"],
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        labels = targets['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels
        }


# =============================================================================
# WEIGHTED DATA COLLATOR
# =============================================================================

class WeightedDataCollator(DataCollatorForSeq2Seq):
    """Data collator that preserves sample weights."""

    def __call__(self, features):
        # Extract weights before standard collation
        weights = None
        if 'weight' in features[0]:
            weights = torch.stack([f.pop('weight') for f in features])

        # Standard collation
        batch = super().__call__(features)

        # Add weights back
        if weights is not None:
            batch['weights'] = weights

        return batch


# =============================================================================
# WEIGHTED TRAINER
# =============================================================================

class WeightedSeq2SeqTrainer(Seq2SeqTrainer):
    """Seq2Seq trainer with confidence-weighted loss."""

    _debug_logged = False  # Class variable to log only once

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute weighted cross-entropy loss."""
        # Extract weights
        weights = inputs.pop('weights', None)

        # Debug logging (only first batch)
        if not WeightedSeq2SeqTrainer._debug_logged:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"[DEBUG] compute_loss called, weights present: {weights is not None}")
            if weights is not None:
                logger.info(f"[DEBUG] weights shape: {weights.shape}, values: {weights[:4]}")
            WeightedSeq2SeqTrainer._debug_logged = True

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs['labels']

        if weights is not None:
            # Compute per-token loss without reduction
            # NOTE: T5 is encoder-decoder, NOT decoder-only - no shifting needed!
            batch_size, seq_len, vocab_size = logits.shape
            flat_logits = logits.view(-1, vocab_size)
            flat_labels = labels.view(-1)

            # Per-token loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            per_token_loss = loss_fct(flat_logits, flat_labels)

            # Reshape to (batch_size, seq_len)
            per_token_loss = per_token_loss.view(batch_size, seq_len)

            # Compute per-example loss (mean over tokens, ignoring padding)
            mask = (labels != -100).float()
            per_example_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            # Apply sample weights
            # Normalize weights to sum to batch_size for stable training
            normalized_weights = weights * (batch_size / weights.sum())

            # Weighted mean loss
            loss = (per_example_loss * normalized_weights).mean()
        else:
            # Standard loss (for evaluation)
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


# =============================================================================
# DEV CER CALLBACK
# =============================================================================

class DevCERCallback(TrainerCallback):
    """Evaluate CER on dev set after each epoch."""

    def __init__(self, model, tokenizer, dev_df, config, logger):
        self.model = model
        self.tokenizer = tokenizer
        self.dev_df = dev_df
        self.config = config
        self.logger = logger
        self.epoch_metrics = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        self.logger.info(f"\n{'='*40}")
        self.logger.info(f"Epoch {epoch} - Evaluating CER on dev set")
        self.logger.info(f"{'='*40}")

        self.model.eval()

        cers = []
        sample_size = len(self.dev_df)

        for idx, row in tqdm(self.dev_df.iterrows(), total=sample_size, desc="Dev eval"):
            input_text = row['Synthetic_Input']
            ref_text = row['Better_Ref']

            inputs = self.tokenizer(
                input_text,
                return_tensors='pt',
                max_length=self.config.max_input_length,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=min(self.config.max_target_length, int(len(input_text) * 1.3) + 10),
                    min_length=max(1, int(len(input_text) * 0.5)),
                    num_beams=self.config.num_beams,
                    no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                    repetition_penalty=self.config.repetition_penalty,
                    early_stopping=True,
                    do_sample=False,
                )

            corrected_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            baseline_cer = compute_cer(input_text, ref_text)
            corrected_cer = compute_cer(corrected_text, ref_text)

            cers.append({
                "baseline": baseline_cer,
                "corrected": corrected_cer,
                "delta": baseline_cer - corrected_cer,
            })

        baseline_mean = np.mean([c["baseline"] for c in cers])
        corrected_mean = np.mean([c["corrected"] for c in cers])
        n_improved = sum(1 for c in cers if c["delta"] > 0)
        n_degraded = sum(1 for c in cers if c["delta"] < 0)

        metrics = {
            "epoch": epoch,
            "global_step": state.global_step,
            "dev_sample_size": sample_size,
            "baseline_cer": baseline_mean,
            "corrected_cer": corrected_mean,
            "delta_cer": baseline_mean - corrected_mean,
            "n_improved": n_improved,
            "n_degraded": n_degraded,
            "pct_improved": 100 * n_improved / len(cers) if cers else 0,
        }
        self.epoch_metrics.append(metrics)

        self.logger.info(f"  Baseline CER: {baseline_mean:.4f}")
        self.logger.info(f"  Corrected CER: {corrected_mean:.4f}")
        self.logger.info(f"  Improved: {n_improved}/{len(cers)} ({100*n_improved/len(cers):.1f}%)")
        self.logger.info(f"  Degraded: {n_degraded}/{len(cers)} ({100*n_degraded/len(cers):.1f}%)")

        self.model.train()
        return control


# =============================================================================
# CORRECTOR AND EVALUATION
# =============================================================================

class ByT5Corrector:
    """ByT5 model wrapper for correction."""

    def __init__(self, model_path: str, config: TrainingConfig, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def correct(self, text: str) -> str:
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.config.max_input_length,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=min(self.config.max_target_length, int(len(text) * 1.3) + 10),
                min_length=max(1, int(len(text) * 0.5)),
                num_beams=self.config.num_beams,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                repetition_penalty=self.config.repetition_penalty,
                length_penalty=self.config.length_penalty,
                early_stopping=True,
                do_sample=False,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def evaluate_on_test(df_test: pd.DataFrame, corrector: ByT5Corrector, logger) -> Tuple[Dict, pd.DataFrame]:
    """Evaluate on test set (no gating, standard correction)."""
    logger.info(f"Evaluating on {len(df_test)} test samples...")

    results = []
    for idx, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Test evaluation"):
        input_text = row['Synthetic_Input']
        ref_text = row['Better_Ref']
        corrected_text = corrector.correct(input_text)

        results.append({
            'RowIndex': row.get('original_idx', idx),
            'Pred_Original': input_text,
            'Pred_Corrected': corrected_text,
            'Ref': ref_text,
            'CER_Original': compute_cer(input_text, ref_text),
            'CER_Corrected': compute_cer(corrected_text, ref_text)
        })

    results_df = pd.DataFrame(results)

    metrics = {
        'total_samples': len(results_df),
        'cer_original_mean': float(results_df['CER_Original'].mean()),
        'cer_corrected_mean': float(results_df['CER_Corrected'].mean()),
        'cer_improvement': float(results_df['CER_Original'].mean() - results_df['CER_Corrected'].mean()),
        'improved': int((results_df['CER_Corrected'] < results_df['CER_Original']).sum()),
        'degraded': int((results_df['CER_Corrected'] > results_df['CER_Original']).sum()),
        'unchanged': int((results_df['CER_Corrected'] == results_df['CER_Original']).sum()),
        'made_perfect': int(((results_df['CER_Original'] > 0) & (results_df['CER_Corrected'] == 0)).sum())
    }

    return metrics, results_df


# =============================================================================
# TRAINING
# =============================================================================

def train_byt5_with_weighted_loss(
    train_pairs: List[Dict],
    train_weights: np.ndarray,
    dev_pairs: List[Dict],
    df_dev: pd.DataFrame,
    config: TrainingConfig,
    logger
) -> Tuple[str, List[Dict]]:
    """Train ByT5 with confidence-weighted loss."""

    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = T5ForConditionalGeneration.from_pretrained(config.model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    model = model.to(device)

    logger.info("\nCreating datasets...")
    train_dataset = WeightedByT5Dataset(
        train_pairs, tokenizer, train_weights,
        config.max_input_length, config.max_target_length
    )
    dev_dataset = UnweightedByT5Dataset(
        dev_pairs, tokenizer,
        config.max_input_length, config.max_target_length
    )

    logger.info(f"  Train: {len(train_dataset)} examples")
    logger.info(f"  Dev:   {len(dev_dataset)} examples")

    data_collator = WeightedDataCollator(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(config.output_dir, "checkpoints"),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,  # CHANGED: was warmup_steps
        logging_dir=os.path.join(config.output_dir, "logs"),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=config.num_epochs,
        predict_with_generate=True,
        seed=config.seed,
        data_seed=config.seed,
        report_to=[],
        bf16=True,
        fp16=False,
        max_grad_norm=0.5,
        dataloader_num_workers=0,  # ADDED: match ungated
        remove_unused_columns=False,  # CRITICAL: Keep 'weights' column for weighted loss!
        # NO gradient_checkpointing - match ungated
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_threshold=config.early_stopping_threshold,
    )

    dev_cer_callback = DevCERCallback(
        model=model,
        tokenizer=tokenizer,
        dev_df=df_dev,
        config=config,
        logger=logger
    )

    trainer = WeightedSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[early_stopping, dev_cer_callback]
    )

    logger.info(f"\n{'='*60}")
    logger.info("TRAINING CONFIGURATION")
    logger.info(f"{'='*60}")
    logger.info(f"  Epochs: {config.num_epochs}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Warmup ratio: {config.warmup_ratio}")
    logger.info(f"  Beta calibration: a={config.beta_a}, b={config.beta_b}")
    logger.info(f"  Weight bounds: [{config.weight_floor}, {config.weight_ceiling}]")
    logger.info(f"  Synthetic weight: {config.synthetic_weight}")
    logger.info(f"  Identity weight: {config.identity_weight}")

    logger.info(f"\n{'='*60}")
    logger.info("STARTING TRAINING WITH CONFIDENCE-WEIGHTED LOSS")
    logger.info(f"{'='*60}")

    trainer.train()

    epoch_metrics = dev_cer_callback.epoch_metrics

    # Select best model by CER
    logger.info(f"\n{'='*60}")
    logger.info("SELECTING BEST MODEL BY CER")
    logger.info(f"{'='*60}")

    best_model_path = os.path.join(config.output_dir, "best_model")

    if epoch_metrics:
        for m in epoch_metrics:
            logger.info(f"  Epoch {m['epoch']} (step {m['global_step']}): CER={m['corrected_cer']:.4f}")

        best_epoch_metrics = min(epoch_metrics, key=lambda x: x['corrected_cer'])
        best_step = best_epoch_metrics['global_step']
        logger.info(f"\n  Best epoch by CER: {best_epoch_metrics['epoch']} at step {best_step}")

        checkpoint_dir = os.path.join(config.output_dir, "checkpoints")
        best_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{best_step}")

        if os.path.exists(best_checkpoint_path):
            logger.info(f"  Loading checkpoint: {best_checkpoint_path}")
            model = T5ForConditionalGeneration.from_pretrained(best_checkpoint_path)
        else:
            checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
            if checkpoint_paths:
                def get_step(p):
                    try:
                        return int(os.path.basename(p).split("-")[-1])
                    except:
                        return 0
                valid = [(p, get_step(p)) for p in checkpoint_paths if get_step(p) <= best_step]
                if valid:
                    best_checkpoint = max(valid, key=lambda x: x[1])[0]
                    model = T5ForConditionalGeneration.from_pretrained(best_checkpoint)

    model.save_pretrained(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    logger.info(f"  Best model saved to: {best_model_path}")

    return best_model_path, epoch_metrics


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main pipeline."""

    config = TrainingConfig(
        model_name="google/byt5-small",
        batch_size=2,  # Match ungated
        gradient_accumulation_steps=4,
        num_epochs=10,
        learning_rate=5e-5,
        warmup_ratio=0.1,  # Match ungated
        output_dir=OUTPUT_DIR,
        # Weighting parameters
        use_beta_calibration=True,
        beta_a=0.13869323124087574,
        beta_b=0.22760454627596885,
        weight_floor=0.2,
        weight_ceiling=1.0,
        synthetic_weight=0.5,
        identity_weight=1.0,  # Identity pairs get full weight
        # Identity pairs
        include_identity_pairs=True,
        identity_error_threshold=0.05,
    )

    # Setup
    os.makedirs(config.output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    set_seed(config.seed)
    logger.info(f"Seed: {config.seed}")

    logger.info("=" * 60)
    logger.info("BYT5 OCR CORRECTION - WEIGHTED (V2 MATCHED TO UNGATED)")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now()}")
    logger.info(f"Output directory: {config.output_dir}")

    # =========================================================================
    # LOAD DATA
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("LOADING DATA")
    logger.info("="*60)

    synth_df, conf_df = load_and_merge_data(logger)
    df_train, df_dev, df_test = load_existing_split(synth_df, SPLIT_INFO_PATH, logger)

    # =========================================================================
    # ANALYZE CONFIDENCE DISTRIBUTION
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("CONFIDENCE DISTRIBUTION ANALYSIS")
    logger.info("="*60)

    train_original = df_train[~df_train['is_synthetic']] if 'is_synthetic' in df_train.columns else df_train
    min_confs = train_original['min_confidence'].dropna()

    if len(min_confs) > 0:
        logger.info(f"Train (original) min_confidence stats:")
        logger.info(f"  Count: {len(min_confs)}")
        logger.info(f"  Mean: {min_confs.mean():.4f}")
        logger.info(f"  Std: {min_confs.std():.4f}")
        logger.info(f"  Percentiles: 25%={min_confs.quantile(0.25):.4f}, 50%={min_confs.quantile(0.5):.4f}, 75%={min_confs.quantile(0.75):.4f}")

    # =========================================================================
    # CREATE TRAINING PAIRS WITH WEIGHTS
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("CREATING TRAINING PAIRS WITH WEIGHTS")
    logger.info("="*60)

    train_pairs, train_weights = create_training_pairs_with_weights(
        df_train, config, logger,
        include_identity=config.include_identity_pairs,
        split_name="train"
    )

    dev_pairs, _ = create_training_pairs_with_weights(
        df_dev, config, logger,
        include_identity=False,
        split_name="dev"
    )

    # =========================================================================
    # TRAIN MODEL
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("TRAINING WITH CONFIDENCE-WEIGHTED LOSS")
    logger.info("="*60)

    model_path, epoch_metrics = train_byt5_with_weighted_loss(
        train_pairs, train_weights, dev_pairs, df_dev, config, logger
    )

    with open(os.path.join(config.output_dir, 'epoch_metrics.json'), 'w') as f:
        json.dump(epoch_metrics, f, indent=2)

    # =========================================================================
    # EVALUATE ON TEST
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("EVALUATING ON TEST SET")
    logger.info("="*60)

    corrector = ByT5Corrector(model_path, config)
    metrics, results_df = evaluate_on_test(df_test, corrector, logger)

    logger.info("\n" + "-"*40)
    logger.info("TEST RESULTS")
    logger.info("-"*40)
    logger.info(f"Total samples: {metrics['total_samples']}")
    logger.info(f"CER original: {metrics['cer_original_mean']:.4f} ({metrics['cer_original_mean']*100:.2f}%)")
    logger.info(f"CER corrected: {metrics['cer_corrected_mean']:.4f} ({metrics['cer_corrected_mean']*100:.2f}%)")
    logger.info(f"CER improvement: {metrics['cer_improvement']:.4f} ({metrics['cer_improvement']*100:.2f} pts)")
    logger.info(f"Improved: {metrics['improved']}")
    logger.info(f"Degraded: {metrics['degraded']}")
    logger.info(f"Made perfect: {metrics['made_perfect']}")

    # Save results
    results_df.to_pickle(os.path.join(config.output_dir, 'test_results.pkl'))
    with open(os.path.join(config.output_dir, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(config.output_dir, 'training_config.json'), 'w') as f:
        json.dump(asdict(config), f, indent=2)

    # Save split info
    split_info = {
        'source': SPLIT_INFO_PATH,
        'timestamp': datetime.now().isoformat(),
        'train_pairs': len(train_pairs),
        'dev_pairs': len(dev_pairs),
        'test_size': len(df_test),
    }
    with open(os.path.join(config.output_dir, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)

    logger.info(f"\nAll results saved to {config.output_dir}/")
    logger.info(f"\nCompleted at: {datetime.now()}")


if __name__ == "__main__":
    main()
