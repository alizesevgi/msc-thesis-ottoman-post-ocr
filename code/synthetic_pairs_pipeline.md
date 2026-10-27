# Synthetic Training Pair Generation Pipeline

## Table of Contents
1. [Project Context](#project-context)
2. [Problem Statement](#problem-statement)
3. [Solution Overview](#solution-overview)
4. [Input Data](#input-data)
5. [Algorithm Deep Dive](#algorithm-deep-dive)
6. [Data Flow Diagram](#data-flow-diagram)
7. [Output Format](#output-format)
8. [Usage in Model Training](#usage-in-model-training)

---

## Project Context

This pipeline is part of an **OCR post-correction system for historical Turkish texts**. The broader project uses a sequence-to-sequence model (ByT5) to correct errors produced by an OCR engine when digitizing Ottoman/historical Turkish documents.

### The Two-Stage Problem

1. **Stage 1 (OCR)**: An OCR engine reads scanned document images and produces text predictions
2. **Stage 2 (Post-Correction)**: A neural network corrects systematic OCR errors

This pipeline generates training data for **Stage 2** - the post-correction model.

### Why Synthetic Data?

Real OCR errors are limited to whatever mistakes the OCR engine actually makes. By synthetically generating "partially corrected" versions of OCR outputs, we can:

- **Augment training data** (up to two partially corrected variants per training sequence)
- **Control error distribution** in training
- **Teach the model incremental correction** (not just full correction)

---

## Problem Statement

### The Trivial Error Problem

OCR post-correction models trained on raw (prediction, reference) pairs often learn to prioritize **trivial fixes**:

| Error Type | Example | Semantic Impact |
|------------|---------|-----------------|
| Case change | `The` → `the` | None |
| Punctuation | `word,` → `word.` | Minimal |
| Spacing | `word word` → `word  word` | Minimal |
| **Diacritic loss** | `â` → `a` | **HIGH** (changes meaning) |
| **Vowel confusion** | `i` → `ı` | **HIGH** (different letters in Turkish) |

Models naturally gravitate toward easy patterns. A model can reduce its error rate by fixing case and punctuation while ignoring the diacritic errors that matter more for readability.

### Turkish-Specific Challenges

Turkish has unique orthographic features that OCR systems struggle with:

| Character Pair | Issue |
|----------------|-------|
| `i` ↔ `ı` | Dotted vs. dotless i (different vowels) |
| `â` ↔ `a` | Circumflex a (affects pronunciation) |
| `î` ↔ `i` | Circumflex i |
| `ç`, `ş`, `ğ`, `ö`, `ü` | Diacritics often lost in OCR |

---

## Solution Overview

The pipeline implements **priority-weighted synthetic data generation**:

1. **Classify each OCR error by priority** (HIGH/MEDIUM/LOW)
2. **Generate synthetic training pairs** by randomly "fixing" some errors
3. **Weight the random selection** so LOW priority errors are fixed more often
4. **Result**: Training data has more examples with HIGH priority errors remaining

The model therefore sees more training examples in which the important errors are still present.

---

## Input Data

### Primary Input: `df_tagged_better.pkl`

A pandas DataFrame with aligned OCR predictions and ground truth references.

| Column | Type | Description |
|--------|------|-------------|
| `Better_Pred` | str | OCR engine output (the "source" text with errors) |
| `Better_Ref` | str | Ground truth reference (the "target" correct text) |
| `Book` | str | Source book identifier |
| `PageNum` | int | Page number in source book |

**Note**: The "Better" prefix indicates these columns come from an improved alignment process (vs. raw OCR output).

### Split Information: `split_info.json`

```json
{
  "train_indices": [0, 1, 2, ...],
  "dev_indices": [1000, 1001, ...],
  "test_indices": [2000, 2001, ...]
}
```

This ensures consistent train/dev/test splits across all experiments.

---

## Algorithm Deep Dive

### Step 1: Error Priority Classification

Every character-level difference between `pred` and `ref` is classified into one of three priority levels.

#### Priority Definitions

```python
# HIGH PRIORITY - Turkish diacritics and common OCR confusions
TOP_PRIORITY_PAIRS = {
    ('â', 'a'), ('a', 'â'),  # circumflex a
    ('î', 'i'), ('i', 'î'),  # circumflex i
    ('i', 'ı'), ('ı', 'i'),  # dotted/dotless i (CRITICAL for Turkish)
    ('e', 'a'), ('a', 'e'),  # visual similarity confusion
    ('e', 'i'), ('i', 'e'),  # vowel confusion
    # ... plus uppercase versions
}

# Characters that indicate diacritic involvement
DIACRITIC_CHARS = {'Â', 'Ç', 'Î', 'Ö', 'Û', 'Ü', 'â', 'ç', 'î', 'ö', 'û', 'ü',
                   'ğ', 'İ', 'ı', 'Ş', 'ş', 'Ğ'}

# LOW PRIORITY - trivial errors
LOW_PRIORITY_CHARS = {' ', '\t', '\n', '.', ',', ';', ':', '!', '?',
                      '(', ')', '[', ']', '"', "'", '-', '–', '—', '/'}
```

#### Classification Logic

```python
def classify_diff_priority(diff):
    pred_char = diff['pred_text']
    ref_char = diff['ref_text']

    # Single character substitution
    if len(pred_char) == 1 and len(ref_char) == 1:

        # Check if it's a top priority pair
        if (pred_char, ref_char) in TOP_PRIORITY_PAIRS:
            return 'high'

        # Case-only difference (e.g., 'A' → 'a')
        if pred_char.lower() == ref_char.lower():
            return 'low'

        # Punctuation or spacing
        if pred_char in LOW_PRIORITY_CHARS or ref_char in LOW_PRIORITY_CHARS:
            return 'low'

        # Any diacritic involvement
        if pred_char in DIACRITIC_CHARS or ref_char in DIACRITIC_CHARS:
            return 'medium'

        # Other vowel confusion
        if pred_char in VOWELS and ref_char in VOWELS:
            return 'medium'

        # Consonant errors
        return 'medium'

    # Multi-character diffs: check for diacritic content
    # ... (similar logic)
```

### Step 2: Diff Detection

Using Python's `difflib.SequenceMatcher`, we find all character-level differences:

```python
def get_string_diffs(pred, ref):
    """
    Find all differences between pred and ref strings.

    Example:
        pred = "Türkiye'nin başkenti"
        ref  = "Türkiye'nin başkenti"

        If pred had "Turkiye" instead of "Türkiye", we'd get:
        [{'pred_start': 1, 'pred_end': 2, 'pred_text': 'u',
          'ref_start': 1, 'ref_end': 2, 'ref_text': 'ü', 'tag': 'replace'}]
    """
    matcher = SequenceMatcher(None, pred, ref)
    diffs = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            diffs.append({
                'pred_start': i1,    # Start position in pred
                'pred_end': i2,      # End position in pred
                'ref_start': j1,     # Start position in ref
                'ref_end': j2,       # End position in ref
                'pred_text': pred[i1:i2],  # What's in pred
                'ref_text': ref[j1:j2],    # What should be there
                'tag': tag           # 'replace', 'insert', or 'delete'
            })

    return diffs
```

### Step 3: Diff Filtering

Not all differences are training-worthy. We filter:

```python
def filter_diffs(diffs):
    """
    Keep: Content errors, izafet patterns
    Exclude: Punctuation-only differences
    Returns (filtered_diffs, number_excluded).
    """
    filtered = []
    excluded = 0

    for diff in diffs:
        # Always keep izafet (Turkish grammatical hyphenation)
        if is_izafet_diff(diff['pred_text'], diff['ref_text']):
            diff['reason'] = 'izafet'
            diff['priority'] = classify_diff_priority(diff)
            filtered.append(diff)

        # Exclude pure punctuation differences
        elif is_punctuation_only_diff(diff['pred_text'], diff['ref_text']):
            excluded += 1

        # Keep all other content differences
        else:
            diff['reason'] = 'content'
            diff['priority'] = classify_diff_priority(diff)
            filtered.append(diff)

    return filtered, excluded
```

**Izafet**: A Turkish grammatical construct where suffixes are connected with hyphens (e.g., `devlet-i`). These are semantically important and always kept.

### Step 4: Weighted Random Sampling

This is the central step. When selecting which errors to "fix" in synthetic variants, we use weighted sampling:

```python
PRIORITY_FIX_WEIGHTS = {
    'high': 0.3,    # Lowest relative weight: least likely to be picked for fixing
    'medium': 1.0,  # Baseline
    'low': 3.0,     # 3x the medium weight: most likely to be picked for fixing
}
```

The weights are relative, not probabilities. At each draw, an error is picked with probability equal to its weight divided by the total weight of the errors not yet picked, so a HIGH error's chance of being fixed depends on which other errors are in the same sequence.

#### The Weighted Sampling Algorithm

```python
def weighted_sample(items, weights, k):
    """
    Sample k items without replacement, weighted by weights.
    Higher weight = more likely to be selected.

    This is NOT random.choices() - we need without replacement.
    """
    if k >= len(items):
        return items[:]

    items = list(items)
    weights = list(weights)
    selected = []

    for _ in range(k):
        # Calculate total remaining weight
        total = sum(weights)
        if total == 0:
            break

        # Random point in weight space
        r = random.random() * total

        # Find which item this corresponds to
        cumsum = 0
        for i, w in enumerate(weights):
            cumsum += w
            if r <= cumsum:
                selected.append(items[i])
                # Remove selected item (no replacement)
                items.pop(i)
                weights.pop(i)
                break

    return selected
```

### Step 5: Synthetic Variant Generation

For each original (pred, ref) pair, we keep the original pair and add up to two synthetic variants. Sequences with fewer than two filtered errors get no variants, and a sequence with exactly two errors gets one.

```python
def generate_variants_random_weighted(pred, ref, max_variants=2):
    """
    Generate synthetic training pairs using weighted random selection.

    Returns: List of tuples
        (synthetic_input, target, pair_type, n_remaining, n_fixed,
         fixed_indices, high_priority_remaining)
    """
    # Get all errors
    all_diffs = get_string_diffs(pred, ref)
    filtered_diffs, _ = filter_diffs(all_diffs)
    n_errors = len(filtered_diffs)

    # Count initial priorities
    init_counts = {'high': 0, 'medium': 0, 'low': 0}
    for d in filtered_diffs:
        init_counts[d.get('priority', 'medium')] += 1

    # Start with original pair (no fixes applied)
    pairs = [(pred, ref, 'original', n_errors, 0, [], init_counts['high'])]

    # Need at least 2 errors to create meaningful variants
    if n_errors < 2:
        return pairs

    # At most n_errors - 1 variants, so a 2-error sequence gets one variant
    n_variants = min(max_variants, n_errors - 1)

    for v in range(n_variants):
        # Randomly decide how many errors to fix (at least 1, at most n-1)
        n_to_fix = random.randint(1, n_errors - 1)

        # Build weights: LOW priority → high weight → more likely to fix
        fix_weights = [PRIORITY_FIX_WEIGHTS[d.get('priority', 'medium')] for d in filtered_diffs]

        # Weighted sample which diffs to fix
        diffs_to_fix = weighted_sample(filtered_diffs, fix_weights, n_to_fix)
        fixed_indices = sorted([filtered_diffs.index(d) for d in diffs_to_fix])

        # Apply the fixes to create synthetic input
        synthetic_input = apply_corrections(pred, diffs_to_fix)

        # Count remaining priorities
        priority_counts = count_priority_remaining(filtered_diffs, diffs_to_fix)

        pairs.append((
            synthetic_input,           # What model sees as input
            ref,                       # Target is always the reference
            f'random_{v+1}',           # Pair type identifier
            n_errors - n_to_fix,       # Errors remaining
            n_to_fix,                  # Errors fixed
            fixed_indices,             # Which filtered diffs were fixed
            priority_counts['high']    # HIGH errors still present
        ))

    return pairs
```

### Step 6: Applying Corrections

When we "fix" errors to create synthetic inputs, we must apply them in reverse order to avoid offset issues:

```python
def apply_corrections(pred, diffs_to_fix):
    """
    Apply selected corrections to pred string.

    CRITICAL: Sort by position DESCENDING to avoid offset issues.
    A fix can change the string's length. If we fix position 5 first,
    position 10 may shift. If we fix position 10 first, position 5 is
    still valid.
    """
    result = pred

    # Sort by pred_start in descending order
    for diff in sorted(diffs_to_fix, key=lambda x: x['pred_start'], reverse=True):
        # Replace the pred_text with ref_text
        result = result[:diff['pred_start']] + diff['ref_text'] + result[diff['pred_end']:]

    return result
```

#### Example

```
Original pred: "Turkiye'nin baskenti Ankara'dir"
Reference:     "Türkiye'nin başkenti Ankara'dır"

Diffs found:
  [0] pos 1-2:   'u' → 'ü' (MEDIUM - diacritic, not in TOP_PRIORITY_PAIRS)
  [1] pos 14-15: 's' → 'ş' (MEDIUM - diacritic)
  [2] pos 29-30: 'i' → 'ı' (HIGH - dotted/dotless)

With 3 errors, up to 2 variants are made, each fixing 1 or 2 errors.
Suppose a variant fixes [1]:

Synthetic input: "Turkiye'nin başkenti Ankara'dir"
                   ^-- 'u' still wrong (MEDIUM)
                                               ^-- 'i' still wrong (HIGH)

Target: "Türkiye'nin başkenti Ankara'dır"
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INPUT LAYER                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────┐         ┌──────────────────────┐                  │
│  │  df_tagged_better.pkl │         │   split_info.json    │                  │
│  │                      │         │                      │                  │
│  │  - Better_Pred (str) │         │  - train_indices []  │                  │
│  │  - Better_Ref (str)  │         │  - dev_indices []    │                  │
│  │  - Book, PageNum     │         │  - test_indices []   │                  │
│  └──────────┬───────────┘         └──────────┬───────────┘                  │
│             │                                │                               │
└─────────────┼────────────────────────────────┼───────────────────────────────┘
              │                                │
              ▼                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SPLIT & ROUTE                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│         ┌─────────────────┬─────────────────┬─────────────────┐             │
│         │                 │                 │                 │             │
│         ▼                 ▼                 ▼                 │             │
│   ┌───────────┐     ┌───────────┐     ┌───────────┐          │             │
│   │   TRAIN   │     │    DEV    │     │   TEST    │          │             │
│   │  indices  │     │  indices  │     │  indices  │          │             │
│   └─────┬─────┘     └─────┬─────┘     └─────┬─────┘          │             │
│         │                 │                 │                 │             │
└─────────┼─────────────────┼─────────────────┼─────────────────┘             │
          │                 │                 │                               │
          ▼                 │                 │                               │
┌─────────────────────────┐ │                 │                               │
│  SYNTHETIC GENERATION   │ │                 │                               │
├─────────────────────────┤ │                 │                               │
│                         │ │                 │                               │
│  For each train row:    │ │                 │                               │
│                         │ │                 │                               │
│  ┌───────────────────┐  │ │                 │                               │
│  │ 1. GET DIFFS      │  │ │                 │                               │
│  │                   │  │ │                 │                               │
│  │ pred: "Turkiye"   │  │ │                 │                               │
│  │ ref:  "Türkiye"   │  │ │                 │                               │
│  │         ↓         │  │ │                 │                               │
│  │ diff: u→ü @ pos 1 │  │ │                 │                               │
│  └─────────┬─────────┘  │ │                 │                               │
│            ▼            │ │                 │                               │
│  ┌───────────────────┐  │ │                 │                               │
│  │ 2. CLASSIFY       │  │ │                 │                               │
│  │                   │  │ │                 │                               │
│  │ u→ü : diacritic   │  │ │                 │                               │
│  │     → MEDIUM      │  │ │                 │                               │
│  │                   │  │ │                 │                               │
│  │ i→ı : dotted i    │  │ │                 │                               │
│  │     → HIGH        │  │ │                 │                               │
│  └─────────┬─────────┘  │ │                 │                               │
│            ▼            │ │                 │                               │
│  ┌───────────────────┐  │ │                 │                               │
│  │ 3. WEIGHTED SAMPLE│  │ │                 │                               │
│  │                   │  │ │                 │                               │
│  │ Fix weights:      │  │ │                 │                               │
│  │  HIGH:   0.3      │  │ │                 │                               │
│  │  MEDIUM: 1.0      │  │ │                 │                               │
│  │  LOW:    3.0      │  │ │                 │                               │
│  │                   │  │ │                 │                               │
│  │ Sample which to   │  │ │                 │                               │
│  │ fix (weighted)    │  │ │                 │                               │
│  └─────────┬─────────┘  │ │                 │                               │
│            ▼            │ │                 │                               │
│  ┌───────────────────┐  │ │                 │                               │
│  │ 4. GENERATE       │  │ │                 │                               │
│  │    VARIANTS       │  │ │                 │                               │
│  │                   │  │ │                 │                               │
│  │ Original: 0 fixed │  │ │                 │                               │
│  │ Variant1: k fixed │  │ │                 │                               │
│  │ Variant2: m fixed │  │ │                 │                               │
│  └─────────┬─────────┘  │ │                 │                               │
│            │            │ │                 │                               │
└────────────┼────────────┘ │                 │                               │
             │              │                 │                               │
             ▼              ▼                 ▼                               │
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INDEX ASSIGNMENT                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐           │
│   │  TRAIN RECORDS  │   │   DEV RECORDS   │   │  TEST RECORDS   │           │
│   │                 │   │                 │   │                 │           │
│   │  Index: -1      │   │  Index: 1000    │   │  Index: 2000    │           │
│   │  Index: -2      │   │  Index: 1001    │   │  Index: 2001    │           │
│   │  Index: -3      │   │  Index: 1002    │   │  Index: 2002    │           │
│   │  ...            │   │  ...            │   │  ...            │           │
│   │                 │   │                 │   │                 │           │
│   │  (~12,000 rows) │   │  (~850 rows)    │   │  (~830 rows)    │           │
│   │  (+ variants)   │   │  (no augment)   │   │  (no augment)   │           │
│   └────────┬────────┘   └────────┬────────┘   └────────┬────────┘           │
│            │                     │                     │                    │
└────────────┼─────────────────────┼─────────────────────┼────────────────────┘
             │                     │                     │
             └─────────────────────┼─────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │                             │
                    │  df_synthetic_better_random │
                    │          .pkl               │
                    │                             │
                    │   13,757 total rows         │
                    │                             │
                    └─────────────────────────────┘
```

---

## Output Format

### File: `df_synthetic_better_random.pkl`

A pandas DataFrame with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| `Better_Pred` | str | Original OCR prediction (before any fixes) |
| `Better_Ref` | str | Ground truth reference text |
| `Synthetic_Input` | str | **MODEL INPUT**: partially corrected text (or original pred for eval) |
| `Book` | str | Source book identifier |
| `PageNum` | int | Page number in source |
| `pair_type` | str | `"original"`, `"random_1"`, or `"random_2"` |
| `errors_remaining` | int | Number of errors still in Synthetic_Input |
| `errors_fixed` | int | Number of errors that were corrected |
| `fixed_indices` | str | JSON list of which diff indices were fixed |
| `original_idx` | int | Index in source df_tagged_better.pkl |
| `is_synthetic` | bool | True if this is an augmented variant |
| `high_priority_remaining` | int | Count of HIGH priority errors remaining |

### Index Convention

| Index Range | Meaning |
|-------------|---------|
| Negative (`-1, -2, ...`) | Training data (original + synthetic) |
| Positive (original) | Dev/test data (no augmentation, preserves original indices) |

This allows easy filtering:
```python
train_df = df[df.index < 0]
eval_df = df[df.index >= 0]
```

---

## Usage in Model Training

### Basic Training Loop

```python
# Load the synthetic data
df = pd.read_pickle('df_synthetic_better_random.pkl')

# Split by index
train_df = df[df.index < 0]
dev_df = df[df.index.isin(dev_indices)]
test_df = df[df.index.isin(test_indices)]

# For each training example:
for _, row in train_df.iterrows():
    input_text = row['Synthetic_Input']   # Model sees this
    target_text = row['Better_Ref']       # Model should produce this

    # Train the model...
```

### How the Two Training Scripts Use This File

Both training scripts in this folder (`train_byt5_unweighted.py` and `train_byt5_confidence_weighted.py`) read `Synthetic_Input` as the model input and `Better_Ref` as the target. Every row outside the dev and test indices is a training row. Both scripts also add an identity pair (`Better_Ref` → `Better_Ref`) for each original, non-synthetic training row whose input has a CER below 0.05.

The confidence-weighted script additionally weights each training pair's loss:

| Pair | Weight |
|------|--------|
| Synthetic variant | 0.5 |
| Original pair | Beta-calibrated error probability computed from the lowest token confidence of the OCR output, clipped to [0.2, 1.0] (0.5 if no confidence is available) |
| Identity pair | 1.0 |

Neither script uses `high_priority_remaining`. The column is recorded for analysis only.

---

## Key Design Decisions

### Why Weighted Sampling Instead of Filtering?

We could simply exclude low-priority errors from training. However, weighted sampling is better because:

1. **The model still sees some low-priority errors** - it needs to handle them in production
2. **Probabilistic approach is more robust** - no hard cutoffs that might miss edge cases
3. **Tunable** - weights can be adjusted based on validation performance

### Why Negative Indices for Training Data?

Using negative indices for augmented training data:

1. **Avoids index collisions** with dev/test data
2. **Easy to identify synthetic rows** (`if index < 0`)
3. **Preserves original indices** for traceability back to source data

### Why Keep Original Pairs Too?

The pipeline generates both original (no fixes) and synthetic (some fixes) pairs:

1. **Original pairs teach full correction** - from raw OCR to perfect output
2. **Synthetic pairs teach incremental correction** - from partial correction to perfect output
3. **Diversity** - model sees the same content with different error patterns

---

## Files Reference

| File | Role |
|------|------|
| `synthetic_pairs_generation.ipynb` | Main pipeline implementation |
| `df_tagged_better.pkl` | Input: aligned OCR predictions and references |
| `byt5_ungated_baseline/split_info.json` | Input: train/dev/test split indices (the path the notebook reads) |
| `df_synthetic_random.pkl` | Input to the verification cell only: an earlier version of the synthetic data, used to check that test indices line up |
| `df_synthetic_better_random.pkl` | Output: priority-weighted synthetic training data |

---

## Summary

This pipeline transforms raw (OCR prediction, reference) pairs into a training dataset that emphasizes important Turkish diacritic and vowel errors. By using weighted random sampling during synthetic data generation, we ensure the model sees more examples where it must correct the errors that actually matter for text quality, rather than spending capacity on trivial case and punctuation fixes.

The method combines a **priority classification system** with **inverse-weighted sampling**: errors we want the model to learn (HIGH priority) are kept in synthetic inputs more often, while trivial errors (LOW priority) are pre-fixed more often and so appear less in the training signal.
