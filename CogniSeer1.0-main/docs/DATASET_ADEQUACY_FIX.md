# Dataset Adequacy Issue - Resolution Summary

## Problem Statement

The included raw telemetry dataset (`datasets/raw/robot_telemetry.jsonl`) contains **only 320 samples** collected over ~6.4 seconds. This is insufficient to produce a robust robot control policy using the field-enhanced spiking neural network (SNN) architecture in this project.

**Impact**: Models trained on this dataset will severely overfit, producing brittle control policies that fail on unseen states.

---

## Root Cause Analysis

1. **Dataset Size**: 320 samples << 10,000 recommended minimum for SNN
2. **Limited Diversity**: Single ~6.4-second trajectory (no environmental variation)
3. **No Guidance**: Users unaware of dataset requirements
4. **No Tools**: No utilities provided for data collection or augmentation
5. **Silent Failure**: No validation warnings during training

---

## Solution Implemented

### 1. **Comprehensive Dataset Requirements Documentation**

**Location**: [docs/DATASET_REQUIREMENTS.md](docs/DATASET_REQUIREMENTS.md)

**Contents**:
- Sample size guidelines table (500 → 100K+ samples)
- Data distribution requirements (temporal diversity, coverage, edge cases)
- Quality metrics (observation range, motor target range, etc.)
- Current dataset analysis (320 samples from single trajectory)
- **Three data collection strategies**:
  - Quick (5 min continuous logging → 15K samples)
  - Recommended (multiple trajectories → 50K+ samples)
  - Advanced (simulation-based generation → 500K+ samples)
- **Four augmentation techniques** with implementation examples
- 3-phase implementation roadmap with timelines
- Testing & validation guidelines
- Expected training impact across dataset sizes

---

### 2. **Data Augmentation Utility**

**Location**: [scripts/augment_telemetry.py](scripts/augment_telemetry.py)

**Strategies**:
- **Temporal Interpolation**: Creates intermediate samples between consecutive records
- **Noise Injection**: Adds realistic Gaussian sensor/actuator noise
- **Trajectory Scaling**: Simulates variations in operating conditions via baseline shifts

**Example**:
```bash
# Expand 320 samples to ~1600 via augmentation
python scripts/augment_telemetry.py \
  --input datasets/raw/robot_telemetry.jsonl \
  --output datasets/raw/robot_telemetry_augmented.jsonl \
  --multiplier 5 \
  --noise-std 0.02 \
  --seed 42
```

**Result**: 320 → 1088 samples (useful for quick validation)

**Warning**: Augmentation should supplement, not replace, real data collection.

---

### 3. **Telemetry Merge Utility**

**Location**: [scripts/merge_telemetry.py](scripts/merge_telemetry.py)

**Features**:
- Merge multiple JSONL files (e.g., from separate robot runs)
- Automatic deduplication
- Sorting by timestamp or other fields
- Record ID remapping for uniqueness

**Example**:
```bash
# Combine 10 trajectory runs into one dataset
python scripts/merge_telemetry.py \
  --inputs datasets/raw/trajectory_*.jsonl \
  --output datasets/raw/robot_telemetry_merged.jsonl \
  --sort-by timestamp_ms
```

---

### 4. **Data Distribution Analyzer**

**Location**: [scripts/check_data_distribution.py](scripts/check_data_distribution.py)

**Capabilities**:
- Compute observation & motor target statistics (mean, std, min, max)
- Check for low-variance dimensions (indicating limited diversity)
- Validate dataset adequacy with warnings
- Generate detailed analysis reports

**Example Output**:
```
Dataset Size: 320 samples (32 observations, 8 motor outputs)
Status: ❌ INSUFFICIENT

⚠️  CRITICAL: Dataset has 320 samples, but minimum viable is 1000.
   Training will likely overfit severely.
```

---

### 5. **Data Pipeline Validation**

**Location**: [data_pipeline.py](data_pipeline.py)

**Changes**:
- Added `_check_dataset_adequacy()` function
- Integrated adequacy checks into dataset building workflow
- Issues warnings when dataset is insufficient for SNN training
- Provides guidance pointing to `docs/DATASET_REQUIREMENTS.md`

**Example**: Training code now prints:
```
UserWarning: ⚠️  CRITICAL: Dataset has 320 samples, but minimum viable is 1000.
Training will likely overfit severely. See docs/DATASET_REQUIREMENTS.md.
```

---

## Quick Start: Expand the Dataset

### Phase 1: Quick Validation (Today)
```bash
# Augment existing 320 samples to ~1600 for testing
python scripts/augment_telemetry.py \
  --input datasets/raw/robot_telemetry.jsonl \
  --output datasets/raw/robot_telemetry_augmented.jsonl \
  --multiplier 5

# Check adequacy
python scripts/check_data_distribution.py \
  --input datasets/raw/robot_telemetry_augmented.jsonl

# Train and evaluate (will warn about insufficient data)
python train_brain.py
```

### Phase 2: Real Data Collection (This Week)
Collect robot telemetry:
```bash
# Option A: Extend continuous logging (5+ minutes)
# - Yields 15K-30K samples
# - Lowest effort, captures single trajectory

# Option B: Multiple task runs (5-10 runs)
# - Yields 50K+ diverse samples
# - Recommended for robust policy
# - Merge with: python scripts/merge_telemetry.py

# Option C: Simulation-based generation
# - Yields 100K+ samples
# - Requires environment simulation
# - See docs/DATASET_REQUIREMENTS.md for templates
```

### Phase 3: Long-term (Ongoing)
- Implement continuous telemetry logging in production
- Archive high-quality trajectories with metadata
- Version datasets and track training performance
- Maintain train/eval distribution balance

---

## Expected Impact

| Dataset Size | Training Outcome | Policy Quality |
|---|---|---|
| **50-320 samples** (current) | Severe overfitting | ❌ Very Poor |
| **500-1K samples** (augmented) | Heavy overfitting | ❌ Poor |
| **5K samples** | Moderate overfitting | ⚠️ Fair |
| **50K samples** | Good generalization | ✅ Good |
| **100K+ samples** | Excellent fit | ✅ Excellent |

---

## Tools at a Glance

| Tool | Purpose | Input | Output | Time |
|---|---|---|---|---|
| `augment_telemetry.py` | Expand dataset via synthetic methods | JSONL | JSONL | 1 min |
| `merge_telemetry.py` | Combine multiple datasets | JSONL(s) | JSONL | 1 min |
| `check_data_distribution.py` | Analyze dataset adequacy | JSONL | Report | 30 sec |
| `train_brain.py` | Train model (with validation warnings) | JSONL | Checkpoint | Hours |

---

## Testing & Validation

### 1. Verify Scripts Work
```bash
# All three scripts should execute successfully
python scripts/augment_telemetry.py --help
python scripts/merge_telemetry.py --help
python scripts/check_data_distribution.py --help
```

### 2. Check Dataset Adequacy
```bash
python scripts/check_data_distribution.py \
  --input datasets/raw/robot_telemetry.jsonl
```
Expected: Report showing 320 samples, ❌ INSUFFICIENT status, critical warning.

### 3. Validate Pipeline Warnings
```bash
python train_brain.py  # or cogniseer-train command
```
Expected: UserWarning about dataset size, recommendation to expand.

---

## Documentation

- **For Users**: Start with [docs/DATASET_REQUIREMENTS.md](docs/DATASET_REQUIREMENTS.md)
- **For Developers**: See inline docstrings in utility scripts
- **For Integration**: Data pipeline automatically validates during training

---

## Success Criteria

✅ **Implemented**:
1. Clear documentation of dataset requirements
2. Tools for data collection and augmentation
3. Data distribution analysis and validation
4. Training pipeline warnings about inadequate datasets
5. Guidance for all three phases (validation → collection → production)

✅ **Tested**:
1. `augment_telemetry.py`: 320 → 1088 samples ✓
2. `merge_telemetry.py`: Correctly merges and deduplicates ✓
3. `check_data_distribution.py`: Identifies 320 samples as INSUFFICIENT ✓
4. `data_pipeline.py`: Issues critical warnings during training ✓

---

## Next Steps for Users

1. **Read**: [docs/DATASET_REQUIREMENTS.md](docs/DATASET_REQUIREMENTS.md)
2. **Validate**: `python scripts/check_data_distribution.py --input datasets/raw/robot_telemetry.jsonl`
3. **Choose**: One of three collection strategies based on available time/resources
4. **Collect**: Real robot telemetry (minimum 15K, recommended 50K+ samples)
5. **Merge**: Use `scripts/merge_telemetry.py` to combine multiple runs
6. **Train**: Retrain model with expanded dataset

---

**Status**: ✅ **Issue addressed with comprehensive solution**  
**Key Takeaway**: Dataset size matters. The 320-sample dataset is a proof-of-concept. Expand to ≥10K samples for robust training.
