# Dataset Requirements for Robust Robot Policy

The current sample dataset (`datasets/raw/robot_telemetry.jsonl`) contains only **50 samples** and is insufficient to produce a robust robot control policy. This document outlines minimum dataset requirements and provides strategies for data collection and augmentation.

## Minimum Dataset Requirements

### Sample Size Guidelines

| Policy Type | Minimum Samples | Recommended | Optimal |
|---|---|---|---|
| **Simple 1-DOF control** | 500 | 2,000 | 5,000+ |
| **Multi-DOF (≥6 axes)** | 5,000 | 20,000 | 50,000+ |
| **Field-enhanced SNN** (this project) | 10,000 | 50,000 | 100,000+ |

**Why these numbers?**
- The spiking neural network architecture in this project uses field-enhanced layers with chaos reduction, requiring more diverse training data to stabilize learning.
- With 8 motor outputs and 32 observation dimensions, the model has ~50K-100K parameters.
- A rule of thumb: aim for 10-100× the number of model parameters in training samples.
- The 50 samples provide ~0.05-0.1× coverage—far below adequacy.

### Data Distribution Requirements

1. **Temporal Diversity**: Samples should span different robot states, velocities, and accelerations
2. **Coverage**: Multiple trajectories through state space (avoid overfitting to single trajectory)
3. **Environmental Variation**: Different task contexts, obstacles, or operating modes (if applicable)
4. **Edge Cases**: Include recovery scenarios, saturation limits, and boundary behaviors

### Quality Metrics

- **Observation range**: Values should span [-1.5, 1.5] (not clustered at single point)
- **Motor target range**: Should show variation across safe operating range
- **Temporal coherence**: Consecutive samples (20ms intervals in current data) should show smooth state transitions
- **Stale samples**: Remove duplicate or near-identical consecutive records

## Current State Analysis

The sample dataset shows:
- **50 samples** collected over 1.28 seconds (20ms intervals)
- **Observation dimension**: 32 floats
- **Motor output dimension**: 8 floats
- **Coverage**: Single continuous trajectory (no diversity)

This represents a **proof-of-concept snapshot**, not production training data.

## Data Collection Strategies

### 1. Increase Sampling Duration (Minimum Viable)
**Effort**: Low | **Time**: ~5 min | **Result**: 500-1K samples

Extend telemetry logging during robot operation:
```python
# Collect 5 minutes of continuous telemetry
duration_seconds = 300
sample_rate_hz = 50  # 50 Hz = 20ms intervals
expected_samples = duration_seconds * sample_rate_hz  # 15,000 samples
```

### 2. Multiple Distinct Trajectories (Recommended)
**Effort**: Medium | **Time**: ~1-2 hours | **Result**: 20K-50K samples

- Run the same task multiple times with variations:
  - Different starting positions
  - Different goal positions
  - Varying execution speeds
  - Different environmental conditions

```bash
# Collect from 10 independent task runs
for i in {1..10}; do
  # Run task, log telemetry to: datasets/raw/trajectory_$i.jsonl
  roslaunch robot_brain_stack sim_world.launch.py
done

# Merge trajectories
python scripts/merge_telemetry.py \
  --input datasets/raw/trajectory_*.jsonl \
  --output datasets/raw/robot_telemetry.jsonl
```

### 3. Simulation-Based Generation (Advanced)
**Effort**: High | **Time**: ~4-8 hours | **Result**: 50K-500K samples

Generate synthetic telemetry from physics simulation:
- Use Gazebo or PyBullet to simulate robot dynamics
- Record observation-action pairs during exploration or policy rollouts
- Create deterministic but varied scenarios

See [scripts/generate_synthetic_data.py](../scripts/generate_synthetic_data.py) for a template.

## Data Augmentation Strategies

### 1. Temporal Smoothing & Interpolation
**Multiplier**: 2-3× increase

Create intermediate samples between existing records:
```python
# Linear interpolation between frame N and N+1
interpolated_obs = alpha * obs[i] + (1 - alpha) * obs[i+1]
# Add small Gaussian noise to preserve diversity
augmented_obs = interpolated_obs + 0.01 * np.random.randn(32)
```

### 2. Noise Injection
**Multiplier**: 1.5-2×

Add realistic sensor noise to existing samples:
```python
# Gaussian observation noise (sensor uncertainty)
noise_std = 0.02  # 2% of typical signal magnitude
augmented_obs = obs + noise_std * np.random.randn(32)

# Small motor command perturbations (actuator jitter)
motor_noise = 0.005 * np.random.randn(8)
augmented_target = target + motor_noise
```

### 3. Trajectory Scaling & Shifting
**Multiplier**: 2-4×

Create variations of existing trajectories:
```python
# Time-stretch: slow down or speed up trajectory
time_scale = 1.5  # 50% slower
new_sample = sample[::int(time_scale)]

# State-shift: adjust baseline observation level
baseline_shift = 0.1 * np.random.randn(32)
augmented_obs = obs + baseline_shift
```

**Warning**: Augmentation can only stretch existing data so far—it cannot replace genuine diversity.

## Implementation Steps

### Phase 1: Quick Validation (Today)
1. ✅ Recognize current 50-sample dataset is insufficient
2. Use data augmentation script (see below) to reach ~5K samples
3. Train model and measure performance
4. Document training loss plateau/overfitting

### Phase 2: Real Data Collection (This Week)
1. Collect 5-10 minutes of continuous robot telemetry → 15K-30K samples
2. Collect 3-5 independent task trajectories → 50K+ samples
3. Perform train/eval split
4. Retrain model with larger dataset

### Phase 3: Long-term (Ongoing)
1. Implement continuous logging in production
2. Archive high-quality trajectories in dataset repository
3. Version datasets with metadata (conditions, performance metrics)
4. Maintain train/eval splits and test distribution shift

## Tools Provided

### Augmentation Utility
```bash
python scripts/augment_telemetry.py \
  --input datasets/raw/robot_telemetry.jsonl \
  --output datasets/raw/robot_telemetry_augmented.jsonl \
  --multiplier 10 \
  --noise-std 0.02
```

See [scripts/augment_telemetry.py](../scripts/augment_telemetry.py) for full documentation.

### Merge Utility
```bash
python scripts/merge_telemetry.py \
  --inputs datasets/raw/trajectory_*.jsonl \
  --output datasets/raw/robot_telemetry_merged.jsonl
```

## Expected Impact on Training

| Dataset Size | Training Outcome | Policy Robustness |
|---|---|---|
| 50 samples | Severe overfitting, poor generalization | ❌ Fragile |
| 500 samples | Overfitting, limited generalization | ⚠️ Poor |
| 5K samples | Moderate overfitting, some generalization | ⚠️ Fair |
| 50K samples | Reasonable fit, good generalization | ✅ Good |
| 100K+ samples | Near-optimal fit with regularization | ✅ Excellent |

## Testing & Validation

After expanding the dataset:

1. **Training Loss Monitoring**
   ```bash
   make train EPOCHS=100  # Monitor loss curves
   ```
   Expect: Train loss decreases smoothly; eval loss plateaus (not diverges)

2. **Policy Smoke Tests**
   ```bash
   make smoke SMOKE_STEPS=100  # Evaluate on test rollouts
   ```
   Expect: Smoother trajectories, fewer instabilities

3. **Distribution Shift Detection**
   ```bash
   python scripts/check_data_distribution.py
   ```
   Verify train/eval distributions are similar

## References

- **Imitation Learning**: [Learning from Demonstrations (Schaal, 1999)](https://ieeexplore.ieee.org/document/784150)
- **Dataset Size Guidelines**: [Scaling Laws for Robotics (OpenAI, 2023)](https://openai.com/research/scaling-transfer)
- **SNN Training**: [Training Spiking Neural Networks (Neftci et al., 2019)](https://www.frontiersin.org/articles/10.3389/fnins.2019.00663)

---

**Status**: ⚠️ **Current dataset is insufficient for robust policy training**  
**Action Required**: Expand dataset to ≥10K samples before production deployment
