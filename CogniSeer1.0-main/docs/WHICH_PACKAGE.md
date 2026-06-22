# Which Package Should I Use?

A quick decision guide to choosing between `robot_brain_ros2` and `robot_brain_stack`.

---

## Quick Decision Tree

```
START
  │
  ├─→ "I'm deploying to a production robot"
  │     └─→ USE: robot_brain_ros2 (CogniSeer1.0-main/)
  │           Why: Minimal footprint, optimized for inference
  │
  ├─→ "I'm developing features or experimenting"
  │     └─→ USE: robot_brain_stack (robot_ws/src/)
  │           Why: Full features, simulation, behavior trees
  │
  ├─→ "I'm just training/exporting models"
  │     └─→ USE: CogniSeer1.0-main/ (training only, no ROS2)
  │           Why: No ROS2 needed, minimal setup
  │
  └─→ "I'm doing research with mission planning"
        └─→ USE: robot_brain_stack (robot_ws/src/)
              Why: Behavior trees, recovery strategies, Nav2
```

---

## One-Sentence Summary

| Your Need | Package | Location |
|-----------|---------|----------|
| **Train/export models** | None (pure Python) | CogniSeer1.0-main/ |
| **Deploy inference on robot** | **robot_brain_ros2** ⭐ | CogniSeer1.0-main/robot_brain_ros2/ |
| **Develop with simulation** | **robot_brain_stack** | robot_ws/src/robot_brain_stack/ |

---

## Scenario-Based Guide

### Scenario 1: "I just got a robot and want to run the brain"

```
✅ Use: robot_brain_ros2 (CogniSeer1.0-main/robot_brain_ros2/)

Commands:
  cd CogniSeer1.0-main
  colcon build --packages-select robot_brain_ros2
  ros2 launch robot_brain_ros2 brain_stack.launch.py
```

**Why**: It's the minimal, production-ready package. Just load your sensors and go.

---

### Scenario 2: "I need to test mission planning and recovery"

```
✅ Use: robot_brain_stack (robot_ws/src/robot_brain_stack/)

Commands:
  cd robot_ws
  colcon build
  ros2 launch robot_brain_stack sim_world.launch.py
```

**Why**: Includes behavior trees, recovery strategies, and Gazebo simulation.

---

### Scenario 3: "I'm developing new inference features"

```
✅ Use: robot_brain_ros2 (CogniSeer1.0-main/robot_brain_ros2/)

Commands:
  # Edit the code
  vim robot_brain_ros2/robot_brain_ros2/ros2_robot_brain_node.py
  
  # Test locally
  colcon build --packages-select robot_brain_ros2
  ros2 launch robot_brain_ros2 brain_stack.launch.py
```

**Why**: Faster iteration, smaller package, direct control over inference.

---

### Scenario 4: "I need integration with Nav2 navigation"

```
✅ Use: robot_brain_stack (robot_ws/src/robot_brain_stack/)

Commands:
  cd robot_ws
  colcon build
  ros2 launch robot_brain_stack nav2_stack.launch.py
```

**Why**: robot_brain_stack includes Nav2 bringup and SLAM toolbox.

---

### Scenario 5: "I'm training models but not deploying yet"

```
✅ Use: CogniSeer1.0-main/ (Python training, NO ROS2)

Commands:
  cd CogniSeer1.0-main
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.lock && pip install -e .
  python train_brain.py
```

**Why**: ROS2 isn't needed for training. Keep it simple.

---

## Feature Comparison Matrix

| Feature | robot_brain_ros2 | robot_brain_stack |
|---------|---|---|
| **Inference Engine** | ✅ | ✅ |
| **Sensor Aggregation** | ✅ | ✅ |
| **Safety Layer** | ✅ | ✅ |
| **Motor Control** | ✅ | ✅ |
| **TF2 Transforms** | ✅ | ✅ |
| **URDF Support** | ✅ | ✅ |
| **Behavior Trees** | ❌ | ✅ |
| **Mission Planning** | ❌ | ✅ |
| **Recovery Manager** | ❌ | ✅ |
| **Nav2 Integration** | ❌ | ✅ |
| **Gazebo Simulation** | ❌ | ✅ |
| **SLAM Toolbox** | ❌ | ✅ |
| **Production-Ready** | ✅ | ⏸️ Dev-focused |
| **Minimal Dependencies** | ✅ | ❌ (Full stack) |
| **Fast Iteration** | ✅ | ⏸️ Larger builds |

---

## Size Comparison

```
robot_brain_ros2:
  ├─ Package size: ~500 KB
  ├─ Build time: ~2-3 minutes
  ├─ Dependencies: 5-10 packages
  └─ Runtime memory: ~100-200 MB

robot_brain_stack:
  ├─ Package size: ~2-3 MB
  ├─ Build time: ~10-15 minutes
  ├─ Dependencies: 30+ packages
  └─ Runtime memory: ~500+ MB
```

---

## Installation Paths

### Path 1: Production Deployment (Recommended)

```
1. Install ROS2 Humble/Iron on robot
2. Clone CogniSeer1.0-main repository
3. Build robot_brain_ros2 only:
     colcon build --packages-select robot_brain_ros2
4. Launch:
     ros2 launch robot_brain_ros2 brain_stack.launch.py
```

**Expected resources**: 2GB disk, 200MB RAM

---

### Path 2: Development & Simulation

```
1. Install ROS2 Humble on development machine
2. Set up robot_ws workspace:
     cd robot_ws
     colcon build
3. Launch simulation:
     ros2 launch robot_brain_stack sim_world.launch.py
4. Develop and test behavior/recovery
5. When ready: Deploy robot_brain_ros2 to actual robot
```

**Expected resources**: 10GB disk (Gazebo), 4GB RAM

---

### Path 3: Training Only (No ROS2 Needed)

```
1. No ROS2 installation needed
2. Clone CogniSeer1.0-main
3. Set up Python environment:
     python -m venv .venv
     source .venv/bin/activate
     pip install -r requirements.lock && pip install -e .
4. Train:
     python train_brain.py
5. Export:
     python export_brain.py
```

**Expected resources**: 2GB disk, 4GB RAM, GPU optional

---

## Common Questions

### Q: Can I use robot_brain_stack on a robot?

**A**: Technically yes, but not recommended. It has extra dependencies (Gazebo, Nav2, SLAM) that you probably don't need on the robot. Use `robot_brain_ros2` instead.

### Q: Can I use robot_brain_ros2 for development?

**A**: Yes, absolutely. It's just smaller and faster. If you need behavior trees or Nav2, upgrade to `robot_brain_stack`.

### Q: Do I need to use both?

**A**: Typically:
- Use `robot_brain_stack` for development
- Use `robot_brain_ros2` for deployment

But you can stick with just one if you prefer.

### Q: Will robot_brain_stack work offline (no internet)?

**A**: Once all dependencies are installed, yes. But initial setup requires downloading Gazebo and dependencies (large).

### Q: Is robot_brain_ros2 suitable for real-time control?

**A**: Yes. It's optimized for inference latency (~50-100ms) and control frequency (20 Hz). It's production-ready.

### Q: Can I extend robot_brain_ros2 with my own code?

**A**: Yes. It's designed for this. Add your own nodes to the launch file or create a wrapper launch file in your own package.

---

## Migration Path

### From robot_brain_stack → robot_brain_ros2

If you're using the full stack and want to simplify for production:

```bash
# Test with minimal stack
colcon build --packages-select robot_brain_ros2
ros2 launch robot_brain_ros2 brain_stack.launch.py

# When ready, deploy to robot
# (robot_brain_ros2 is already lightweight)
```

### From robot_brain_ros2 → robot_brain_stack

If you need more features later:

```bash
# Build full stack
cd robot_ws
colcon build

# Launch with features
ros2 launch robot_brain_stack nav2_stack.launch.py
```

---

## Decision Checklist

Before choosing, ask yourself:

```
Production Deployment?
  [ ] Yes → robot_brain_ros2 ✅
  [ ] No

Simulation/Development?
  [ ] Yes → robot_brain_stack ✅
  [ ] No

Mission Planning/Behavior Trees?
  [ ] Yes → robot_brain_stack ✅
  [ ] No

Minimal Dependencies?
  [ ] Yes → robot_brain_ros2 ✅
  [ ] No

Fast Iteration on Inference?
  [ ] Yes → robot_brain_ros2 ✅
  [ ] No

Full Feature Set?
  [ ] Yes → robot_brain_stack ✅
  [ ] No
```

---

## See Also

- [PACKAGE_ARCHITECTURE.md](PACKAGE_ARCHITECTURE.md) — Detailed architecture comparison
- [ROS2_INTEGRATION_OVERVIEW.md](ROS2_INTEGRATION_OVERVIEW.md) — Complete system overview
- [DEPLOYMENT.md](../DEPLOYMENT.md) — Deployment steps for robot_brain_ros2
- [robot_ws/src/robot_brain_stack/README.md](../../robot_ws/src/robot_brain_stack/README.md) — Development stack details

