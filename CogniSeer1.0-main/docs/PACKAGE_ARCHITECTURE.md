# Package Architecture: robot_brain_ros2 vs robot_brain_stack

## Overview

The CogniSeer robot brain codebase has **two complementary but distinct packages** that serve different purposes:

| Package | Location | Purpose | Use Case |
|---------|----------|---------|----------|
| **`robot_brain_ros2`** | `CogniSeer1.0-main/robot_brain_ros2/` | **Core inference engine** | Production edge deployment |
| **`robot_brain_stack`** | `robot_ws/src/robot_brain_stack/` | **Full integration stack** | Development, simulation, testing |

---

## Package Details

### 1. robot_brain_ros2 (Core Engine)

**Location**: `CogniSeer1.0-main/robot_brain_ros2/`

**Purpose**: The actual robot brain inference node — runs PyTorch model, publishes control commands.

**Components**:
- `ros2_robot_brain_node.py` — Model inference + /cmd_vel publishing
- `sensor_aggregator.py` — Fuses sensor inputs to 32D vector
- `transform_broadcaster.py` — Publishes TF2 transforms
- `safety_layer.py` — Velocity bounds and constraints
- `edge_runtime.py` — Model loading and inference execution
- `sensor_adaptor.py` — Sensor ↔ observation mapping

**Dependencies**:
- PyTorch (torch>=2.9.0)
- NumPy
- ROS2 core (rclpy, std_msgs, geometry_msgs, sensor_msgs, nav_msgs)
- TF2 (tf2_ros)
- OpenCV (cv_bridge)

**Output**: `/cmd_vel` (Twist) + `/brain/motor_command` (Float32MultiArray)

**Deployment Path**: 
```bash
# Build core package only
colcon build --packages-select robot_brain_ros2

# Launch inference engine
ros2 launch robot_brain_ros2 brain_stack.launch.py
```

**Use Cases**:
- ✅ Production robot deployment
- ✅ Embedded/edge devices
- ✅ Minimal dependencies
- ✅ Real-time inference on constrained hardware

---

### 2. robot_brain_stack (Integration Stack)

**Location**: `robot_ws/src/robot_brain_stack/`

**Purpose**: Full simulation and development environment — includes executive control, mission planning, behavior trees, Nav2 integration.

**Components**:
- `executive_context.py` — Behavior tree execution context
- `bt_executor.py` — Behavior tree orchestration
- `goal_manager.py` — Mission goal handling
- `health_monitor.py` — System health and diagnostics
- `recovery_manager.py` — Failure recovery strategies
- `safety_layer.py` — Extended safety monitoring
- `failure_injection.py` — Testing and robustness validation
- `watchdog.py` — Timeout and deadlock detection
- Launch files for Gazebo simulation, SLAM, Nav2

**Dependencies**:
- All robot_brain_ros2 dependencies PLUS:
- Nav2 (nav2_bringup, nav2_msgs)
- Gazebo (gazebo_ros)
- SLAM Toolbox (slam_toolbox)
- Action servers (action_msgs)

**Output**: Mission objectives + nav goals → /cmd_vel

**Deployment Path**:
```bash
# Build full stack
colcon build --packages-select robot_brain_stack

# Launch simulation with full stack
ros2 launch robot_brain_stack sim_world.launch.py

# Or with Nav2
ros2 launch robot_brain_stack nav2_stack.launch.py
```

**Use Cases**:
- ✅ Development and testing
- ✅ Behavior validation in simulation
- ✅ Mission planning and recovery
- ✅ Integration with navigation stack
- ✅ Learning and experimentation

---

## Relationship & Data Flow

```
Production/Edge Device:
┌─────────────────────────────────────────┐
│  robot_brain_ros2 (minimal)             │
│  ├─ Sensor Aggregator                   │
│  ├─ Brain Inference                     │
│  ├─ Safety Layer                        │
│  └─ Motor Control                       │
└─────────────────────────────────────────┘
  Lightweight, real-time, production-ready


Development/Simulation:
┌─────────────────────────────────────────┐
│  robot_brain_stack (full integration)    │
│  ├─ Executive Control (Behavior Trees)  │
│  ├─ Mission Planning & Recovery         │
│  ├─ Health Monitoring                   │
│  ├─ [Includes robot_brain_ros2 core]    │
│  ├─ Nav2 Integration                    │
│  ├─ SLAM & Localization                 │
│  └─ Gazebo Simulation                   │
└─────────────────────────────────────────┘
  Full-featured, development-oriented
```

---

## When to Use Each

### Use **robot_brain_ros2** When:

```
✅ Deploying to production hardware
✅ Running on edge/embedded devices  
✅ Minimal dependencies required
✅ Real-time inference is critical
✅ Brain-only control (no mission planning)
✅ Limited compute resources
✅ You want to integrate your own high-level planning
```

**Example**: Robotic arm with vision-based grasping, small rover with basic obstacle avoidance.

### Use **robot_brain_stack** When:

```
✅ Development and testing
✅ Simulation validation
✅ Complex mission planning needed
✅ Behavior tree orchestration
✅ Full Nav2 navigation stack
✅ Learning and experimentation
✅ You need recovery and failure handling
```

**Example**: Mobile robot with autonomous navigation, multi-task learning scenarios, research projects.

---

## Directory Structure & Canonical Paths

### Production Path (Canonical for Deployment)

```
CogniSeer1.0-main/
├── robot_brain_ros2/              ← CANONICAL RUNTIME PACKAGE
│   ├── robot_brain_ros2/
│   │   ├── ros2_robot_brain_node.py       (inference)
│   │   ├── sensor_aggregator.py           (sensor fusion)
│   │   ├── transform_broadcaster.py       (TF2)
│   │   ├── safety_layer.py                (constraints)
│   │   ├── edge_runtime.py                (model loading)
│   │   └── sensor_adaptor.py              (mapping)
│   ├── urdf/cogniseer_robot.urdf.xacro    (robot description)
│   ├── launch/brain_stack.launch.py       (production launch)
│   ├── config/                            (parameters)
│   ├── package.xml                        (minimal dependencies)
│   └── setup.py
├── deploy/                        (quantized models)
├── config/                        (defaults)
└── scripts/                       (tools)
```

### Development/Simulation Path

```
robot_ws/
└── src/
    └── robot_brain_stack/         ← DEVELOPMENT/SIMULATION PACKAGE
        ├── src/robot_brain_stack/
        │   ├── executive_context.py
        │   ├── bt_executor.py
        │   ├── goal_manager.py
        │   ├── health_monitor.py
        │   ├── recovery_manager.py
        │   └── safety_layer.py
        ├── launch/
        │   ├── brain_stack.launch.py      (full stack launch)
        │   ├── nav2_stack.launch.py       (with Nav2)
        │   ├── slam_localization.launch.py (with SLAM)
        │   └── sim_world.launch.py        (Gazebo simulation)
        ├── config/                        (mission parameters)
        ├── worlds/brain_stack_world.world (Gazebo world)
        └── package.xml                    (full dependencies)
```

---

## Build & Deployment Workflows

### Workflow 1: Production Deployment

```bash
# Only build the core inference engine
cd CogniSeer1.0-main
colcon build --packages-select robot_brain_ros2

# Deploy to robot
# Copy robot_brain_ros2 package to robot device
# Install on robot and run:
source /opt/ros/humble/setup.bash
ros2 launch robot_brain_ros2 brain_stack.launch.py
```

**Result**: Minimal footprint, fast deployment, real-time inference

---

### Workflow 2: Development & Testing

```bash
# Build full stack for development
cd robot_ws
source /opt/ros/humble/setup.bash
colcon build

# Run simulation with full features
ros2 launch robot_brain_stack sim_world.launch.py

# Or test with Nav2
ros2 launch robot_brain_stack nav2_stack.launch.py
```

**Result**: Full simulation environment, behavior testing, mission planning

---

### Workflow 3: Hybrid (Dev on desktop, Deploy to robot)

```bash
# On development machine (desktop)
cd robot_ws
colcon build  # Full stack for testing

ros2 launch robot_brain_stack sim_world.launch.py  # Develop

# On robot hardware
cd CogniSeer1.0-main
colcon build --packages-select robot_brain_ros2  # Minimal

ros2 launch robot_brain_ros2 brain_stack.launch.py  # Deploy
```

---

## Package Dependencies

### robot_brain_ros2 (Minimal)

```
Core:
  ├─ PyTorch (torch>=2.9.0)
  ├─ NumPy (numpy>=1.26.0)
  └─ JSON Schema

ROS2:
  ├─ rclpy
  ├─ std_msgs
  ├─ geometry_msgs
  ├─ sensor_msgs
  ├─ nav_msgs
  ├─ tf2_ros
  ├─ cv_bridge
  └─ python3-yaml

Total: 2-3 core packages + ROS2 (system install)
```

### robot_brain_stack (Full)

```
Everything in robot_brain_ros2 PLUS:

Navigation:
  ├─ nav2_bringup
  ├─ nav2_msgs
  ├─ slam_toolbox
  └─ amcl (localization)

Simulation:
  ├─ gazebo_ros
  ├─ gazebo_plugins
  └─ TF2 (transforms)

Control:
  ├─ ros2_control
  ├─ controller_manager
  └─ diff_drive_controller

Additional:
  ├─ action_msgs
  ├─ diagnostic_msgs
  └─ Launch (launch_ros)

Total: 10+ packages, substantial dependencies
```

---

## Clarification for Users

### Question: "Which package should I use?"

**Answer**:

1. **Is this for a production robot?** → Use `robot_brain_ros2`
   - Lighter, faster, minimal dependencies
   - Designed for embedded/edge deployment
   
2. **Is this for development or simulation?** → Use `robot_brain_stack`
   - Full features, testing tools, learning environment
   - Run in development/simulation workspace (robot_ws)

3. **Do you need both?** → Yes, they complement each other
   - Develop/test with `robot_brain_stack`
   - Deploy to production with `robot_brain_ros2`

### Question: "Why are there two packages?"

**Answer**:
- **robot_brain_ros2** = The actual inference engine (can be reused anywhere)
- **robot_brain_stack** = Integration layer for full system with planning/recovery
- They serve different purposes and have different dependency footprints
- Developers can extend stack without touching core inference

### Question: "Will robot_brain_stack depend on robot_brain_ros2?"

**Answer**: Currently they are separate. A future optimization could be:
```xml
<!-- In robot_brain_stack/package.xml -->
<depend>robot_brain_ros2</depend>  <!-- Reuse core engine -->
```
This would eliminate code duplication and ensure they stay synchronized.

---

## Canonical Deployment Recommendation

**For most users**:
- **Development**: Work in `robot_ws/` with `robot_brain_stack`
- **Production**: Deploy `robot_brain_ros2` to the robot

**For minimal installations**:
- Just use `robot_brain_ros2` everywhere
- It has all you need for inference

**For research/learning**:
- Use `robot_brain_stack` for full experimentation
- Behavior trees, recovery, mission planning included

---

## Migration Guide

### From robot_brain_stack to robot_brain_ros2

If you're using the full stack and want to simplify for production:

1. Test on `robot_brain_ros2` to ensure inference works
2. Copy sensor drivers/configuration
3. Use `robot_brain_ros2/brain_stack.launch.py` instead of full stack
4. Remove Nav2, SLAM, behavior tree dependencies

```bash
# Production deployment
ros2 launch robot_brain_ros2 brain_stack.launch.py
```

### From robot_brain_ros2 to robot_brain_stack

If you need more features for development:

1. Build `robot_brain_stack` in separate workspace
2. Copy configuration from robot_brain_ros2
3. Use robot_brain_stack with Nav2/behavior trees
4. Test mission planning and recovery

```bash
# Development with full features
ros2 launch robot_brain_stack sim_world.launch.py
```

---

## Files Affected by This Clarification

1. **`README.md`** — Add architecture overview section
2. **`CogniSeer1.0-main/robot_brain_ros2/README.md`** — Clarify as "core engine"
3. **`robot_ws/src/robot_brain_stack/README.md`** — Clarify as "development stack"
4. **`docs/DEPLOYMENT.md`** — Update with canonical paths
5. **New**: `docs/PACKAGE_ARCHITECTURE.md` — This document

---

## Summary

| Aspect | robot_brain_ros2 | robot_brain_stack |
|--------|---|---|
| **Location** | CogniSeer1.0-main/ | robot_ws/src/ |
| **Purpose** | Core inference engine | Development/simulation |
| **Size** | ~1000 LOC | ~3000 LOC |
| **Dependencies** | Minimal (torch, ROS2) | Full stack (Nav2, Gazebo, etc.) |
| **For Production** | ✅ Yes | ⏸️ Better for dev |
| **For Development** | ⏸️ Basic only | ✅ Yes |
| **Deployment Path** | **CANONICAL** | Development/Testing |
| **Launch** | `brain_stack.launch.py` | `nav2_stack.launch.py` / `sim_world.launch.py` |

---

## Next Steps

1. Read this document to understand architecture
2. Choose your path:
   - **Production**: Use `CogniSeer1.0-main/robot_brain_ros2`
   - **Development**: Use `robot_ws/src/robot_brain_stack`
3. Build and deploy accordingly
4. Refer to deployment-specific documentation

