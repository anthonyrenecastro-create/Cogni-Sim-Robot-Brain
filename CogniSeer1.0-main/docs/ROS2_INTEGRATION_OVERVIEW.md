# ROS2 Integration - Complete System Overview

**📍 Location**: `robot_brain_ros2/` package (canonical runtime path)

**Alternative**: For full-featured development with simulation and Nav2, see `robot_ws/src/robot_brain_stack/`. See [PACKAGE_ARCHITECTURE.md](PACKAGE_ARCHITECTURE.md) for comparison.

---

## Architecture Overview

The CogniSeer robot brain ROS2 integration consists of:

```
┌─────────────────────────────────────────────────────────┐
│                   Robot Sensors                         │
│  (IMU, LiDAR, Encoders, Battery, Camera, etc.)         │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│          Sensor Drivers (Native ROS2)                   │
│  (Livox, Sick, Xsens, custom wrappers, etc.)           │
└──────────────────┬───────────────────────────────────────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
    /imu/data  /odom   /scan/* (Distance)
        │          │          │
        └──────────┼──────────┘
                   ▼
        ┌──────────────────────┐
        │ Sensor Aggregator    │ ← Transforms
        │ (sensor_aggregator)  │
        └──────────┬───────────┘
                   │
                   ▼ /brain/sensors (32D obs vector)
        ┌──────────────────────┐
        │  Robot Brain Node    │ ← TF Tree
        │  (Inference Engine)  │
        └──────────┬───────────┘
                   │
                   ▼ /cmd_vel
        ┌──────────────────────┐
        │  Safety Layer        │
        │  (Bounds, Throttle)  │
        └──────────┬───────────┘
                   │
                   ▼
        Motor Controllers → Robot Hardware
```

---

## Component Overview

### 1. **Transform Broadcaster**
- **File**: `transform_broadcaster.py`
- **Purpose**: Publish robot's kinematic tree (TF2 transforms)
- **Publishes**:
  - `base_footprint` ↔ `base_link`
  - `base_link` ↔ wheels, IMU, camera, sensors
- **Launch**: `brain_stack.launch.py` (with_transform_broadcaster:=true)

### 2. **Sensor Aggregator**
- **File**: `sensor_aggregator.py`
- **Purpose**: Combine multiple sensor inputs into unified observation vector
- **Subscribes**:
  - `/imu/data` (sensor_msgs/Imu)
  - `/scan/front`, `/scan/left`, `/scan/right` (sensor_msgs/Range)
  - `/odom` (nav_msgs/Odometry)
  - `/system/battery_voltage`, `/system/cpu_temp` (std_msgs/Float32)
  - `/camera/rgb/compressed` (sensor_msgs/CompressedImage, optional)
- **Publishes**: `/brain/sensors` (std_msgs/Float32MultiArray, 32 dimensions)
- **Launch**: `brain_stack.launch.py` (with_sensor_aggregator:=true)

### 3. **Robot Brain Node**
- **File**: `ros2_robot_brain_node.py`
- **Purpose**: Inference engine (SNN model) + control loop
- **Subscribes**: `/brain/sensors` (observation vector)
- **Publishes**:
  - `/cmd_vel` (geometry_msgs/Twist) — Velocity commands
  - `/brain/motor_command` (std_msgs/Float32MultiArray) — Raw motor outputs
- **Frequency**: 20 Hz (configurable via timer)
- **Launch**: Part of `brain_stack.launch.py`

### 4. **Safety Layer**
- **Purpose**: Bound and validate control commands
- **Integrated into**: `robot_brain_node.py`
- **Checks**:
  - Max linear velocity: 0.5 m/s
  - Max angular velocity: 1.0 rad/s
  - Acceleration limits: 0.1 m/s² linear, 0.2 rad/s² angular
  - Sensor timeout: 0.5 seconds
  - Battery level warnings

---

## File Structure

```
robot_brain_ros2/
├── robot_brain_ros2/
│   ├── __init__.py
│   ├── ros2_robot_brain_node.py      # Brain inference node
│   ├── sensor_aggregator.py          # Sensor fusion node (NEW)
│   ├── transform_broadcaster.py      # TF2 broadcaster (NEW)
│   ├── edge_runtime.py               # Model loading/inference
│   ├── safety_layer.py               # Safety constraints
│   └── sensor_adaptor.py             # Sensor ↔ observation mapping
├── urdf/
│   └── cogniseer_robot.urdf.xacro    # Robot description (NEW)
├── launch/
│   └── brain_stack.launch.py          # Main launch file (UPDATED)
├── config/
│   ├── defaults.json                  # Brain parameters
│   └── brain_params.yaml              # ROS2 parameters (NEW)
├── package.xml                        # ROS2 package manifest (UPDATED)
├── setup.py                           # Python entry points (UPDATED)
└── README.md
```

---

## Quick Start

### 1. Install ROS2 (if not already done)

```bash
# See docs/ROS2_SETUP.md for complete instructions
source /opt/ros/humble/setup.bash
```

### 2. Build the Package

```bash
cd ~/robot_ws
colcon build --packages-select robot_brain_ros2
source install/setup.bash
```

### 3. Launch the Brain Stack

```bash
# Basic launch (requires sensors publishing to standard topics)
ros2 launch robot_brain_ros2 brain_stack.launch.py

# With visualization
ros2 launch robot_brain_ros2 brain_stack.launch.py with_rviz:=true

# Custom configuration
ros2 launch robot_brain_ros2 brain_stack.launch.py \
    model_path:=/path/to/custom_model.pt \
    namespace:=my_robot
```

### 4. Verify Integration

```bash
# Check running nodes
ros2 node list

# Verify topic publishing
ros2 topic hz /brain/sensors
ros2 topic echo /cmd_vel | head -10

# Check transforms
ros2 run tf2_tools view_frames.py && evince frames.pdf

# Run compatibility checker
python verify_topic_compatibility.py --config config/brain_params.yaml
```

---

## Sensor Integration Checklist

### Step 1: Connect Hardware
- [ ] IMU connected and powered
- [ ] Distance sensors connected
- [ ] Wheel encoders connected/configured
- [ ] Motor controllers connected
- [ ] (Optional) Camera connected
- [ ] (Optional) Battery voltage tap connected

### Step 2: Install Sensor Drivers
- [ ] IMU driver installed (`ros-humble-xsens-mti-driver`, etc.)
- [ ] Distance sensor driver installed
- [ ] Odometry publisher configured
- [ ] Motor controller driver ready

### Step 3: Launch Sensor Drivers
```bash
# In separate terminal windows
ros2 launch xsens_mti_driver xsens_mti_node.launch.py
ros2 launch sick_scan2 sick_tim781s.launch.py
ros2 run my_odom_node odometry_publisher
```

### Step 4: Verify Topics
```bash
# Check all topics are publishing
ros2 topic list | grep -E 'imu|scan|odom|battery'

# Check rates
ros2 topic hz /imu/data
ros2 topic hz /odom
ros2 topic hz /scan/front
```

### Step 5: Launch Brain Stack
```bash
ros2 launch robot_brain_ros2 brain_stack.launch.py
```

### Step 6: Monitor Output
```bash
# Watch brain sensor input
ros2 topic echo /brain/sensors

# Watch velocity commands
ros2 topic echo /cmd_vel

# Monitor logs
ros2 node info /robot_brain_node
```

---

## Documentation Structure

| Document | Purpose | Audience |
|----------|---------|----------|
| **ROS2_SETUP.md** | ROS2 installation and environment setup | All users |
| **ROS2_SENSOR_TOPICS_AND_CONTROLLERS.md** | Detailed sensor topic specs and examples | System integrators |
| **ROS2_DRIVER_INTEGRATION.md** | Sensor driver patterns and examples | Hardware developers |
| **ROS2_DEPENDENCY_RESOLUTION.md** | How ROS2 dependency was resolved | Developers |
| **cogniseer_robot.urdf.xacro** | Robot kinematic description | Simulation/visualization |

---

## Key New Features

### 1. **Unified Sensor Aggregation**
- Single 32D observation vector
- Configurable per-sensor enable/disable
- Handles missing sensors gracefully
- Health check service: `/brain/sensor_health`

### 2. **Transform Broadcasting**
- Complete robot kinematic tree
- Calibration parameters (offsets for sensors)
- Compatible with RViz and Nav2

### 3. **Robot Description (URDF)**
- 3D model of robot structure
- Sensor mounting points
- Wheel and joint definitions
- Transmission definitions for control

### 4. **Topic Compatibility Verification**
- Automated compatibility checking
- Frequency monitoring
- TF validation
- Time synchronization checking

### 5. **Comprehensive Driver Integration Guide**
- Examples for 10+ common sensors
- Pattern matching for new sensors
- Troubleshooting guide

---

## Configuration

### Default Configuration
All parameters can be set via ROS2 launch file:

```python
# In brain_stack.launch.py
parameters=[
    {"model_path": "path/to/model.pt"},
    {"input_dim": 32},
    {"cmd_scale_linear": 0.5},
    {"cmd_scale_angular": 1.0},
    # Safety parameters
    {"safety.motor_min": -1.0},
    {"safety.motor_max": 1.0},
    {"safety.max_linear_x": 0.5},
    {"safety.max_angular_z": 1.0},
    {"safety.max_linear_delta": 0.10},
    {"safety.max_angular_delta": 0.20},
    {"safety.sensor_timeout_sec": 0.50},
    {"safety.stop_on_stale_sensor": True},
]
```

### Sensor Configuration (sensor_aggregator)

```python
parameters=[
    {"observation_dim": 32},
    {"imu_enabled": True},
    {"distance_enabled": True},
    {"encoder_enabled": True},
    {"battery_enabled": True},
    {"camera_enabled": False},  # Disabled by default
]
```

### Transform Offsets (transform_broadcaster)

```python
parameters=[
    {"imu.x_offset": 0.0},
    {"imu.y_offset": 0.0},
    {"imu.z_offset": 0.08},
    {"camera.x_offset": 0.2},
    # ...
]
```

---

## Troubleshooting

### Topic Not Publishing

```bash
# Check if driver is running
ros2 node list | grep -i sensor

# Check topic existence
ros2 topic list | grep imu

# Monitor raw output
ros2 topic echo /imu/data
```

**Solution**: Start the sensor driver node manually.

### Frequency Too Low

```bash
# Check actual frequency
ros2 topic hz /imu/data
# Expected: ≥ 50 Hz for IMU

# Check for drops
ros2 topic hz /imu/data --window 100  # Last 100 messages
```

**Solution**: Check USB cable quality, reduce other load, enable driver-side buffering.

### TF Transform Missing

```bash
# View all transforms
ros2 run tf2_tools view_frames.py

# Find specific transform
ros2 run tf2_tools view_frames.py | grep -A2 "imu_link"
```

**Solution**: Check `transform_broadcaster.py` is running, verify frame names match.

### Safety Layer Shutting Down Motors

```bash
# Check safety status
ros2 topic echo /robot_brain_node/debug_info | grep safety

# Check sensor timeout
ros2 topic hz /brain/sensors  # Should be ≥ 50 Hz
```

**Solution**: Ensure sensor_aggregator is publishing at expected rate.

---

## Testing & Validation

### Unit Tests

```bash
cd tests/
python -m pytest test_integration_pipeline_export_smoke.py -v
```

### Integration Tests

```bash
# Verify all components together
python verify_topic_compatibility.py --config config/brain_params.yaml

# Simulate sensor inputs and monitor outputs
ros2 launch robot_brain_ros2 brain_stack.launch.py
```

### Hardware Validation

```bash
# On real robot, check actual motor movement
ros2 launch robot_brain_ros2 brain_stack.launch.py

# Monitor safety constraints
ros2 topic echo /cmd_vel
```

---

## Performance Metrics

| Component | Update Rate | Latency | CPU Usage |
|-----------|---|---|---|
| Sensor Aggregator | 50 Hz | <5 ms | <5% (single core) |
| Brain Inference | 20 Hz | ~50 ms (depends on GPU) | 10-30% |
| Transform Broadcaster | 20 Hz | <1 ms | <1% |
| Safety Layer | 20 Hz | <1 ms | <1% |
| **Total** | **20 Hz control** | **~60-70 ms** | **15-40%** |

---

## Next Steps

1. **Read**: [ROS2_SETUP.md](ROS2_SETUP.md) to install ROS2
2. **Integrate**: Sensor drivers using [ROS2_DRIVER_INTEGRATION.md](ROS2_DRIVER_INTEGRATION.md)
3. **Verify**: Topics using `verify_topic_compatibility.py`
4. **Launch**: Brain stack and test on real hardware
5. **Monitor**: `/cmd_vel` output and safety constraints
6. **Tune**: Safety parameters for your specific robot

---

## References

- **ROS2 Documentation**: https://docs.ros.org/
- **ROS2 Message Types**: https://github.com/ros2/common_interfaces
- **TF2 Documentation**: https://docs.ros.org/en/humble/Concepts/Intermediate/Tf2/Main.html
- **URDF Tutorial**: https://wiki.ros.org/urdf/Tutorials
- **ros2_control**: https://control.ros.org/

---

**Status**: ✅ **ROS2 integration complete with drivers, sensors, transforms, URDF, and verification**

