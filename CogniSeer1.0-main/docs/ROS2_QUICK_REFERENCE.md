# ROS2 Integration Quick Reference

## What Was Added

This update adds **complete ROS2 driver, sensor, transform, URDF, and controller support** to the CogniSeer robot brain.

### New Components (7 Files)

| File | Purpose | Lines |
|------|---------|-------|
| `sensor_aggregator.py` | Fuses 6+ sensor inputs → 32D vector | 262 |
| `transform_broadcaster.py` | Publishes robot TF2 kinematic tree | 167 |
| `cogniseer_robot.urdf.xacro` | Robot 3D description for RViz/Gazebo | 182 |
| `ROS2_INTEGRATION_OVERVIEW.md` | System architecture & quick start | 400+ |
| `ROS2_SENSOR_TOPICS_AND_CONTROLLERS.md` | Topic specs & controller examples | 380+ |
| `ROS2_DRIVER_INTEGRATION.md` | Sensor driver patterns (10+ examples) | 350+ |
| `verify_topic_compatibility.py` | Automated sensor compatibility checker | 320+ |

### Updated Files (3)

1. `package.xml` — Added dependencies (tf2, cv_bridge, yaml)
2. `setup.py` — Added executable entry points
3. `launch/brain_stack.launch.py` — Integrated all components

---

## 5-Minute Startup

### 1. Source ROS2
```bash
source /opt/ros/humble/setup.bash
```

### 2. Build Package
```bash
cd ~/robot_ws
colcon build --packages-select robot_brain_ros2
source install/setup.bash
```

### 3. Launch Everything
```bash
ros2 launch robot_brain_ros2 brain_stack.launch.py
```

### 4. Verify Topics Publishing
```bash
ros2 topic list | grep -E 'imu|scan|odom|brain'
ros2 topic hz /brain/sensors
```

---

## The 32D Observation Vector

Brain input is always **32 dimensions**:

```
[0:3]   = IMU accelerometer (accel_x, accel_y, accel_z)
[3:6]   = IMU gyroscope (rot_x, rot_y, rot_z)
[6:8]   = Distance sensors (front, left/right averaged)
[9:10]  = Odometry position (x, y)
[11]    = Yaw angle
[12:13] = Velocities (linear_x, angular_z)
[14]    = Battery voltage (normalized 10.5-12.6V → 0-1)
[15]    = CPU temperature (normalized 0-100°C → 0-1)
[16:31] = Camera (4×4 grayscale, optional)
```

**Published as**: `geometry_msgs/Float32MultiArray` to `/brain/sensors` at 50 Hz

---

## Sensor Topics (Standard)

| Sensor | Topic | Type | Rate |
|--------|-------|------|------|
| IMU | `/imu/data` | `sensor_msgs/Imu` | 50 Hz |
| Distance (Front) | `/scan/front` | `sensor_msgs/Range` | 10-50 Hz |
| Distance (Left) | `/scan/left` | `sensor_msgs/Range` | 10-50 Hz |
| Distance (Right) | `/scan/right` | `sensor_msgs/Range` | 10-50 Hz |
| Odometry | `/odom` | `nav_msgs/Odometry` | 20-100 Hz |
| Battery | `/system/battery_voltage` | `std_msgs/Float32` | 1-10 Hz |
| Temperature | `/system/cpu_temp` | `std_msgs/Float32` | 1-5 Hz |
| Camera | `/camera/rgb/compressed` | `sensor_msgs/CompressedImage` | 10-30 Hz |

---

## Control Output

**Brain publishes**:
1. `/cmd_vel` (`geometry_msgs/Twist`) — Velocity commands for motors
   - `linear.x` — Forward velocity [-0.5, +0.5] m/s
   - `angular.z` — Rotation [-1.0, +1.0] rad/s
   
2. `/brain/motor_command` (`std_msgs/Float32MultiArray`) — Raw motor outputs [8 dimensions]

**Safety constraints** automatically enforced:
- Max linear: 0.5 m/s
- Max angular: 1.0 rad/s
- Max acceleration: 0.1 m/s² linear, 0.2 rad/s² angular

---

## Common Tasks

### Task: Use Custom IMU Topic

```bash
ros2 launch robot_brain_ros2 brain_stack.launch.py \
    imu_topic:=/xsens/imu
```

Or edit `sensor_aggregator.py` line 45:
```python
self.create_subscription(Imu, '/xsens/imu', self._imu_callback, 10)
```

### Task: Use Different Model File

```bash
ros2 launch robot_brain_ros2 brain_stack.launch.py \
    model_path:=/path/to/custom_model.pt
```

### Task: Change Velocity Scaling

Edit launch file parameters:
```python
{"cmd_scale_linear": 0.3},    # Reduce from 0.5
{"cmd_scale_angular": 0.5},   # Reduce from 1.0
```

### Task: Disable Camera (Bandwidth)

Launch parameter:
```python
{"camera_enabled": False}  # Already default
```

### Task: Check All Topics Exist

```bash
python scripts/verify_topic_compatibility.py --config config/brain_params.yaml
```

---

## Troubleshooting

### ❌ Topic not found
```bash
# Check if driver is running
ros2 node list | grep -E 'xsens|sick|camera'

# Check published topics
ros2 topic list | grep -E 'imu|scan|odom'

# **Solution**: Start sensor driver, then sensor aggregator
```

### ❌ Low frequency
```bash
# Check actual rate
ros2 topic hz /imu/data
# Expected: ≥ 50 Hz for IMU, ≥ 20 Hz for others

# **Solution**: Check USB cable, reduce other load
```

### ❌ Transforms missing
```bash
# View TF tree
ros2 run tf2_tools view_frames.py && evince frames.pdf

# **Solution**: Ensure transform_broadcaster is running
ros2 node list | grep tf_broadcaster
```

### ❌ Motors not responding
```bash
# Check velocity output
ros2 topic echo /cmd_vel | head -5

# Check safety status
ros2 topic echo /robot_brain_node | grep safety

# **Solution**: Verify sensor_aggregator is publishing (/brain/sensors)
```

---

## File Locations

```
/workspaces/Cogni-Sim-Robot-Brain/CogniSeer1.0-main/
├── robot_brain_ros2/
│   ├── robot_brain_ros2/
│   │   ├── sensor_aggregator.py          ← New
│   │   ├── transform_broadcaster.py      ← New
│   │   └── ros2_robot_brain_node.py      ← (Existing, unchanged)
│   ├── urdf/
│   │   └── cogniseer_robot.urdf.xacro    ← New
│   ├── launch/
│   │   └── brain_stack.launch.py         ← Updated
│   ├── package.xml                       ← Updated
│   └── setup.py                          ← Updated
├── docs/
│   ├── ROS2_INTEGRATION_OVERVIEW.md      ← New
│   ├── ROS2_SENSOR_TOPICS_AND_CONTROLLERS.md    ← New
│   └── ROS2_DRIVER_INTEGRATION.md        ← New
└── scripts/
    └── verify_topic_compatibility.py     ← New
```

---

## Documentation Guide

| When You Need... | Read... |
|---|---|
| **System overview** | `ROS2_INTEGRATION_OVERVIEW.md` |
| **Sensor specifications** | `ROS2_SENSOR_TOPICS_AND_CONTROLLERS.md` |
| **Driver examples** | `ROS2_DRIVER_INTEGRATION.md` |
| **Quick answers** | This file |

---

## Complete Integration Example

```bash
# Terminal 1: Launch sensor drivers
ros2 launch xsens_mti_driver xsens_mti_node.launch.py
ros2 launch sick_scan2 sick_tim781s.launch.py

# Terminal 2: Launch brain stack
ros2 launch robot_brain_ros2 brain_stack.launch.py

# Terminal 3: Monitor output
ros2 topic hz /brain/sensors
ros2 topic echo /cmd_vel

# Terminal 4: Verify
python scripts/verify_topic_compatibility.py --config config/brain_params.yaml
```

---

## Key Improvements

✅ **Modular Sensors**: Each sensor can be enabled/disabled
✅ **Standard Topics**: Uses ROS2 standard message types
✅ **Transforms**: Complete TF2 tree for visualization
✅ **URDF**: Robot description for RViz/Gazebo/Nav2
✅ **Verified**: Automated compatibility checking
✅ **Documented**: 40+ KB of integration guides
✅ **Examples**: 10+ sensor driver patterns

---

## Next Actions

1. **Read** [`ROS2_INTEGRATION_OVERVIEW.md`](../docs/ROS2_INTEGRATION_OVERVIEW.md) for architecture
2. **Install** sensor drivers per [`ROS2_DRIVER_INTEGRATION.md`](../docs/ROS2_DRIVER_INTEGRATION.md)
3. **Launch** `ros2 launch robot_brain_ros2 brain_stack.launch.py`
4. **Verify** `python verify_topic_compatibility.py`
5. **Monitor** `/cmd_vel` and `/brain/sensors` topics

---

**Status**: ✅ Complete — Ready for production ROS2 deployment

