# ROS2 Sensor Topics and Controller Interfaces

## Overview

This document specifies all sensor inputs, outputs, and controller interfaces for the CogniSeer robot brain in ROS2.

---

## Input Topics (Sensor Data)

### Inertial Measurement Unit (IMU)

**Topic**: `/imu/data`  
**Message Type**: `sensor_msgs/Imu`  
**Frequency**: 50-200 Hz  
**Required**: Yes (core for state estimation)  

**Observation Vector Mapping** (Indices 0-5):
- `[0:3]` — Linear acceleration (m/s²) — `linear_acceleration.[x,y,z]`
- `[3:6]` — Angular velocity (rad/s) — `angular_velocity.[x,y,z]`

**Common Drivers**:
- **VectorNav VN-300**: `vectornav_msgs/IMU` → Use adapter to convert
- **Xsens MTi**: `sensor_msgs/Imu` — Native support
- **ICM-20948**: ROS driver publishes `sensor_msgs/Imu`
- **BNO055**: Common open-source ROS drivers available

**Calibration**:
- Accel bias: Typically handled by driver
- Gyro bias: Estimate at startup (vehicle at rest)
- Temperature compensation: Driver-dependent

---

### Distance Sensors (Proximity/Ultrasonic/LiDAR)

**Topics**:
- `/scan/front` — Front-facing sensor
- `/scan/left` — Left-facing sensor  
- `/scan/right` — Right-facing sensor

**Message Type**: `sensor_msgs/Range`  
**Frequency**: 10-50 Hz per sensor  
**Required**: Yes (obstacle avoidance)

**Observation Vector Mapping** (Indices 6-8):
- `[6]` — Front distance (normalized: 0=close, 1=far)
- `[7]` — Left distance (normalized)
- `[8]` — Right distance (normalized)

**Range Field Mapping**:
```
observation_idx = (sensor_range - min_range) / (max_range - min_range)
observation_idx = clip(observation_idx, 0.0, 1.0)
```

**Common Drivers**:
- **HC-SR04 Ultrasonic**: ROS2 driver at `github.com/ros-drivers/ros_astra_camera`
- **Livox Mid-360 LiDAR**: Native ROS2 driver available
- **Sick TiM781S**: ROS2 sick_scan2 package
- **VL53L0X IR Sensor**: Custom ROS2 wrapper needed

**Typical Ranges**:
- Ultrasonic: 2cm - 4m
- IR: 10cm - 2m
- LiDAR: 5cm - 25m (varies by model)

---

### Odometry (Wheel Encoders)

**Topic**: `/odom`  
**Message Type**: `nav_msgs/Odometry`  
**Frequency**: 20-100 Hz  
**Required**: Yes (for localization)

**Observation Vector Mapping** (Indices 9-13):
- `[9:11]` — Position [x, y] (meters) — `pose.pose.position.[x,y]`
- `[11]` — Heading theta (radians) — Extracted from quaternion
- `[12]` — Linear velocity (m/s) — `twist.twist.linear.x`
- `[13]` — Angular velocity (rad/s) — `twist.twist.angular.z`

**Frame Requirements**:
- **Parent Frame**: `odom`
- **Child Frame**: `base_link`
- Must be a valid TF transform

**Common Drivers**:
- **Pololu Motor Encoders**: Custom ROS2 driver
- **MaxSonar MB1092**: Encoder via GPIO
- **ROS2 differential_drive_controller**: Integrates wheel odometry

---

### Battery Voltage

**Topic**: `/system/battery_voltage`  
**Message Type**: `std_msgs/Float32`  
**Frequency**: 1-10 Hz  
**Required**: Optional (safety-critical if present)

**Observation Vector Mapping** (Index 14):
- `[14]` — Battery level (normalized: 0-1)
- Mapping: `(voltage - min_v) / (max_v - min_v)`

**Typical Voltage Ranges**:
- 3S LiPo: 10.5V (discharged) - 12.6V (full)
- 4S LiPo: 14.0V - 16.8V
- Lead-acid: 11.2V - 13.2V

**Safety Thresholds**:
- Yellow alert: < 20% battery
- Red alert/shutdown: < 10% battery
- Configure in `safety.yaml`

---

### System Temperature

**Topic**: `/system/cpu_temp`  
**Message Type**: `std_msgs/Float32`  
**Frequency**: 1-5 Hz  
**Required**: Optional (thermal throttling prevention)

**Observation Vector Mapping** (Index 15):
- `[15]` — CPU temperature (normalized: 0-1)
- Mapping: `temperature / 100.0` (assumes 0-100°C range)

**Safe Operating Range**:
- Optimal: 25-60°C
- Thermal warning: > 75°C
- Shutdown: > 90°C (configurable)

**Typical Sources**:
- Jetson Nano: `/sys/devices/virtual/thermal/thermal_zone0/temp`
- x86/Arm CPU: `lm-sensors` via ROS bridge
- Custom thermal sensor on main board

---

### Camera (Optional)

**Topic**: `/camera/rgb/compressed`  
**Message Type**: `sensor_msgs/CompressedImage`  
**Frequency**: 10-30 Hz  
**Required**: Optional (visual processing)

**Observation Vector Mapping** (Indices 16-31):
- `[16:32]` — Compressed image features (16 values from 4x4 resize)

**Processing Pipeline**:
1. Decompress JPEG/PNG
2. Resize to 4×4 pixels (16 values)
3. Convert to grayscale
4. Normalize to [0, 1]

**Common Drivers**:
- **USB Webcam**: `ros2_usb_camera`
- **CSI Camera (Raspberry Pi)**: `rpicam-ros`
- **Intel RealSense**: `realsense-ros` package
- **Basler ace2**: `pylon_ros2` package

**Bandwidth Notes**:
- JPEG compression essential (10-50 KB per frame vs 600+ KB uncompressed)
- At 20 Hz, compressed: 200-1000 KB/s bandwidth

---

## Output Topics (Control Commands)

### Velocity Command

**Topic**: `/cmd_vel`  
**Message Type**: `geometry_msgs/Twist`  
**Frequency**: 20-100 Hz (as produced by brain node)  
**Source**: `robot_brain_node`

**Message Structure**:
```yaml
linear:
  x: 0.5  # Forward velocity (m/s), range: [-1.0, 1.0]
  y: 0.0  # Lateral velocity (m/s), typically 0 for differential drive
  z: 0.0  # Vertical velocity (m/s), typically 0 for ground robots
angular:
  x: 0.0  # Roll rate (rad/s), typically 0
  y: 0.0  # Pitch rate (rad/s), typically 0
  z: 0.5  # Yaw rate (rad/s), range: [-1.0, 1.0]
```

**Typical Ranges**:
- **Differential Drive Robot**: x ∈ [-1, 1] m/s, z ∈ [-1, 1] rad/s
- **Holonomic Robot**: x, y ∈ [-1, 1] m/s, z ∈ [-1, 1] rad/s
- **Ackermann Steering**: x ∈ [0, 1] m/s, steering angle encoded in z

**Common Subscribers**:
- Motor controller drivers (convert Twist → PWM signals)
- Simulation (gazebo_ros_pkgs)
- Nav2 stack (when operating autonomously)

**Safety Constraints** (Enforced by `safety_layer.py`):
```python
max_linear_x = 0.5  # m/s
max_angular_z = 1.0  # rad/s
max_linear_delta = 0.10  # Max acceleration (m/s²) per control step
max_angular_delta = 0.20  # Max angular acceleration (rad/s²) per step
```

---

### Motor Command (Raw)

**Topic**: `/brain/motor_command`  
**Message Type**: `std_msgs/Float32MultiArray`  
**Frequency**: 20-100 Hz (synchronized with brain updates)  
**Source**: `robot_brain_node`

**Message Structure**:
```yaml
layout:
  dim:
    - label: "motor_command"
      size: 8
      stride: 8
data: [0.1, 0.2, -0.05, 0.15, ...]  # 8 float values
```

**Typical Usage**:
- `[0:2]` — Left wheel [velocity, brake]
- `[2:4]` — Right wheel [velocity, brake]
- `[4:6]` — Servo/joint commands
- `[6:8]` — Reserved for future use

**Range**: Typically [-1.0, 1.0] per motor

**Common Subscribers**:
- Custom motor controller driver
- Hardware abstraction layer
- Simulation (can map directly to Gazebo)

---

## Sensor Topics (Aggregated)

### Brain Sensor Input

**Topic**: `/brain/sensors`  
**Message Type**: `std_msgs/Float32MultiArray`  
**Frequency**: 50 Hz (fixed update rate)  
**Source**: `sensor_aggregator` node

**Observation Vector Structure** (32 dimensions):

| Index | Sensor | Unit | Range | Description |
|-------|--------|------|-------|-------------|
| 0-2 | IMU Accel | m/s² | [-20, 20] | Linear acceleration [x,y,z] |
| 3-5 | IMU Gyro | rad/s | [-3.14, 3.14] | Angular velocity [x,y,z] |
| 6 | Distance Front | normalized | [0, 1] | Front obstacle distance |
| 7 | Distance Left | normalized | [0, 1] | Left obstacle distance |
| 8 | Distance Right | normalized | [0, 1] | Right obstacle distance |
| 9-10 | Odometry Pos | meters | [-inf, inf] | Position [x, y] |
| 11 | Odometry Heading | radians | [-π, π] | Yaw angle |
| 12 | Odometry Lin Vel | m/s | [-1, 1] | Linear velocity |
| 13 | Odometry Ang Vel | rad/s | [-1, 1] | Angular velocity |
| 14 | Battery | normalized | [0, 1] | Battery voltage (normalized) |
| 15 | Temperature | normalized | [0, 1] | CPU temperature (normalized) |
| 16-31 | Camera | - | [0, 1] | Camera features (4×4 image) |

---

## Topic Compatibility Matrix

### Verified Compatible Topics

| Topic | Message Type | Driver/Source | ROS2 Version | Status |
|-------|---|---|---|---|
| `/imu/data` | `sensor_msgs/Imu` | Standard drivers | Humble, Iron | ✅ Tested |
| `/scan/front` | `sensor_msgs/Range` | Custom drivers | Humble, Iron | ✅ Tested |
| `/odom` | `nav_msgs/Odometry` | Custom drivers | Humble, Iron | ✅ Tested |
| `/cmd_vel` | `geometry_msgs/Twist` | Standard | Humble, Iron | ✅ Tested |
| `/system/battery_voltage` | `std_msgs/Float32` | Custom | Humble, Iron | ⚠️ Optional |
| `/system/cpu_temp` | `std_msgs/Float32` | Custom | Humble, Iron | ⚠️ Optional |
| `/camera/rgb/compressed` | `sensor_msgs/CompressedImage` | Various | Humble, Iron | ⚠️ Optional |

### Topic Remapping

If your sensor drivers publish to different topics, use ROS2 remapping:

```bash
# Launch with remapping
ros2 launch robot_brain_ros2 brain_stack.launch.py \
  imu_topic:=/imu0 \
  distance_topic:=/lidar/ranges \
  odom_topic:=/wheel_odom
```

Or in launch file:
```python
# launch.py
remappings = [
    ('imu_topic', '/custom/imu'),
    ('distance_topic', '/custom/distance'),
    ('odom_topic', '/custom/odom'),
]
```

---

## Motor Controller Interfaces

### Example 1: PWM Motor Controller

**Hardware**: Standard PWM motor controller (e.g., Adafruit Motor FeatherWing)

```python
# Pseudocode: ROS2 node to convert /cmd_vel → PWM signals
from geometry_msgs.msg import Twist
import board
import pwmio

class MotorControllerNode(Node):
    def __init__(self):
        super().__init__('motor_controller')
        self.left_pwm = pwmio.PWMOut(board.D5, frequency=1000, duty_cycle=0)
        self.right_pwm = pwmio.PWMOut(board.D6, frequency=1000, duty_cycle=0)
        self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
    
    def cmd_callback(self, msg: Twist):
        # Convert to motor commands
        left_speed = msg.linear.x - msg.angular.z * 0.1
        right_speed = msg.linear.x + msg.angular.z * 0.1
        
        # Map to PWM [0, 65535]
        self.left_pwm.duty_cycle = int((left_speed + 1.0) / 2.0 * 65535)
        self.right_pwm.duty_cycle = int((right_speed + 1.0) / 2.0 * 65535)
```

### Example 2: ROS2 Control Interface

**Using `ros2_control` with transmission interfaces:**

```xml
<!-- In URDF -->
<transmission name="left_wheel">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="left_wheel_joint">
    <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
  </joint>
  <actuator name="left_motor">
    <mechanicalReduction>1</mechanicalReduction>
    <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
  </actuator>
</transmission>
```

Then use standard `joint_trajectory_controller`:

```bash
ros2 control list_hardware_interfaces
ros2 control list_controllers
ros2 control load_controller differential_drive_controller
```

### Example 3: CAN Bus Motor Controller

**For high-speed systems using CAN FD:**

```python
# Custom CAN message formatter
def twist_to_can(msg: Twist):
    left_velocity = msg.linear.x - msg.angular.z * 0.125
    right_velocity = msg.linear.x + msg.angular.z * 0.125
    
    # Pack into CAN frames (ID 0x100, 0x101)
    # Frame 0x100: left motor velocity (int16)
    # Frame 0x101: right motor velocity (int16)
    ...
```

---

## Sensor Integration Checklist

### Before Running the Brain Node

- [ ] IMU publishing to `/imu/data` at ≥50 Hz
- [ ] Distance sensors publishing to `/scan/[front,left,right]`
- [ ] Odometry publishing to `/odom` with valid TF
- [ ] Motor controller subscribed to `/cmd_vel`
- [ ] All timestamps are synchronized (NTP or bus synchronization)

### Health Checks

```bash
# Check topic rates
ros2 topic hz /imu/data
ros2 topic hz /scan/front
ros2 topic hz /odom

# Verify message contents
ros2 topic echo /imu/data | head -20
ros2 topic echo /odom | head -10

# Check TF tree
ros2 run tf2_tools view_frames.py
# Creates frames.pdf showing transform hierarchy

# Verify sensor aggregator
ros2 service call /brain/sensor_health std_srvs/Trigger
```

---

## Sensor Fusion & Kalman Filtering

For production systems, consider sensor fusion:

```bash
# Use robot_localization for state estimation
ros2 launch robot_localization ekf.launch.py

# This provides fused /odometry/filtered topic
# More robust than individual sensor outputs
```

**Advantages**:
- Smooths noisy sensor data
- Handles sensor dropouts gracefully
- Estimates uncertainties
- Handles time synchronization

---

## References

- **ROS2 Message Definitions**: https://github.com/ros2/common_interfaces
- **Sensor Drivers**: https://github.com/ros-drivers/
- **Robot Localization (Sensor Fusion)**: http://docs.ros.org/en/humble/Concepts/Intermediate/ROS-2-Sensor-Fusion.html
- **CogniSeer Sensor Adaptor**: `robot_brain_ros2/sensor_aggregator.py`

