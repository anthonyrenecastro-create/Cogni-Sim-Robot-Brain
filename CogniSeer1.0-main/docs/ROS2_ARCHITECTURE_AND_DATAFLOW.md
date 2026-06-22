# ROS2 Software Architecture & Data Flow

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     HARDWARE LAYER                          │
│  (IMU, LiDAR, Encoders, Camera, Battery Sensor, etc.)      │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  SENSOR DRIVERS                             │
│  (Native ROS2 drivers for each hardware sensor)             │
│  - Xsens MTi (IMU)                                          │
│  - Sick/Livox (LiDAR/Distance)                              │
│  - Wheel encoders (Odometry)                                │
│  - USB camera, etc.                                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼───────────────┬──────────────┐
        │              │               │              │
        ▼              ▼               ▼              ▼
    /imu/data    /odom            /scan/*    /system/*
   (Imu msg)  (Odometry)      (Range msgs)  (Float32)
        │              │               │              │
        └──────────────┼───────────────┴──────────────┘
                       │
        ┌──────────────▼──────────────────────┐
        │     SENSOR AGGREGATOR NODE          │
        │  (sensor_aggregator.py)             │
        │                                     │
        │  • Subscribes to all sensors        │
        │  • Normalizes to [0, 1] range       │
        │  • Creates 32D vector               │
        │  • Publishes /brain/sensors         │
        │  • Health monitoring                │
        └──────────────┬──────────────────────┘
                       │ /brain/sensors
                       │ (32D vector @ 50Hz)
                       │
        ┌──────────────▼──────────────────────┐
        │    ROBOT BRAIN NODE                 │
        │  (ros2_robot_brain_node.py)         │
        │                                     │
        │  • Loads INT8 PyTorch model         │
        │  • Inference loop (20 Hz)           │
        │  • Converts output to Twist         │
        │  • Publishes /cmd_vel               │
        └──────────────┬──────────────────────┘
                       │ /cmd_vel
                       │ (Twist msg)
                       │
        ┌──────────────▼──────────────────────┐
        │      SAFETY LAYER                   │
        │  (safety_layer.py)                  │
        │                                     │
        │  • Bounds velocities                │
        │  • Enforces accel limits            │
        │  • Checks timeouts                  │
        │  • Stops on stale sensors           │
        └──────────────┬──────────────────────┘
                       │ (Bounded velocities)
                       │
        ┌──────────────▼──────────────────────┐
        │    MOTOR CONTROLLERS                │
        │  (Hardware dependent)               │
        │                                     │
        │  • PWM drivers                      │
        │  • ros2_control interface           │
        │  • CAN bus interface                │
        └──────────────┬──────────────────────┘
                       │
                       ▼
                   MOTORS → MOVEMENT
```

---

## Data Flow Details

### Input: Sensor Data → Observation Vector

**Stage 1: Raw Sensor Data (Multiple Topics)**

```python
# Each sensor publishes independently
/imu/data: sensor_msgs/Imu
  ├─ imu.linear_acceleration: [ax, ay, az]
  ├─ imu.angular_velocity: [wx, wy, wz]
  └─ imu.header.stamp: timestamp

/scan/front: sensor_msgs/Range
  ├─ range: distance in meters
  └─ range_min, range_max

/odom: nav_msgs/Odometry
  ├─ pose.pose.position: [x, y, z]
  ├─ pose.pose.orientation: [qx, qy, qz, qw]
  └─ twist.twist.linear/angular velocities
```

**Stage 2: Sensor Aggregator Processing**

```python
# sensor_aggregator.py processes all topics into single vector

class SensorAggregatorNode(Node):
    def __init__(self):
        # Subscribe to all sensors
        self.create_subscription(Imu, '/imu/data', self._imu_callback)
        self.create_subscription(Range, '/scan/front', self._distance_callback)
        self.create_subscription(Odometry, '/odom', self._odom_callback)
        # ... more subscriptions
        
        # Maintain buffers for each sensor
        self.imu_data = [0] * 6
        self.distance_data = [0] * 3
        self.odom_data = [0] * 5
        # ...
        
        # Publish aggregated vector
        self.pub = self.create_publisher(Float32MultiArray, '/brain/sensors', 10)
        
        # Timer: aggregate and publish every 20ms (50 Hz)
        self.timer = self.create_timer(0.02, self._publish_aggregated)
    
    def _imu_callback(self, msg: Imu):
        """Extract IMU data to [0:6]"""
        self.imu_data[0:3] = msg.linear_acceleration  # [0:3]
        self.imu_data[3:6] = msg.angular_velocity     # [3:6]
    
    def _distance_callback(self, msg: Range):
        """Accumulate distance data to [6:8]"""
        # Aggregate front/left/right to [6:8]
        ...
    
    def _odom_callback(self, msg: Odometry):
        """Extract odometry to [9:13]"""
        self.odom_data[0:2] = msg.pose.pose.position  # [9:10]
        self.odom_data[2] = quaternion_to_yaw(...)    # [11]
        self.odom_data[3:5] = msg.twist.twist.linear/angular  # [12:13]
    
    def _publish_aggregated(self):
        """Create 32D vector and publish"""
        obs = Float32MultiArray()
        obs.data = [
            *self.imu_data,           # [0:6]
            *self.distance_data,      # [6:8]
            *self.odom_data,          # [9:13]
            self.battery_voltage,     # [14]
            self.cpu_temperature,     # [15]
            *self.camera_features,    # [16:31]
        ]
        self.pub.publish(obs)
```

**Stage 3: Observation Vector (32D)**

```
Index    Name              Min    Max    Unit       Source
────────────────────────────────────────────────────────────────
0-2      accel_x/y/z       -16   +16    m/s²       IMU
3-5      gyro_x/y/z        -250  +250   rad/s      IMU
6-8      distance_f/l/r    0     1      normalized Range sensors
9-10     position_x/y      any   any    meters     Odometry
11       yaw               -π    +π     radians    Odometry
12-13    vel_linear/ang    any   any    m/s,rad/s  Odometry
14       battery_volt      0.0   1.0    normalized Float32 (10.5-12.6V)
15       temp_celsius      0.0   1.0    normalized Float32 (0-100°C)
16-31    camera_features   0.0   1.0    normalized 4×4 grayscale
────────────────────────────────────────────────────────────────
```

---

### Processing: Observation → Brain Inference → Control

**Stage 1: Brain Node Receives Observation**

```python
# ros2_robot_brain_node.py

class RobotBrainNode(Node):
    def __init__(self):
        # Load the quantized model
        self.model = torch.jit.load("robot_brain_int8_stateful.pt")
        
        # Subscribe to aggregated sensors
        self.create_subscription(Float32MultiArray, '/brain/sensors', 
                               self._sensor_callback)
        
        # Publish velocity commands
        self.pub_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_motor = self.create_publisher(Float32MultiArray, 
                                              '/brain/motor_command', 10)
        
        # Control loop at 20 Hz
        self.timer = self.create_timer(0.05, self._control_loop)
        
        self.latest_obs = None
        
    def _sensor_callback(self, msg: Float32MultiArray):
        """Store latest observation"""
        self.latest_obs = torch.tensor(msg.data, dtype=torch.float32)
    
    def _control_loop(self):
        """Run inference and publish commands"""
        if self.latest_obs is None:
            return
        
        # Inference: 32D input → control outputs
        with torch.no_grad():
            # Model expects batch dimension
            input_tensor = self.latest_obs.unsqueeze(0)  # [1, 32]
            output = self.model(input_tensor)
            
            # Output usually [linear_vel, angular_vel, ...]
            linear_vel = float(output[0, 0])
            angular_vel = float(output[0, 1])
        
        # Convert to ROS2 Twist message
        twist = Twist()
        twist.linear.x = self.cmd_scale_linear * linear_vel
        twist.angular.z = self.cmd_scale_angular * angular_vel
        
        # Apply safety constraints
        twist = self.safety_layer.bound_command(twist)
        
        # Publish
        self.pub_vel.publish(twist)
```

**Stage 2: Safety Layer Constraints**

```python
# safety_layer.py

class SafetyLayer:
    def __init__(self):
        self.max_linear_x = 0.5      # m/s
        self.max_angular_z = 1.0     # rad/s
        self.max_linear_accel = 0.1  # m/s²
        self.max_angular_accel = 0.2 # rad/s²
        self.sensor_timeout = 0.5    # seconds
    
    def bound_command(self, cmd: Twist) -> Twist:
        """Apply safety constraints"""
        # Bound velocities
        cmd.linear.x = np.clip(cmd.linear.x, -self.max_linear_x, self.max_linear_x)
        cmd.angular.z = np.clip(cmd.angular.z, -self.max_angular_z, self.max_angular_z)
        
        # Bound acceleration
        if hasattr(self, 'prev_cmd'):
            dt = (time.time() - self.prev_time)
            lin_accel = (cmd.linear.x - self.prev_cmd.linear.x) / dt
            ang_accel = (cmd.angular.z - self.prev_cmd.angular.z) / dt
            
            if abs(lin_accel) > self.max_linear_accel:
                # Reduce command
                cmd.linear.x = self.prev_cmd.linear.x + \
                             np.sign(lin_accel) * self.max_linear_accel * dt
            # Similar for angular...
        
        # Check sensor timeout
        if self.get_sensor_age() > self.sensor_timeout:
            if self.stop_on_stale_sensor:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
        
        # Store for next iteration
        self.prev_cmd = cmd
        self.prev_time = time.time()
        
        return cmd
```

**Stage 3: Motor Controller Output**

```python
# Motor controllers interpret /cmd_vel

# Option 1: Differential Drive (most common)
def differential_drive_from_twist(twist: Twist, wheelbase: float):
    v = twist.linear.x
    omega = twist.angular.z
    
    # For differential drive robot
    # v_left = v - (wheelbase/2) * omega
    # v_right = v + (wheelbase/2) * omega
    
    left_speed = v - (wheelbase/2) * omega
    right_speed = v + (wheelbase/2) * omega
    
    return left_speed, right_speed

# Option 2: Omnidirectional (Mecanum wheels)
def omnidirectional_from_twist(twist: Twist):
    vx = twist.linear.x
    vy = twist.linear.y
    omega = twist.angular.z
    
    # Convert to 4-wheel commands
    # Each wheel gets component of vx, vy, rotation
    ...
```

---

## Transform Tree (TF2)

**Broadcaster**: `transform_broadcaster.py` publishes:

```
odom
  └─ base_footprint (fixed transform)
      ├─ base_link
      │   ├─ left_wheel_link
      │   ├─ right_wheel_link
      │   ├─ imu_link
      │   ├─ camera_link
      │   └─ distance_sensor_front_link
      └─ ...
```

**Usage**:
- RViz uses TF to visualize robot structure
- Nav2 uses TF for localization
- Motion planning uses TF for coordinate transforms

---

## Execution Timeline (50 Hz Control Loop)

```
Time (ms)  |  Component            |  Operation
───────────┼──────────────────────┼─────────────────────────────
0          |  Sensors             |  Publish latest data
           |  - IMU @ 50-200 Hz   |
           |  - Distance @ 20 Hz  |
           |  - Odometry @ 50 Hz  |
───────────┼──────────────────────┼─────────────────────────────
0-2        |  Sensor Aggregator   |  Receive all sensor callbacks
           |                      |  Update internal buffers
───────────┼──────────────────────┼─────────────────────────────
20         |  Sensor Aggregator   |  Publish /brain/sensors (32D)
───────────┼──────────────────────┼─────────────────────────────
20-30      |  Brain Node          |  Run PyTorch inference
           |                      |  (50-100ms typical)
───────────┼──────────────────────┼─────────────────────────────
30-35      |  Safety Layer        |  Bound velocity commands
───────────┼──────────────────────┼─────────────────────────────
35-40      |  Brain Node          |  Publish /cmd_vel
───────────┼──────────────────────┼─────────────────────────────
40-50      |  Motor Controllers   |  Receive /cmd_vel
           |                      |  Apply to hardware
```

---

## Node Dependencies Graph

```
xsens_mti_driver
    │
    ├─ /imu/data ─────┐
                      │
sick_scan2            │
    │                 │
    ├─ /scan/front ──┐│
                      ││
encoders_driver       ││
    │                 ││
    ├─ /odom ────────┐││
                      │││
battery_monitor       │││
    │                 │││
    ├─ /system/* ───┐│││
                     │││
sensor_aggregator ◄──┴┴┴ Subscribes to all sensors
    │
    ├─ /brain/sensors
    │
    └──► robot_brain_node ◄─── Subscribes
         │
         ├─ /cmd_vel
         │
         └──► Motor Controller
              │
              └──► Hardware (motors, wheels)
```

---

## Configuration Hierarchy

```
Global Configuration
├─ config/defaults.json          (Brain model parameters)
│   ├─ input_dim: 32
│   ├─ hidden_layers: [128, 64]
│   ├─ output_dim: 2
│   └─ model_type: "snn"
│
├─ config/brain_params.yaml      (ROS2 parameters)
│   ├─ robot:
│   │   └─ wheelbase: 0.25
│   ├─ sensors:
│   │   ├─ imu:
│   │   │   ├─ enabled: true
│   │   │   └─ offset: [0.0, 0.0, 0.08]
│   │   └─ camera:
│   │       ├─ enabled: false
│   │       └─ offset: [0.2, 0.0, 0.12]
│   └─ safety:
│       ├─ max_linear: 0.5
│       ├─ max_angular: 1.0
│       └─ sensor_timeout: 0.5
│
└─ brain_stack.launch.py         (Launch-time overrides)
    └─ Parameters can be overridden at launch:
       ros2 launch robot_brain_ros2 brain_stack.launch.py \
           model_path:=/custom/model.pt
```

---

## Debugging: Data Flow Inspection

### Check Sensor Data

```bash
# Raw sensor
ros2 topic echo /imu/data | head -5

# Aggregated
ros2 topic echo /brain/sensors | head -3
```

### Check Inference Output

```bash
# Velocity commands
ros2 topic echo /cmd_vel | head -5

# Motor commands (if published)
ros2 topic echo /brain/motor_command | head -5
```

### Check Frequencies

```bash
# Expected frequencies
ros2 topic hz /brain/sensors  # Should be 50 Hz
ros2 topic hz /cmd_vel        # Should be 20 Hz

# If too low, sensors might be missing
ros2 topic hz /imu/data       # Check individual sensor
```

### Check Transforms

```bash
# View TF tree
ros2 run tf2_tools view_frames.py

# Check specific transform
ros2 run tf2_ros tf2_echo base_link imu_link
```

---

## Performance Analysis

| Stage | Typical Latency |
|-------|-----------------|
| Sensor capture → publish | 2-10 ms (hardware dependent) |
| Sensor aggregator processing | 2-5 ms |
| Brain inference | 50-100 ms (PyTorch INT8) |
| Safety layer | <1 ms |
| Motor controller response | 5-20 ms (hardware dependent) |
| **Total latency** | **~65-135 ms** |
| **Control frequency** | **20 Hz** (50ms cycle time) |

For most navigation tasks, 20 Hz is adequate. For manipulation, may need higher rates (increase to 50-100 Hz with GPU acceleration).

---

## References

- **ROS2 Concepts**: https://docs.ros.org/en/humble/Concepts.html
- **Message Types**: https://github.com/ros2/common_interfaces
- **PyTorch ROS2 Integration**: https://github.com/ros2/ros2_tracing
- **Safety Best Practices**: https://github.com/ros-safety/safe_robot_motion

