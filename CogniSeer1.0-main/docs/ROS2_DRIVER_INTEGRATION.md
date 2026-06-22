# ROS2 Sensor Driver Integration Guide

## Overview

This guide provides examples and best practices for integrating common robot sensors with the CogniSeer robot brain.

---

## Sensor Integration Patterns

### Pattern 1: Native ROS2 Drivers

**Most sensors have native ROS2 drivers available.** These are the easiest to integrate.

**Example: Livox LiDAR**

```bash
# Install driver
sudo apt install ros-humble-livox-ros-driver2

# Launch with topic remapping
ros2 launch livox_ros_driver2 rviz2.launch.py
```

Then remap in your brain launch file:
```python
remappings = [('lidar_frame', '/livox/lidar')],
```

### Pattern 2: Custom ROS2 Node Wrapper

**For sensors without native drivers, create a simple wrapper node.**

Example: Ultrasonic Distance Sensor (HC-SR04)

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range
import RPi.GPIO as GPIO
import time

class UltrasonicDriverNode(Node):
    def __init__(self):
        super().__init__('ultrasonic_driver')
        
        # GPIO setup
        self.TRIG = 23
        self.ECHO = 24
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.TRIG, GPIO.OUT)
        GPIO.setup(self.ECHO, GPIO.IN)
        
        self.pub = self.create_publisher(Range, '/scan/front', 10)
        self.timer = self.create_timer(0.05, self.measure)  # 20 Hz
    
    def measure(self):
        # Send trigger pulse
        GPIO.output(self.TRIG, True)
        time.sleep(0.00001)
        GPIO.output(self.TRIG, False)
        
        # Measure echo time
        while GPIO.input(self.ECHO) == 0:
            pulse_start = time.time()
        while GPIO.input(self.ECHO) == 1:
            pulse_end = time.time()
        
        # Calculate distance
        distance = (pulse_end - pulse_start) * 17150  # cm
        distance = distance / 100  # convert to meters
        
        # Publish
        msg = Range()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "distance_sensor_front_link"
        msg.radiation_type = Range.ULTRASOUND
        msg.field_of_view = 0.5  # radians
        msg.min_range = 0.02
        msg.max_range = 4.0
        msg.range = distance
        
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = UltrasonicDriverNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Pattern 3: Hardware Abstraction Layer (HAL)

**For complex systems, use a HAL to decouple hardware from ROS2.**

```python
# hal.py - Hardware-agnostic sensor reading
class DistanceSensorHAL:
    def read(self) -> float:
        """Returns distance in meters."""
        raise NotImplementedError

class UltrasonicHAL(DistanceSensorHAL):
    def __init__(self, trig_pin: int, echo_pin: int):
        self.trig = trig_pin
        self.echo = echo_pin
    
    def read(self) -> float:
        # Hardware-specific code
        ...

# driver.py - ROS2 node using HAL
class SensorDriver(Node):
    def __init__(self, hal: DistanceSensorHAL):
        self.hal = hal
        self.timer = self.create_timer(0.05, self._publish)
    
    def _publish(self):
        distance = self.hal.read()
        msg = Range()
        msg.range = distance
        self.pub.publish(msg)

# Can swap implementations: UltrasonicHAL ↔ LidarHAL
```

---

## Common Sensor Integration Examples

### 1. IMU — Xsens MTi

**Most common IMU for robotics, native ROS2 support.**

```bash
# Install
sudo apt install ros-humble-xsens-mti-driver

# Launch
ros2 launch xsens_mti_driver xsens_mti_node.launch.py port:=/dev/ttyUSB0

# Verifies: Publishes to /imu/data (sensor_msgs/Imu)
```

**Fallback: VectorNav VN-300 (AHRS-capable)**

```bash
# Install custom driver
git clone https://github.com/ros-drivers/vectornav.git
colcon build

# Then convert to standard Imu message type
```

### 2. Distance Sensors — Multiple Options

#### Option A: Sick TiM781S LiDAR (Recommended for outdoor)

```bash
# Install
sudo apt install ros-humble-sick-scan2

# Launch
ros2 launch sick_scan2 sick_tim781s.launch.py
```

**Remap to standard topic:**
```python
RemapFrom('/scan/lidar')
RemapTo('/scan/front')
```

#### Option B: Livox Mid-360 (Recommended for indoor)

```bash
# More compact and affordable
sudo apt install ros-humble-livox-ros-driver2
ros2 launch livox_ros_driver2 rviz2.launch.py
```

#### Option C: HC-SR04 Ultrasonic (Cheap, DIY)

```bash
# Use custom driver (example above)
ros2 run my_drivers ultrasonic_driver &
```

### 3. Encoders — Wheel Odometry

#### Option A: Motor Encoder Feedback

```python
# Custom node reading encoder values via GPIO/CAN
class OdometryNode(Node):
    def __init__(self):
        self.left_ticks = 0
        self.right_ticks = 0
        
        # Subscribe to encoder events
        self.create_subscription(Int32, '/motor/left/ticks', self._left_tick, 10)
        self.create_subscription(Int32, '/motor/right/ticks', self._right_tick, 10)
        
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.timer = self.create_timer(0.05, self._publish_odom)
    
    def _publish_odom(self):
        msg = Odometry()
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # Convert ticks to distance
        x = self.left_ticks * TICK_TO_METER
        y = 0.0  # Differential drive
        theta = (self.right_ticks - self.left_ticks) * TICK_TO_RAD
        
        msg.pose.pose.position.set(x, y, 0)
        msg.pose.pose.orientation = quaternion_from_euler(0, 0, theta)
        
        self.odom_pub.publish(msg)
```

#### Option B: Visual Odometry (from camera)

```bash
# Use SLAM or visual odometry
ros2 launch rtabmap_ros rtabmap.launch.py rgb_topic:=/camera/rgb/image \
                                             depth_topic:=/camera/depth/image
```

### 4. Battery Monitoring

```python
# Simple battery monitor node
class BatteryMonitorNode(Node):
    def __init__(self):
        self.pub = self.create_publisher(Float32, '/system/battery_voltage', 10)
        
        # Read ADC (Analog-to-Digital) on Raspberry Pi or Jetson
        self.adc = ADC(channel=0, device='/dev/i2c-1')
        self.timer = self.create_timer(1.0, self._publish_voltage)
    
    def _publish_voltage(self):
        # Read ADC and convert to voltage
        raw = self.adc.read()
        voltage = raw * 5.0 / 4095.0 * DIVIDER_RATIO  # Voltage divider correction
        
        msg = Float32(data=voltage)
        self.pub.publish(msg)

# Threshold-based safety
battery_min = 10.5  # 3S LiPo cutoff
if voltage < battery_min:
    self.get_logger().fatal("Battery too low, shutting down")
    # Trigger safe shutdown
```

### 5. Camera Integration

```python
# Compressed camera stream for bandwidth efficiency
ros2 launch usb_cam camera_node.launch.py \
    video_device:=/dev/video0 \
    image_encoding:=jpeg

# Or if using Intel RealSense
ros2 launch realsense2_camera rs_launch.py \
    enable_gyro:=true \
    enable_accel:=true
```

---

## Topic Remapping for Custom Drivers

If your sensor drivers publish to non-standard topics, use remapping:

### In Launch File

```python
# brain_stack.launch.py
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

DeclareLaunchArgument('imu_topic', default_value='/imu/data'),
DeclareLaunchArgument('odom_topic', default_value='/odom'),

# Then in Node definition:
remappings=[
    ('/imu/data', LaunchConfiguration('imu_topic')),
    ('/odom', LaunchConfiguration('odom_topic')),
]
```

### Command Line

```bash
ros2 launch robot_brain_ros2 brain_stack.launch.py \
    imu_topic:=/imu0 \
    odom_topic:=/wheel_odom
```

### In sensor_aggregator.py

```python
# Modify subscription topics
self.create_subscription(Imu, '/imu0', self._imu_callback, 10)  # Custom topic
```

---

## TF2 Transform Setup

All sensors must have valid transforms to the robot frame.

### Static Transforms (Fixed sensors)

```python
# In transform_broadcaster.py
self._broadcast_transform(
    now,
    parent_frame="base_link",
    child_frame="imu_link",
    x=0.0, y=0.0, z=0.08,  # IMU is 8cm above base
    roll=0.0, pitch=0.0, yaw=0.0
)
```

### Dynamic Transforms (Movable sensors, e.g., pan-tilt)

```python
# Separate node that updates transforms dynamically
class PanTiltNode(Node):
    def __init__(self):
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(Float32, '/camera/pan', self._pan_callback, 10)
        self.create_subscription(Float32, '/camera/tilt', self._tilt_callback, 10)
    
    def _pan_callback(self, msg: Float32):
        # Update camera transform based on pan angle
        self._publish_camera_transform(pan=msg.data, tilt=self.tilt)
```

---

## Troubleshooting Integration Issues

### Issue: "No such device or address"

**Cause**: Sensor not connected or wrong device path

```bash
# Check connected devices
ls /dev/tty* | grep -E 'USB|ACM'

# Check USB devices
lsusb

# Fix in launch file
device:=/dev/ttyUSB0  # Adjust number if needed
```

### Issue: "Permission denied" writing to /dev/ttyUSB0

**Cause**: Missing user permissions

```bash
# Add user to dialout group
sudo usermod -a -G dialout $USER
# Log out and back in
```

### Issue: Topic not publishing at expected frequency

**Cause**: Driver buffering or slow bus (I2C, SPI)

```python
# Increase queue depth
self.create_subscription(..., qos_profile=QoSProfile(depth=50))

# Or use different transport (UDP if available)
```

### Issue: Timestamp synchronization problems

**Cause**: Sensors use different clocks

**Solution**: Use `robot_localization` EKF:
```bash
ros2 launch robot_localization ekf.launch.py
# Publishes filtered /odometry/filtered topic
```

---

## Performance Optimization Tips

### 1. Reduce Network Load

```python
# Compress camera images
CompressedImage  # JPEG @ 5-10 KB vs 600+ KB

# Reduce IMU frequency if not needed
sensor_msgs/Imu @ 50 Hz (not 200 Hz)

# Aggregate distance sensors
One topic with 3 range values vs 3 topics
```

### 2. Use Hardware Filters

```python
# IMU low-pass filters (often onboard)
imu.accel_bandwidth = 50  # Hz

# LiDAR skip-every-N-scan
lidar.scan_frequency_hz = 25  # Instead of 50
```

### 3. Optimize Update Rates

```yaml
# control_loop frequency should match sensor updates
control_frequency_hz: 50  # Match IMU @ 50 Hz
```

---

## Testing Sensor Integration

### Health Check Script

```bash
python verify_topic_compatibility.py --config config/brain_params.yaml
```

### Manual Topic Inspection

```bash
# List all published topics
ros2 topic list

# Monitor topic frequency
ros2 topic hz /imu/data

# Echo message content
ros2 topic echo /imu/data | head -20

# Check TF tree
ros2 run tf2_tools view_frames.py && evince frames.pdf
```

---

## References

- **ROS2 Sensor Drivers**: https://github.com/ros-drivers/
- **Common Sensors**:
  - **IMU**: Xsens, Vectornav, Bosch BMI270
  - **LiDAR**: Sick, Livox, Velodyne
  - **Encoders**: Hall sensors, optical encoders
  - **Camera**: USB cameras, RealSense, ZED
  - **Thermal**: FLIR, Seek
- **ROS2 Sensor Fusion**: http://docs.ros.org/en/humble/Concepts/About-ROS-2-Sensor-Fusion.html

