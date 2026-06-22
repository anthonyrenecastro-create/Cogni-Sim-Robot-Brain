#!/usr/bin/env python3
"""
Sensor Adapters — Standardized integration for common robot sensors.

Provides adapters to convert from various sensor driver outputs to the
standard observation vector format used by the robot brain.

Supported Sensors:
- IMU (Inertial Measurement Unit) — acceleration and angular velocity
- Distance/Proximity Sensors — ultrasonic, IR, lidar
- Encoders — wheel odometry
- Cameras — RGB/depth (compressed formats)
- Battery/System Monitors — voltage, temperature

Each adapter is a modular subscriber that integrates sensor data
into the shared observation vector published on /brain/sensors.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, Range, Image, CompressedImage, LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Float32MultiArray, MultiArrayDimension
from std_srvs.srv import Trigger
import cv2
from cv_bridge import CvBridge
from typing import Dict, Any


class SensorAggregatorNode(Node):
    """
    Aggregates data from multiple sensors into a unified observation vector.
    
    Subscribes to individual sensor topics and publishes an aggregated
    Float32MultiArray on /brain/sensors for consumption by the robot brain.
    """

    def __init__(self):
        super().__init__("sensor_aggregator")

        # Configuration
        self.declare_parameter("observation_dim", 32)
        self.declare_parameter("imu_enabled", True)
        self.declare_parameter("distance_enabled", True)
        self.declare_parameter("encoder_enabled", True)
        self.declare_parameter("battery_enabled", True)
        self.declare_parameter("camera_enabled", False)  # Large, optional

        self.obs_dim = self.get_parameter("observation_dim").get_parameter_value().integer_value
        self.observation = np.zeros(self.obs_dim, dtype=np.float32)

        # Sensor status tracking
        self.sensor_status: Dict[str, bool] = {}

        # Subscriptions
        if self.get_parameter("imu_enabled").get_parameter_value().bool_value:
            self.create_subscription(Imu, "/imu/data", self._imu_callback, 10)
            self.sensor_status["imu"] = False

        if self.get_parameter("distance_enabled").get_parameter_value().bool_value:
            self.create_subscription(Range, "/scan/front", self._distance_callback, 10)
            self.create_subscription(Range, "/scan/left", self._distance_callback, 10)
            self.create_subscription(Range, "/scan/right", self._distance_callback, 10)
            self.sensor_status["distance"] = False

        if self.get_parameter("encoder_enabled").get_parameter_value().bool_value:
            self.create_subscription(Odometry, "/odom", self._odom_callback, 10)
            self.sensor_status["encoder"] = False

        if self.get_parameter("battery_enabled").get_parameter_value().bool_value:
            self.create_subscription(Float32, "/system/battery_voltage", self._battery_callback, 10)
            self.create_subscription(Float32, "/system/cpu_temp", self._temp_callback, 10)
            self.sensor_status["battery"] = False

        if self.get_parameter("camera_enabled").get_parameter_value().bool_value:
            self.cv_bridge = CvBridge()
            self.create_subscription(CompressedImage, "/camera/rgb/compressed", self._camera_callback, 5)
            self.sensor_status["camera"] = False

        # Publisher
        self.obs_pub = self.create_publisher(Float32MultiArray, "/brain/sensors", 10)

        # Health status service
        self.create_service(Trigger, "/brain/sensor_health", self._health_callback)

        # Timer to publish aggregated observation
        self.timer = self.create_timer(0.02, self._publish_observation)  # 50 Hz

        self.get_logger().info(f"Sensor aggregator initialized (obs_dim={self.obs_dim})")

    def _imu_callback(self, msg: Imu):
        """Handle IMU data (accel + gyro)."""
        # Indices 0-2: acceleration (m/s²)
        self.observation[0] = msg.linear_acceleration.x
        self.observation[1] = msg.linear_acceleration.y
        self.observation[2] = msg.linear_acceleration.z

        # Indices 3-5: angular velocity (rad/s)
        self.observation[3] = msg.angular_velocity.x
        self.observation[4] = msg.angular_velocity.y
        self.observation[5] = msg.angular_velocity.z

        self.sensor_status["imu"] = True

    def _distance_callback(self, msg: Range):
        """Handle distance sensor data (front/left/right)."""
        # Map sensor frame to observation indices
        frame_to_idx = {
            "distance_sensor_front_link": 6,
            "distance_sensor_left_link": 7,
            "distance_sensor_right_link": 8,
        }

        if msg.header.frame_id in frame_to_idx:
            idx = frame_to_idx[msg.header.frame_id]
            # Normalize distance: 0 = min_range, 1 = max_range, clipped
            normalized = (msg.range - msg.min_range) / (msg.max_range - msg.min_range)
            self.observation[idx] = np.clip(normalized, 0.0, 1.0)

        self.sensor_status["distance"] = True

    def _odom_callback(self, msg: Odometry):
        """Handle odometry/encoder data (position and velocity)."""
        # Indices 9-11: position (x, y, theta)
        self.observation[9] = msg.pose.pose.position.x
        self.observation[10] = msg.pose.pose.position.y
        # Extract yaw from quaternion
        qx, qy, qz, qw = (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                           msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        self.observation[11] = yaw

        # Indices 12-13: linear and angular velocity
        self.observation[12] = msg.twist.twist.linear.x
        self.observation[13] = msg.twist.twist.angular.z

        self.sensor_status["encoder"] = True

    def _battery_callback(self, msg: Float32):
        """Handle battery voltage."""
        # Index 14: battery voltage (normalized: 0-1)
        # Typical range: 10.5V - 12.6V (3S LiPo)
        normalized = (msg.data - 10.5) / (12.6 - 10.5)
        self.observation[14] = np.clip(normalized, 0.0, 1.0)
        self.sensor_status["battery"] = True

    def _temp_callback(self, msg: Float32):
        """Handle system temperature."""
        # Index 15: CPU temperature (normalized: 0-100°C)
        normalized = msg.data / 100.0
        self.observation[15] = np.clip(normalized, 0.0, 1.0)

    def _camera_callback(self, msg: CompressedImage):
        """Handle camera data (compressed to reduce bandwidth)."""
        try:
            # Decompress image
            cv_image = self.cv_bridge.compressed_imgmsg_to_cv2(msg)
            
            # Resize to small representation (e.g., 4x4 = 16 values)
            small = cv2.resize(cv_image, (4, 4))
            
            # Convert to grayscale and normalize
            if len(small.shape) == 3:
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            else:
                gray = small
            
            # Flatten and normalize to 0-1
            flat = gray.flatten().astype(np.float32) / 255.0
            
            # Store in observation (indices 16-31, assuming obs_dim >= 32)
            if self.obs_dim >= 32:
                self.observation[16:32] = flat[:16]
            
            self.sensor_status["camera"] = True
        except Exception as e:
            self.get_logger().warning(f"Camera callback error: {e}")

    def _publish_observation(self):
        """Publish current observation vector."""
        msg = Float32MultiArray()
        msg.layout.dim = [MultiArrayDimension(label="observation", size=self.obs_dim, stride=self.obs_dim)]
        msg.data = self.observation.tolist()
        self.obs_pub.publish(msg)

    def _health_callback(self, request, response):
        """Report sensor health status."""
        all_ok = all(self.sensor_status.values())
        response.success = all_ok
        response.message = f"Sensor status: {self.sensor_status}"
        return response


def main(args=None):
    rclpy.init(args=args)
    node = SensorAggregatorNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
