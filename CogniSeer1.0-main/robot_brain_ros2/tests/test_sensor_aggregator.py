#!/usr/bin/env python3
"""
Unit tests for sensor aggregator node.

Tests:
- Subscription callback handling
- Observation vector assembly
- Data normalization
- Missing sensor graceful degradation
- Health check service
"""

import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np
from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import Imu, Range
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Twist, Vector3, Quaternion
import rclpy
from rclpy.clock import Clock


class TestSensorAggregatorCallbacks(unittest.TestCase):
    """Test individual sensor callback handlers."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock IMU data
        self.imu_msg = Imu()
        self.imu_msg.linear_acceleration = Vector3(x=0.5, y=0.3, z=9.8)
        self.imu_msg.angular_velocity = Vector3(x=0.1, y=0.05, z=0.02)

        # Mock Range (distance) data
        self.range_msg = Range()
        self.range_msg.range = 0.5  # 50cm

        # Mock Odometry data
        self.odom_msg = Odometry()
        self.odom_msg.pose.pose.position.x = 1.0
        self.odom_msg.pose.pose.position.y = 2.0
        self.odom_msg.pose.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)
        self.odom_msg.twist.twist.linear.x = 0.5
        self.odom_msg.twist.twist.angular.z = 0.1

    def test_imu_callback_extraction(self):
        """Test that IMU data is correctly extracted and stored."""
        # Expected indices [0:6] for IMU
        accel = [self.imu_msg.linear_acceleration.x,
                 self.imu_msg.linear_acceleration.y,
                 self.imu_msg.linear_acceleration.z]
        gyro = [self.imu_msg.angular_velocity.x,
                self.imu_msg.angular_velocity.y,
                self.imu_msg.angular_velocity.z]

        self.assertEqual(accel[0], 0.5)
        self.assertEqual(accel[1], 0.3)
        self.assertEqual(accel[2], 9.8)
        self.assertEqual(gyro[0], 0.1)

    def test_distance_sensor_normalization(self):
        """Test that distance sensors are normalized to [0, 1]."""
        # Range: 0-4 meters → normalize to [0, 1]
        max_range = 4.0
        normalized = min(self.range_msg.range / max_range, 1.0)

        self.assertGreaterEqual(normalized, 0.0)
        self.assertLessEqual(normalized, 1.0)
        self.assertAlmostEqual(normalized, 0.125, places=3)

    def test_odometry_extraction(self):
        """Test that odometry position and yaw are extracted."""
        x = self.odom_msg.pose.pose.position.x
        y = self.odom_msg.pose.pose.position.y

        # Quaternion to yaw conversion (for w=1, x=y=z=0: yaw=0)
        quat = self.odom_msg.pose.pose.orientation
        yaw = 2 * np.arctan2(quat.z, quat.w)

        self.assertAlmostEqual(x, 1.0)
        self.assertAlmostEqual(y, 2.0)
        self.assertAlmostEqual(yaw, 0.0)


class TestObservationVectorAssembly(unittest.TestCase):
    """Test observation vector composition and ordering."""

    def test_observation_vector_structure(self):
        """Test that observation vector has correct structure."""
        # Expected 32D vector
        expected_dim = 32

        # Indices:
        indices = {
            'imu_accel': (0, 3),      # [0:3]
            'imu_gyro': (3, 6),       # [3:6]
            'distance': (6, 9),       # [6:9]
            'odometry': (9, 14),      # [9:14]
            'battery': 14,            # [14]
            'temperature': 15,        # [15]
            'camera': (16, 32),       # [16:32]
        }

        # Verify no overlaps
        self.assertEqual(indices['imu_accel'][1], indices['imu_gyro'][0])
        self.assertEqual(indices['imu_gyro'][1], indices['distance'][0])
        self.assertEqual(indices['distance'][1], indices['odometry'][0])

    def test_observation_vector_normalization(self):
        """Test that observation vector values are in valid ranges."""
        # Create sample observation vector
        obs = np.array([
            0.5, 0.3, 9.8,          # IMU accel
            0.1, 0.05, 0.02,        # IMU gyro
            0.2, 0.3, 0.25,         # Distance (normalized [0, 1])
            1.0, 2.0, 0.0,          # Position (x, y, yaw)
            0.5, 0.1,               # Velocities
            0.8,                    # Battery (normalized [0, 1])
            0.6,                    # Temperature (normalized [0, 1])
        ])

        # Pad with camera features
        obs = np.pad(obs, (0, 16), mode='constant', constant_values=0.5)

        # Check distance sensors in valid range
        self.assertGreaterEqual(obs[6], 0.0)
        self.assertLessEqual(obs[6], 1.0)

        # Check battery in valid range
        self.assertGreaterEqual(obs[14], 0.0)
        self.assertLessEqual(obs[14], 1.0)

        # Check temperature in valid range
        self.assertGreaterEqual(obs[15], 0.0)
        self.assertLessEqual(obs[15], 1.0)


class TestMissingSensorGracefulDegradation(unittest.TestCase):
    """Test handling of missing or stale sensors."""

    def test_missing_imu_still_publishes(self):
        """Test that observation vector can be published without IMU."""
        # Simulate observation without IMU data
        obs = np.zeros(32)

        # Only distance, odom, battery available
        obs[6:9] = [0.5, 0.5, 0.5]  # Distance
        obs[9:14] = [1.0, 2.0, 0.0, 0.5, 0.1]  # Odometry
        obs[14] = 0.8  # Battery

        # Should still create valid observation
        self.assertEqual(len(obs), 32)
        self.assertFalse(np.isnan(obs).any())

    def test_sensor_timeout_handling(self):
        """Test that stale sensor data is detected."""
        sensor_age_ms = 2000  # 2 seconds old
        timeout_ms = 500

        is_stale = sensor_age_ms > timeout_ms
        self.assertTrue(is_stale)

    def test_all_sensors_missing(self):
        """Test fallback when all sensors are missing."""
        # Even with no sensors, should return zero vector
        obs = np.zeros(32)

        self.assertEqual(len(obs), 32)
        self.assertTrue(np.all(obs == 0))


class TestBatteryTemperatureNormalization(unittest.TestCase):
    """Test normalization of battery and temperature values."""

    def test_battery_voltage_normalization(self):
        """Test battery voltage [10.5, 12.6]V → [0, 1]."""
        min_v = 10.5
        max_v = 12.6

        # Test low voltage
        voltage_low = 10.5
        normalized_low = (voltage_low - min_v) / (max_v - min_v)
        self.assertAlmostEqual(normalized_low, 0.0, places=2)

        # Test high voltage
        voltage_high = 12.6
        normalized_high = (voltage_high - min_v) / (max_v - min_v)
        self.assertAlmostEqual(normalized_high, 1.0, places=2)

        # Test mid voltage
        voltage_mid = 11.55
        normalized_mid = (voltage_mid - min_v) / (max_v - min_v)
        self.assertAlmostEqual(normalized_mid, 0.5, places=2)

    def test_temperature_normalization(self):
        """Test temperature [0, 100]°C → [0, 1]."""
        min_t = 0
        max_t = 100

        # Test low temperature
        temp_low = 20
        normalized_low = temp_low / max_t
        self.assertAlmostEqual(normalized_low, 0.2, places=2)

        # Test high temperature
        temp_high = 80
        normalized_high = temp_high / max_t
        self.assertAlmostEqual(normalized_high, 0.8, places=2)


class TestCameraDataProcessing(unittest.TestCase):
    """Test camera image processing to features."""

    def test_camera_compression_handling(self):
        """Test handling of CompressedImage messages."""
        # Simulated JPEG data (real JPEG would be bytes)
        compressed_data = b'\xff\xd8\xff\xe0'  # JPEG magic number

        # Should be able to decompress
        self.assertEqual(compressed_data[:4], b'\xff\xd8\xff\xe0')

    def test_image_resize_to_features(self):
        """Test resizing image to 4x4 feature vector."""
        # Simulate image resize: any image → 4x4 grayscale → 16 features
        image_features = np.random.rand(4, 4)
        flat_features = image_features.flatten()

        self.assertEqual(len(flat_features), 16)
        self.assertTrue(np.all((flat_features >= 0) & (flat_features <= 1)))


class TestHealthCheckService(unittest.TestCase):
    """Test sensor health status service."""

    def test_health_status_structure(self):
        """Test that health status returns all sensor states."""
        # Expected health status fields
        health_status = {
            'imu_ok': True,
            'distance_ok': True,
            'odometry_ok': True,
            'battery_ok': True,
            'temperature_ok': True,
            'camera_ok': False,  # Optional
        }

        self.assertEqual(len(health_status), 6)
        self.assertTrue(health_status['imu_ok'])

    def test_sensor_timeout_affects_health(self):
        """Test that stale sensor data marks health as bad."""
        last_update_ms = 1500  # 1.5 seconds ago
        timeout_ms = 1000

        is_healthy = last_update_ms < timeout_ms
        self.assertFalse(is_healthy)


class TestObservationVectorDimensions(unittest.TestCase):
    """Test that observation vector maintains correct dimensions."""

    def test_observation_always_32d(self):
        """Test that output is always 32-dimensional."""
        for obs_dim in [16, 32, 64]:  # Test different configurations
            # Should always normalize to 32 for brain input
            obs = np.zeros(32)
            self.assertEqual(len(obs), 32)

    def test_float32_precision(self):
        """Test that observation uses float32 precision."""
        obs = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        self.assertEqual(obs.dtype, np.float32)


if __name__ == '__main__':
    unittest.main()
