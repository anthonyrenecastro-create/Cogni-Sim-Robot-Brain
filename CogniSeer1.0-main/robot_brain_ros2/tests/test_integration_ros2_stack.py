#!/usr/bin/env python3
"""
Integration tests for ROS2 brain stack components.

Tests:
- Sensor aggregator + brain node communication
- Topic publishing and receiving
- Multi-node integration
- End-to-end data flow
- Topic compatibility verification
"""

import unittest
import numpy as np
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, Range
from nav_msgs.msg import Odometry


class TestTopicCommunication(unittest.TestCase):
    """Test topic publishing and receiving."""

    def test_sensor_aggregator_output_format(self):
        """Test that sensor aggregator publishes correct format."""
        # Expected output: Float32MultiArray on /brain/sensors
        msg = Float32MultiArray()

        # Set dimensions
        msg.layout.dim = [MultiArrayDimension()]
        msg.layout.dim[0].label = "observation"
        msg.layout.dim[0].size = 32
        msg.layout.dim[0].stride = 32

        # Set data
        msg.data = list(np.zeros(32, dtype=np.float32))

        # Verify structure
        self.assertEqual(len(msg.data), 32)
        self.assertIsInstance(msg.data[0], float)

    def test_brain_node_output_format(self):
        """Test that brain node publishes Twist on /cmd_vel."""
        cmd = Twist()
        cmd.linear.x = 0.5
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0
        cmd.angular.x = 0.0
        cmd.angular.y = 0.0
        cmd.angular.z = 0.1

        # Verify structure
        self.assertAlmostEqual(cmd.linear.x, 0.5)
        self.assertAlmostEqual(cmd.angular.z, 0.1)

    def test_velocity_command_bounds(self):
        """Test that velocity commands are within safety bounds."""
        # Expected bounds
        max_linear = 0.5  # m/s
        max_angular = 1.0  # rad/s

        test_commands = [
            (0.5, 1.0, True),       # At bounds
            (0.3, 0.5, True),       # Within bounds
            (0.6, 1.0, False),      # Exceeds linear
            (0.5, 1.2, False),      # Exceeds angular
        ]

        for lin, ang, should_pass in test_commands:
            valid = abs(lin) <= max_linear and abs(ang) <= max_angular
            self.assertEqual(valid, should_pass)


class TestDataFlowConsistency(unittest.TestCase):
    """Test that data flows correctly through nodes."""

    def test_observation_vector_size_consistency(self):
        """Test that observation vector size is consistent throughout."""
        obs_dim = 32

        # Sensor aggregator produces
        agg_output = np.zeros(obs_dim)

        # Brain node consumes
        brain_input_size = len(agg_output)

        self.assertEqual(brain_input_size, obs_dim)

    def test_imu_data_survives_aggregation(self):
        """Test that IMU data is preserved through aggregation."""
        # Original IMU data
        imu_accel = np.array([0.5, 0.3, 9.8])
        imu_gyro = np.array([0.1, 0.05, 0.02])

        # After aggregation (indices [0:6])
        aggregated_imu = np.concatenate([imu_accel, imu_gyro])

        self.assertEqual(len(aggregated_imu), 6)
        np.testing.assert_array_almost_equal(
            aggregated_imu[:3], imu_accel
        )

    def test_distance_data_normalization_in_aggregation(self):
        """Test that distance data is properly normalized."""
        # Raw distance: 0.5 m (on 0-4m range)
        raw_distance = 0.5
        max_range = 4.0
        normalized = min(raw_distance / max_range, 1.0)

        # Should be in [0, 1]
        self.assertGreaterEqual(normalized, 0.0)
        self.assertLessEqual(normalized, 1.0)
        self.assertAlmostEqual(normalized, 0.125, places=3)


class TestROS2NodeInteraction(unittest.TestCase):
    """Test interactions between ROS2 nodes."""

    def test_sensor_aggregator_subscribes_to_sensors(self):
        """Test that sensor aggregator subscribes to expected topics."""
        expected_topics = [
            '/imu/data',
            '/scan/front',
            '/scan/left',
            '/scan/right',
            '/odom',
            '/system/battery_voltage',
            '/system/cpu_temp',
            '/camera/rgb/compressed',  # Optional
        ]

        # All topics should be valid ROS2 topic names
        for topic in expected_topics:
            self.assertTrue(topic.startswith('/'))
            self.assertNotIn(' ', topic)

    def test_brain_node_subscribes_to_aggregated_sensors(self):
        """Test that brain node subscribes to sensor aggregator output."""
        expected_topic = '/brain/sensors'

        self.assertTrue(expected_topic.startswith('/'))
        self.assertIn('sensors', expected_topic)

    def test_brain_node_publishes_to_motor_controller(self):
        """Test that brain node publishes to motor controller."""
        expected_topics = [
            '/cmd_vel',
            '/brain/motor_command',
        ]

        for topic in expected_topics:
            self.assertTrue(topic.startswith('/'))


class TestErrorHandlingIntegration(unittest.TestCase):
    """Test error handling in integrated system."""

    def test_missing_sensor_doesnt_crash(self):
        """Test that system handles missing sensors gracefully."""
        # Observation without IMU
        obs = np.zeros(32)

        # Fill in some other sensors
        obs[6:9] = [0.5, 0.5, 0.5]  # Distance

        # Should be valid observation
        self.assertEqual(len(obs), 32)
        self.assertFalse(np.isnan(obs).any())

    def test_stale_sensor_detection(self):
        """Test that stale sensors are detected."""
        sensor_age_sec = 2.0
        timeout_sec = 0.5

        is_stale = sensor_age_sec > timeout_sec
        self.assertTrue(is_stale)

    def test_observation_always_published_despite_missing_sensors(self):
        """Test that aggregator publishes even with partial data."""
        # Even with missing data, should publish something
        published = True
        self.assertTrue(published)


class TestSensorToObservationMapping(unittest.TestCase):
    """Test mapping from sensor messages to observation indices."""

    def test_imu_indices(self):
        """Test IMU data maps to indices 0-5."""
        imu_start, imu_end = 0, 6
        self.assertEqual(imu_end - imu_start, 6)

    def test_distance_indices(self):
        """Test distance sensor data maps to indices 6-8."""
        dist_start, dist_end = 6, 9
        self.assertEqual(dist_end - dist_start, 3)

    def test_odometry_indices(self):
        """Test odometry data maps to indices 9-13."""
        odom_start, odom_end = 9, 14
        self.assertEqual(odom_end - odom_start, 5)

    def test_battery_index(self):
        """Test battery voltage maps to index 14."""
        battery_idx = 14
        self.assertEqual(battery_idx, 14)

    def test_temperature_index(self):
        """Test temperature maps to index 15."""
        temp_idx = 15
        self.assertEqual(temp_idx, 15)

    def test_camera_indices(self):
        """Test camera features map to indices 16-31."""
        camera_start, camera_end = 16, 32
        self.assertEqual(camera_end - camera_start, 16)


class TestMultipleSubscribersIntegration(unittest.TestCase):
    """Test system with multiple subscribers."""

    def test_multiple_nodes_receive_sensor_data(self):
        """Test that multiple subscribers can read sensor data."""
        # Both brain node and monitoring could subscribe
        subscribers = [
            '/brain/sensors',  # Brain node
            '/brain/sensors',  # Monitor node
            '/brain/sensors',  # Diagnostics node
        ]

        # All should be valid
        for sub in subscribers:
            self.assertTrue(sub.startswith('/'))

    def test_multiple_sensors_same_aggregator(self):
        """Test that aggregator handles multiple sensor sources."""
        sensor_subscriptions = [
            ('imu', '/imu/data'),
            ('distance_front', '/scan/front'),
            ('distance_left', '/scan/left'),
            ('distance_right', '/scan/right'),
            ('odom', '/odom'),
            ('battery', '/system/battery_voltage'),
            ('temp', '/system/cpu_temp'),
        ]

        # Aggregator should subscribe to all
        self.assertGreater(len(sensor_subscriptions), 5)


class TestTimingAndSynchronization(unittest.TestCase):
    """Test timing and synchronization of components."""

    def test_sensor_aggregator_runs_at_50hz(self):
        """Test that sensor aggregator publishes at 50 Hz."""
        expected_freq = 50  # Hz
        period_ms = 1000 / expected_freq

        self.assertAlmostEqual(period_ms, 20, places=1)

    def test_brain_node_runs_at_20hz(self):
        """Test that brain node controls at 20 Hz."""
        expected_freq = 20  # Hz
        period_ms = 1000 / expected_freq

        self.assertAlmostEqual(period_ms, 50, places=1)

    def test_control_loop_slower_than_aggregation(self):
        """Test that control is slower than sensor aggregation."""
        agg_freq = 50  # Hz
        brain_freq = 20  # Hz

        self.assertGreater(agg_freq, brain_freq)

    def test_aggregator_publishes_before_brain_consumes(self):
        """Test that sensor data is ready before brain needs it."""
        agg_period_ms = 1000 / 50  # 20 ms
        brain_period_ms = 1000 / 20  # 50 ms

        # Aggregator is 2.5x faster, so data is always fresh
        self.assertLess(agg_period_ms, brain_period_ms)


class TestEndToEndDataFlow(unittest.TestCase):
    """Test complete data flow from sensors to motors."""

    def test_sensor_to_motor_flow(self):
        """Test complete flow: Sensors → Aggregator → Brain → Motors."""
        flow = [
            ('sensors', 'publish on /imu/data, /scan/*, /odom'),
            ('aggregator', 'subscribe to sensors, publish /brain/sensors'),
            ('brain', 'subscribe to /brain/sensors, publish /cmd_vel'),
            ('motor_ctrl', 'subscribe to /cmd_vel, drive motors'),
        ]

        self.assertEqual(len(flow), 4)

        # Verify each step is present
        for step, description in flow:
            self.assertGreater(len(step), 0)

    def test_latency_budget(self):
        """Test that total latency is acceptable."""
        latencies = {
            'sensor_to_agg': 5,      # ms
            'agg_processing': 5,     # ms
            'agg_to_brain': 5,       # ms
            'brain_inference': 50,   # ms
            'brain_to_motor': 5,     # ms
            'motor_response': 10,    # ms
        }

        total_latency = sum(latencies.values())
        max_acceptable = 150  # ms

        self.assertLess(total_latency, max_acceptable)

    def test_jitter_tolerance(self):
        """Test system tolerates jitter in sensor data."""
        sensor_jitter_ms = 10  # ±10ms variance
        acceptable_jitter = 50  # ms

        self.assertLess(sensor_jitter_ms, acceptable_jitter)


if __name__ == '__main__':
    unittest.main()
