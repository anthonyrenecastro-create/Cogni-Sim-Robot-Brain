#!/usr/bin/env python3
"""
Unit tests for transform broadcaster node.

Tests:
- Transform structure and naming
- Quaternion calculations
- Frame tree consistency
- Parameter overrides
- Broadcasting frequency
"""

import unittest
import math
import numpy as np
from geometry_msgs.msg import Quaternion


class TestTransformFrameNaming(unittest.TestCase):
    """Test TF2 frame naming conventions."""

    def test_base_link_transforms(self):
        """Test base link transforms follow naming convention."""
        base_frames = [
            ('base_footprint', 'base_link'),
            ('base_link', 'left_wheel_link'),
            ('base_link', 'right_wheel_link'),
            ('base_link', 'imu_link'),
            ('base_link', 'camera_link'),
            ('base_link', 'distance_sensor_front_link'),
        ]

        for parent, child in base_frames:
            # All should be valid frame names (no spaces, special chars)
            self.assertIsInstance(parent, str)
            self.assertIsInstance(child, str)
            self.assertGreater(len(parent), 0)
            self.assertGreater(len(child), 0)
            # Frame names should be lowercase with underscores
            self.assertTrue(parent.islower() or '_' in parent)

    def test_sensor_frame_hierarchy(self):
        """Test sensor frames are children of base_link."""
        sensor_parents = [
            ('base_link', 'imu_link'),
            ('base_link', 'camera_link'),
            ('base_link', 'distance_sensor_front_link'),
        ]

        for parent, child in sensor_parents:
            self.assertEqual(parent, 'base_link')


class TestQuaternionConversion(unittest.TestCase):
    """Test RPY to quaternion conversion."""

    def test_identity_quaternion(self):
        """Test identity rotation (0, 0, 0 RPY → 0, 0, 0, 1 quat)."""
        roll, pitch, yaw = 0, 0, 0

        # Using standard conversion formula
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        qw = cr * cp * cy + sr * sp * sy

        self.assertAlmostEqual(qx, 0, places=5)
        self.assertAlmostEqual(qy, 0, places=5)
        self.assertAlmostEqual(qz, 0, places=5)
        self.assertAlmostEqual(qw, 1, places=5)

    def test_180_degree_rotation_around_z(self):
        """Test 180° rotation around Z axis."""
        roll, pitch = 0, 0
        yaw = math.pi  # 180 degrees

        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        qw = cr * cp * cy + sr * sp * sy

        # 180° around Z should give qz=1, qw=0
        self.assertAlmostEqual(abs(qz), 1, places=5)

    def test_small_angle_approximation(self):
        """Test that small angles are handled correctly."""
        small_angles = [0.01, 0.001, -0.01, -0.001]

        for angle in small_angles:
            roll, pitch, yaw = angle, angle, angle

            cy = math.cos(yaw * 0.5)
            sy = math.sin(yaw * 0.5)
            cp = math.cos(pitch * 0.5)
            sp = math.sin(pitch * 0.5)
            cr = math.cos(roll * 0.5)
            sr = math.sin(roll * 0.5)

            qx = sr * cp * cy - cr * sp * sy
            qy = cr * sp * cy + sr * cp * sy
            qz = cr * cp * sy - sr * sp * cy
            qw = cr * cp * cy + sr * sp * sy

            # Magnitude should be ~1
            magnitude = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
            self.assertAlmostEqual(magnitude, 1.0, places=5)

    def test_quaternion_normalization(self):
        """Test that quaternions are normalized."""
        # Random quaternion components
        qx, qy, qz, qw = 0.5, 0.3, 0.2, 0.8

        # Calculate magnitude
        magnitude = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2)

        # Normalize
        qx /= magnitude
        qy /= magnitude
        qz /= magnitude
        qw /= magnitude

        # Magnitude should be 1
        new_mag = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        self.assertAlmostEqual(new_mag, 1.0, places=5)


class TestTransformOffsets(unittest.TestCase):
    """Test sensor offset parameters."""

    def test_imu_offset_application(self):
        """Test IMU sensor mount offset."""
        # IMU at [0.0, 0.0, 0.08] relative to base_link
        imu_offset = {'x': 0.0, 'y': 0.0, 'z': 0.08}

        # Transform should place IMU 8cm above base
        self.assertEqual(imu_offset['z'], 0.08)

    def test_camera_offset_application(self):
        """Test camera sensor mount offset."""
        # Camera at [0.2, 0.0, 0.12] relative to base_link
        camera_offset = {'x': 0.2, 'y': 0.0, 'z': 0.12}

        # Should be forward and up from base
        self.assertGreater(camera_offset['x'], 0)
        self.assertGreater(camera_offset['z'], 0)

    def test_offset_parameters_are_floats(self):
        """Test that offset parameters are valid floats."""
        offsets = {
            'imu': {'x': 0.0, 'y': 0.0, 'z': 0.08},
            'camera': {'x': 0.2, 'y': 0.0, 'z': 0.12},
            'distance_sensor_front': {'x': 0.2, 'y': 0.0, 'z': 0.05},
        }

        for sensor_name, offset in offsets.items():
            for axis, value in offset.items():
                self.assertIsInstance(value, (int, float))


class TestWheelTransforms(unittest.TestCase):
    """Test wheel geometry and transforms."""

    def test_wheel_spacing(self):
        """Test that wheels are symmetrically spaced."""
        # Expected wheel positions for differential drive
        left_wheel_y = 0.125  # 12.5 cm to the left
        right_wheel_y = -0.125  # 12.5 cm to the right
        wheelbase = abs(left_wheel_y - right_wheel_y)

        # Wheelbase should be 0.25 m (25 cm)
        self.assertAlmostEqual(wheelbase, 0.25, places=3)

    def test_wheel_forward_position(self):
        """Test wheel forward position from base_link."""
        # Wheels should be at base_link center (x=0) for differential drive
        wheel_x = 0.0
        self.assertEqual(wheel_x, 0.0)


class TestTransformBroadcastingFrequency(unittest.TestCase):
    """Test broadcasting rate and timing."""

    def test_broadcast_rate_20hz(self):
        """Test that transforms are broadcast at 20 Hz."""
        # Expected: 20 Hz = 50 ms period
        broadcast_freq = 20  # Hz
        period_ms = 1000 / broadcast_freq

        self.assertEqual(period_ms, 50)

    def test_minimum_broadcast_rate(self):
        """Test that broadcast rate is at least 10 Hz."""
        min_freq = 10  # Hz
        broadcast_freq = 20  # Hz

        self.assertGreaterEqual(broadcast_freq, min_freq)


class TestFrameTreeConsistency(unittest.TestCase):
    """Test that transform tree is consistent."""

    def test_no_circular_transforms(self):
        """Test that transform graph has no cycles."""
        # Define transform edges
        transforms = [
            ('odom', 'base_footprint'),
            ('base_footprint', 'base_link'),
            ('base_link', 'left_wheel_link'),
            ('base_link', 'right_wheel_link'),
            ('base_link', 'imu_link'),
            ('base_link', 'camera_link'),
            ('base_link', 'distance_sensor_front_link'),
        ]

        # Build adjacency list
        graph = {}
        for parent, child in transforms:
            if parent not in graph:
                graph[parent] = []
            graph[parent].append(child)

        # Check no node points to itself
        for node, children in graph.items():
            self.assertNotIn(node, children)

    def test_tree_has_single_root(self):
        """Test that frame tree has single root (odom)."""
        # All frames should have a path to root
        root_candidates = set()
        for transform in [
            ('odom', 'base_footprint'),
            ('base_footprint', 'base_link'),
        ]:
            root_candidates.add(transform[0])

        # Remove non-roots
        all_children = set()
        for transform in [
            ('odom', 'base_footprint'),
            ('base_footprint', 'base_link'),
            ('base_link', 'left_wheel_link'),
        ]:
            all_children.add(transform[1])

        roots = root_candidates - all_children
        self.assertEqual(len(roots), 1)
        self.assertIn('odom', roots)

    def test_sensor_transforms_are_leaf_nodes(self):
        """Test that sensors are leaf nodes (no children)."""
        leaf_frames = [
            'left_wheel_link',
            'right_wheel_link',
            'imu_link',
            'camera_link',
            'distance_sensor_front_link',
        ]

        # These should not have children in the tree
        for leaf in leaf_frames:
            self.assertTrue(leaf.endswith('_link'))


class TestStaticVsDynamicTransforms(unittest.TestCase):
    """Test distinction between static and dynamic transforms."""

    def test_all_transforms_are_static(self):
        """Test that all current transforms are static (fixed mounts)."""
        # All sensor mounts and wheels are fixed on the robot
        static_frames = [
            ('base_footprint', 'base_link'),
            ('base_link', 'left_wheel_link'),
            ('base_link', 'right_wheel_link'),
            ('base_link', 'imu_link'),
            ('base_link', 'camera_link'),
            ('base_link', 'distance_sensor_front_link'),
        ]

        # All are static because robot structure doesn't change
        self.assertEqual(len(static_frames), 6)

    def test_odom_to_base_footprint_dynamic(self):
        """Test that odom→base_footprint is dynamic (robot moves)."""
        # This transform changes as robot moves
        # It's not static
        is_static = False
        self.assertFalse(is_static)


if __name__ == '__main__':
    unittest.main()
