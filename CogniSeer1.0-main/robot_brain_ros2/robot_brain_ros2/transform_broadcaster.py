#!/usr/bin/env python3
"""
Transform Broadcaster for the CogniSeer Robot.

Publishes TF2 transforms for the robot's kinematic chain:
- base_link ↔ base_footprint
- base_link ↔ wheels (left/right)
- base_link ↔ sensors (IMU, cameras, distance sensors)

These transforms are typically fixed, but can be updated dynamically
if needed for simulation or calibration.
"""

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math


class RobotTransformBroadcaster(Node):
    """Publishes robot kinematic transforms."""

    def __init__(self):
        super().__init__("robot_tf_broadcaster")

        self.tf_broadcaster = TransformBroadcaster(self)

        # Declare parameters for transform offsets (for calibration)
        self.declare_parameter("imu.x_offset", 0.0)
        self.declare_parameter("imu.y_offset", 0.0)
        self.declare_parameter("imu.z_offset", 0.08)
        
        self.declare_parameter("camera.x_offset", 0.2)
        self.declare_parameter("camera.y_offset", 0.0)
        self.declare_parameter("camera.z_offset", 0.12)
        
        self.declare_parameter("distance_sensor_front.x_offset", 0.2)
        self.declare_parameter("distance_sensor_front.y_offset", 0.0)
        self.declare_parameter("distance_sensor_front.z_offset", 0.05)

        # Timer to publish transforms periodically (fixed transforms)
        self.timer = self.create_timer(0.05, self.broadcast_transforms)

        self.get_logger().info("Robot transform broadcaster started")

    def broadcast_transforms(self):
        """Broadcast all robot transforms."""
        now = self.get_clock().now()

        # 1. base_footprint ↔ base_link (usually fixed, height of robot center)
        self._broadcast_transform(
            now,
            parent_frame="base_footprint",
            child_frame="base_link",
            x=0.0, y=0.0, z=0.0,
            roll=0.0, pitch=0.0, yaw=0.0
        )

        # 2. base_link ↔ left_wheel_link
        self._broadcast_transform(
            now,
            parent_frame="base_link",
            child_frame="left_wheel_link",
            x=0.0, y=0.125, z=0.0,
            roll=0.0, pitch=0.0, yaw=0.0
        )

        # 3. base_link ↔ right_wheel_link
        self._broadcast_transform(
            now,
            parent_frame="base_link",
            child_frame="right_wheel_link",
            x=0.0, y=-0.125, z=0.0,
            roll=0.0, pitch=0.0, yaw=0.0
        )

        # 4. base_link ↔ imu_link
        imu_x = self.get_parameter("imu.x_offset").get_parameter_value().double_value
        imu_y = self.get_parameter("imu.y_offset").get_parameter_value().double_value
        imu_z = self.get_parameter("imu.z_offset").get_parameter_value().double_value
        self._broadcast_transform(
            now,
            parent_frame="base_link",
            child_frame="imu_link",
            x=imu_x, y=imu_y, z=imu_z,
            roll=0.0, pitch=0.0, yaw=0.0
        )

        # 5. base_link ↔ camera_link
        cam_x = self.get_parameter("camera.x_offset").get_parameter_value().double_value
        cam_y = self.get_parameter("camera.y_offset").get_parameter_value().double_value
        cam_z = self.get_parameter("camera.z_offset").get_parameter_value().double_value
        self._broadcast_transform(
            now,
            parent_frame="base_link",
            child_frame="camera_link",
            x=cam_x, y=cam_y, z=cam_z,
            roll=0.0, pitch=0.0, yaw=0.0
        )

        # 6. base_link ↔ distance_sensor_front_link
        ds_x = self.get_parameter("distance_sensor_front.x_offset").get_parameter_value().double_value
        ds_y = self.get_parameter("distance_sensor_front.y_offset").get_parameter_value().double_value
        ds_z = self.get_parameter("distance_sensor_front.z_offset").get_parameter_value().double_value
        self._broadcast_transform(
            now,
            parent_frame="base_link",
            child_frame="distance_sensor_front_link",
            x=ds_x, y=ds_y, z=ds_z,
            roll=0.0, pitch=0.0, yaw=0.0
        )

    def _broadcast_transform(
        self, stamp, parent_frame, child_frame, x, y, z, roll, pitch, yaw
    ):
        """Broadcast a single transform using RPY angles."""
        # Convert RPY to quaternion
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

        t = TransformStamped()
        t.header.stamp = stamp.to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame
        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = float(z)
        t.transform.rotation.x = float(qx)
        t.transform.rotation.y = float(qy)
        t.transform.rotation.z = float(qz)
        t.transform.rotation.w = float(qw)

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = RobotTransformBroadcaster()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
