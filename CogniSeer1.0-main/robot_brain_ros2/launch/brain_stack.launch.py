#!/usr/bin/env python3
"""
Main launch file for CogniSeer Robot Brain Stack.

Launches:
1. Transform broadcaster (publishes robot TF tree)
2. Sensor aggregator (combines sensor inputs)
3. Robot brain node (runs the SNN inference)
4. Safety layer (monitors and bounds outputs)

Usage:
    ros2 launch robot_brain_ros2 brain_stack.launch.py
    ros2 launch robot_brain_ros2 brain_stack.launch.py with_rviz:=true
    ros2 launch robot_brain_ros2 brain_stack.launch.py robot_description:=custom.urdf
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get package directories
    pkg_dir = get_package_share_directory("robot_brain_ros2")
    config_dir = os.path.join(pkg_dir, "config")

    # Declare launch arguments
    return LaunchDescription([
        # Arguments
        DeclareLaunchArgument(
            "model_path",
            default_value=os.path.join(pkg_dir, "..", "..", "deploy", "robot_brain_int8_stateful.pt"),
            description="Path to the quantized robot brain model",
        ),
        DeclareLaunchArgument(
            "config_file",
            default_value=os.path.join(config_dir, "defaults.json"),
            description="Configuration file for brain parameters",
        ),
        DeclareLaunchArgument(
            "robot_description",
            default_value=os.path.join(pkg_dir, "urdf", "cogniseer_robot.urdf.xacro"),
            description="Path to robot URDF/XACRO file",
        ),
        DeclareLaunchArgument(
            "with_rviz",
            default_value="false",
            description="Whether to launch RViz for visualization",
        ),
        DeclareLaunchArgument(
            "with_sensor_aggregator",
            default_value="true",
            description="Whether to launch sensor aggregator",
        ),
        DeclareLaunchArgument(
            "with_transform_broadcaster",
            default_value="true",
            description="Whether to launch transform broadcaster",
        ),
        DeclareLaunchArgument(
            "namespace",
            default_value="",
            description="Robot namespace",
        ),

        # Transform broadcaster (publishes robot's TF tree)
        Node(
            condition=IncludeLaunchDescription.IfCondition(LaunchConfiguration("with_transform_broadcaster")),
            package="robot_brain_ros2",
            executable="transform_broadcaster.py",
            name="tf_broadcaster",
            namespace=LaunchConfiguration("namespace"),
            parameters=[
                {"imu.x_offset": 0.0},
                {"imu.y_offset": 0.0},
                {"imu.z_offset": 0.08},
                {"camera.x_offset": 0.2},
                {"camera.y_offset": 0.0},
                {"camera.z_offset": 0.12},
                {"distance_sensor_front.x_offset": 0.2},
                {"distance_sensor_front.y_offset": 0.0},
                {"distance_sensor_front.z_offset": 0.05},
            ],
            output="screen",
        ),

        # Sensor aggregator (combines all sensor inputs)
        Node(
            condition=IncludeLaunchDescription.IfCondition(LaunchConfiguration("with_sensor_aggregator")),
            package="robot_brain_ros2",
            executable="sensor_aggregator.py",
            name="sensor_aggregator",
            namespace=LaunchConfiguration("namespace"),
            parameters=[
                {"observation_dim": 32},
                {"imu_enabled": True},
                {"distance_enabled": True},
                {"encoder_enabled": True},
                {"battery_enabled": True},
                {"camera_enabled": False},  # Disabled by default (bandwidth)
            ],
            output="screen",
        ),

        # Robot brain node (inference + control)
        Node(
            package="robot_brain_ros2",
            executable="robot_brain_node.py",
            name="robot_brain_node",
            namespace=LaunchConfiguration("namespace"),
            parameters=[
                {"model_path": LaunchConfiguration("model_path")},
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
            ],
            output="screen",
        ),
    ])
