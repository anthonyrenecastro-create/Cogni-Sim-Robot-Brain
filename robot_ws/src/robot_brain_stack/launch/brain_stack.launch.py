from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg = get_package_share_directory('robot_brain_stack')
    params = os.path.join(pkg, 'config', 'brain_params.yaml')

    return LaunchDescription([
        Node(
            package='robot_brain_stack',
            executable='goal_manager',
            name='goal_manager',
            parameters=[params],
        ),
        Node(
            package='robot_brain_stack',
            executable='object_memory',
            name='object_memory',
            parameters=[params],
        ),
        Node(
            package='robot_brain_stack',
            executable='recovery_manager',
            name='recovery_manager',
            parameters=[params],
        ),
        Node(
            package='robot_brain_stack',
            executable='safety_layer',
            name='safety_layer',
            parameters=[params],
        ),
        Node(
            package='robot_brain_stack',
            executable='executive_context',
            name='executive_context',
            parameters=[params],
        ),
        Node(
            package='robot_brain_stack',
            executable='bt_executor',
            name='bt_executor',
            parameters=[params],
        ),
        Node(
            package='robot_brain_stack',
            executable='health_monitor',
            name='health_monitor',
            parameters=[params],
        ),
        Node(
            package='robot_brain_stack',
            executable='watchdog',
            name='watchdog',
            parameters=[params],
        ),
    ])