import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('robot_brain_stack')
    default_world = os.path.join(pkg, 'worlds', 'brain_stack_world.world')

    world_arg = DeclareLaunchArgument('world', default_value=default_world)

    gazebo = Node(
        package='gazebo_ros',
        executable='gazebo',
        name='gazebo',
        output='screen',
        arguments=['-s', 'libgazebo_ros_factory.so', LaunchConfiguration('world')],
    )

    return LaunchDescription([world_arg, gazebo])
