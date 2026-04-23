import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    pkg = get_package_share_directory('robot_brain_stack')
    nav2_params = os.path.join(pkg, 'config', 'nav2_params.yaml')

    nav2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource('/opt/ros/humble/share/nav2_bringup/launch/navigation_launch.py'),
        launch_arguments={
            'params_file': nav2_params,
            'use_sim_time': 'true',
        }.items(),
    )

    return LaunchDescription([nav2_bringup])
