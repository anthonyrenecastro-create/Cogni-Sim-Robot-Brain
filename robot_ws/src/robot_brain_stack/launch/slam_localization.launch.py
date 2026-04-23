from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource('/opt/ros/humble/share/slam_toolbox/launch/online_async_launch.py'),
        launch_arguments={'use_sim_time': 'true'}.items(),
    )
    return LaunchDescription([slam])
