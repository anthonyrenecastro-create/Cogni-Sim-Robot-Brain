from setuptools import setup

package_name = "robot_brain_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["../config/defaults.json"]),
        ("share/" + package_name + "/urdf", ["../urdf/cogniseer_robot.urdf.xacro"]),
        ("share/" + package_name + "/launch", ["../launch/brain_stack.launch.py"]),
    ],
    # NOTE: ROS2 dependencies (rclpy, std_msgs, geometry_msgs) are NOT listed here.
    # ROS2 cannot be installed via pip—it must be installed system-wide.
    # This package assumes ROS2 is already installed in the environment.
    # If ROS2 is not available, imports will fail with a clear error message.
    # See docs/ROS2_SETUP.md for installation instructions.
    install_requires=["setuptools", "numpy>=1.26.0"],
    zip_safe=True,
    maintainer="anthonyrenecastro-create",
    maintainer_email="anthonyrenecastro-create@users.noreply.github.com",
    description="CogniSeer edge robot brain — ROS2 node publishing safe velocity commands from an INT8-quantized PyTorch brain model.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "robot_brain_node.py = robot_brain_ros2.ros2_robot_brain_node:main",
            "sensor_aggregator.py = robot_brain_ros2.sensor_aggregator:main",
            "transform_broadcaster.py = robot_brain_ros2.transform_broadcaster:main",
        ],
    },
)
