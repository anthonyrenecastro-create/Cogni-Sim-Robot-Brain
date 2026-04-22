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
    ],
    install_requires=["setuptools", "numpy>=1.26.0"],
    zip_safe=True,
    maintainer="anthonyrenecastro-create",
    maintainer_email="anthonyrenecastro-create@users.noreply.github.com",
    description="CogniSeer edge robot brain — ROS2 node publishing safe velocity commands from an INT8-quantized PyTorch brain model.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "robot_brain_node = robot_brain_ros2.ros2_robot_brain_node:main",
        ],
    },
)
