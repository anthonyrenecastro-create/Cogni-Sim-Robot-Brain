import os
from glob import glob

from setuptools import find_packages, setup


package_name = "robot_brain_stack"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "worlds"), glob("worlds/*.world")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="anthonyrenecastro-create",
    maintainer_email="anthonyrenecastro-create@users.noreply.github.com",
    description="Local robot brain stack with goal management, memory, recovery, BT, safety, and executive context.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "goal_manager = robot_brain_stack.goal_manager:main",
            "object_memory = robot_brain_stack.object_memory:main",
            "recovery_manager = robot_brain_stack.recovery_manager:main",
            "safety_layer = robot_brain_stack.safety_layer:main",
            "executive_context = robot_brain_stack.executive_context:main",
            "bt_executor = robot_brain_stack.bt_executor:main",
            "health_monitor = robot_brain_stack.health_monitor:main",
            "watchdog = robot_brain_stack.watchdog:main",
        ],
    },
)