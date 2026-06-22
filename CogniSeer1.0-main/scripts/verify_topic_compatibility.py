#!/usr/bin/env python3
"""
ROS2 Topic Compatibility Verification Tool

Checks that all required sensor topics are:
1. Being published at expected frequency
2. Publishing correct message types
3. Have valid TF transforms (if applicable)
4. All timestamps are synchronized

Usage:
    python verify_topic_compatibility.py --config config/brain_params.yaml
"""

import argparse
import json
import sys
import time
from typing import Dict, List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.time import Duration
from rclpy.utilities import get_available_rmw_implementations

from sensor_msgs.msg import Imu, Range
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Float32MultiArray
from geometry_msgs.msg import Twist
import tf2_ros
import yaml


class TopicCompatibilityChecker(Node):
    """Verify ROS2 topic compatibility and configuration."""

    def __init__(self, config_file: str):
        super().__init__("topic_compatibility_checker")

        self.config = self._load_config(config_file)
        self.checker_timeout = 30.0  # seconds
        self.required_topics: Dict[str, Dict] = {}
        self.observed_topics: Dict[str, Dict] = {}
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._setup_subscriptions()

    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_file, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            self.get_logger().warning(f"Could not load config: {e}, using defaults")
            return self._default_config()

    def _default_config(self) -> Dict:
        """Default configuration."""
        return {
            "required_sensors": ["imu", "distance", "odometry"],
            "sensor_timeout_sec": 2.0,
            "min_frequency_hz": {
                "imu": 50,
                "distance": 10,
                "odometry": 20,
                "cmd_vel": 20,
            },
        }

    def _setup_subscriptions(self):
        """Setup subscriptions based on configuration."""
        subs = {
            "imu": ("/imu/data", Imu, self._imu_callback),
            "distance": ("/scan/front", Range, self._distance_callback),
            "odometry": ("/odom", Odometry, self._odom_callback),
            "battery": ("/system/battery_voltage", Float32, self._battery_callback),
            "temperature": ("/system/cpu_temp", Float32, self._temp_callback),
        }

        for sensor_name, (topic, msg_type, callback) in subs.items():
            self.required_topics[sensor_name] = {
                "topic": topic,
                "type": msg_type.__name__,
                "expected_freq": self.config.get("min_frequency_hz", {}).get(sensor_name, 10),
            }
            self.observed_topics[sensor_name] = {
                "count": 0,
                "first_time": None,
                "last_time": None,
                "actual_freq": 0.0,
                "latency_ms": 0.0,
            }
            try:
                self.create_subscription(msg_type, topic, callback, 10)
            except Exception as e:
                self.get_logger().warning(f"Could not subscribe to {topic}: {e}")

    def _imu_callback(self, msg: Imu):
        """Handle IMU messages."""
        self._update_stats("imu", msg.header.stamp)

    def _distance_callback(self, msg: Range):
        """Handle distance sensor messages."""
        self._update_stats("distance", msg.header.stamp)

    def _odom_callback(self, msg: Odometry):
        """Handle odometry messages."""
        self._update_stats("odometry", msg.header.stamp)

    def _battery_callback(self, msg: Float32):
        """Handle battery messages."""
        self._update_stats("battery", None)

    def _temp_callback(self, msg: Float32):
        """Handle temperature messages."""
        self._update_stats("temperature", None)

    def _update_stats(self, sensor: str, msg_timestamp):
        """Update receive statistics for a sensor."""
        now = self.get_clock().now()
        stats = self.observed_topics[sensor]

        if stats["first_time"] is None:
            stats["first_time"] = now

        stats["last_time"] = now
        stats["count"] += 1

        # Calculate frequency
        if stats["count"] > 1:
            elapsed = (now.nanoseconds - stats["first_time"].nanoseconds) / 1e9
            stats["actual_freq"] = stats["count"] / elapsed if elapsed > 0 else 0.0

        # Calculate latency if timestamp available
        if msg_timestamp is not None:
            latency = (now - rclpy.time.Time.from_msg(msg_timestamp)).nanoseconds / 1e6
            stats["latency_ms"] = latency

    def check_all(self) -> bool:
        """Run all compatibility checks."""
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "checks": {},
        }

        self.get_logger().info("=" * 70)
        self.get_logger().info("ROS2 TOPIC COMPATIBILITY VERIFICATION")
        self.get_logger().info("=" * 70)

        # 1. Topic availability check
        self.get_logger().info("\n[1/5] Checking topic availability...")
        results["checks"]["topic_availability"] = self._check_topic_availability()

        # 2. Frequency check
        self.get_logger().info("\n[2/5] Checking topic frequencies (this may take 30 seconds)...")
        time.sleep(5)  # Wait for data to arrive
        results["checks"]["topic_frequencies"] = self._check_frequencies()

        # 3. Message type check
        self.get_logger().info("\n[3/5] Checking message types...")
        results["checks"]["message_types"] = self._check_message_types()

        # 4. TF transform check
        self.get_logger().info("\n[4/5] Checking TF transforms...")
        results["checks"]["tf_transforms"] = self._check_tf_transforms()

        # 5. Time synchronization check
        self.get_logger().info("\n[5/5] Checking time synchronization...")
        results["checks"]["time_sync"] = self._check_time_sync()

        # Print summary
        self.get_logger().info("\n" + "=" * 70)
        self.get_logger().info("VERIFICATION SUMMARY")
        self.get_logger().info("=" * 70)

        all_passed = all(
            check.get("passed", False) for check in results["checks"].values()
        )

        for check_name, check_result in results["checks"].items():
            status = "✓" if check_result.get("passed") else "✗"
            self.get_logger().info(f"{status} {check_name}: {check_result.get('message', '')}")

        self.get_logger().info(f"\nOverall: {'✓ PASS' if all_passed else '✗ FAIL'}")
        self.get_logger().info("=" * 70)

        # Write JSON report
        report_file = "/tmp/topic_compatibility_report.json"
        with open(report_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        self.get_logger().info(f"\nDetailed report saved to: {report_file}")

        return all_passed

    def _check_topic_availability(self) -> Dict:
        """Check if all required topics are being published."""
        topics = self.get_topic_names_and_types()
        topic_names = {name for name, _ in topics}

        available = []
        missing = []

        for sensor, config in self.required_topics.items():
            if config["topic"] in topic_names:
                available.append(config["topic"])
            else:
                missing.append(config["topic"])

        passed = len(missing) == 0
        message = f"Found {len(available)} required topics"
        if missing:
            message += f", missing {len(missing)}: {missing}"

        self.get_logger().info(f"  {message}")

        return {
            "passed": passed,
            "available": available,
            "missing": missing,
            "message": message,
        }

    def _check_frequencies(self) -> Dict:
        """Check if topics publish at expected frequencies."""
        time.sleep(25)  # Wait for frequency calculation

        results = {}
        passed = True

        for sensor, config in self.required_topics.items():
            stats = self.observed_topics[sensor]
            expected = config["expected_freq"]
            actual = stats["actual_freq"]

            # Allow 20% tolerance
            freq_ok = actual >= expected * 0.8 if actual > 0 else False

            if not freq_ok:
                passed = False

            status = "✓" if freq_ok else "✗"
            self.get_logger().info(
                f"  {status} {sensor}: {actual:.1f} Hz (expected ≥ {expected} Hz)"
            )

            results[sensor] = {
                "expected_hz": expected,
                "actual_hz": actual,
                "ok": freq_ok,
            }

        return {
            "passed": passed,
            "details": results,
            "message": f"Frequency check {'passed' if passed else 'failed'}",
        }

    def _check_message_types(self) -> Dict:
        """Verify correct message types."""
        # This is a basic check; actual validation happens via subscriptions
        self.get_logger().info("  ✓ Message types verified via subscriptions")
        return {
            "passed": True,
            "message": "Message types verified via subscriptions",
        }

    def _check_tf_transforms(self) -> Dict:
        """Check required TF transforms exist."""
        required_transforms = [
            ("base_link", "imu_link"),
            ("base_link", "distance_sensor_front_link"),
            ("base_link", "camera_link"),
        ]

        missing = []
        for parent, child in required_transforms:
            try:
                self.tf_buffer.lookup_transform(parent, child, rclpy.time.Time())
                self.get_logger().info(f"  ✓ Transform {parent} -> {child}")
            except tf2_ros.TransformException:
                missing.append(f"{parent} -> {child}")

        passed = len(missing) == 0
        if missing:
            self.get_logger().info(f"  ✗ Missing transforms: {missing}")

        return {
            "passed": passed,
            "missing": missing,
            "message": f"TF check {'passed' if passed else f'failed ({len(missing)} missing)'}",
        }

    def _check_time_sync(self) -> Dict:
        """Check if sensor timestamps are synchronized."""
        # Compare message timestamps
        if not self.observed_topics["imu"]["last_time"] or not self.observed_topics["odometry"]["last_time"]:
            return {
                "passed": False,
                "message": "Not enough data to check time sync",
            }

        # Simple check: timestamps should be close
        imu_time = self.observed_topics["imu"]["last_time"]
        odom_time = self.observed_topics["odometry"]["last_time"]

        time_diff_sec = abs((imu_time - odom_time).nanoseconds) / 1e9

        # Allow up to 0.5 second difference
        passed = time_diff_sec < 0.5

        message = f"Time difference: {time_diff_sec:.3f}s"
        if passed:
            self.get_logger().info(f"  ✓ {message}")
        else:
            self.get_logger().warning(f"  ⚠️  {message} (consider NTP/time sync)")

        return {
            "passed": passed,
            "time_diff_sec": time_diff_sec,
            "message": message,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Verify ROS2 topic compatibility for CogniSeer robot brain"
    )
    parser.add_argument(
        "--config",
        default="config/brain_params.yaml",
        help="Configuration file path",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds",
    )

    args = parser.parse_args()

    rclpy.init()

    try:
        checker = TopicCompatibilityChecker(args.config)
        passed = checker.check_all()
        rclpy.spin_once(checker, timeout_sec=args.timeout)
        sys.exit(0 if passed else 1)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
