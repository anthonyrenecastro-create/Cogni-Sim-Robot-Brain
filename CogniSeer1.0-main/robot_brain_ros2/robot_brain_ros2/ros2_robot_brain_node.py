#!/usr/bin/env python3
import numpy as np
import rclpy

from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist

from .edge_runtime import EdgeRuntimeBrain
from .safety_layer import ActuationSafetyLayer, SafetyConfig


class RobotBrainNode(Node):
    def __init__(self):
        super().__init__("robot_brain_node")

        self.declare_parameter("model_path", "deploy/robot_brain_int8_stateful.pt")
        self.declare_parameter("input_dim", 32)
        self.declare_parameter("cmd_scale_linear", 0.5)
        self.declare_parameter("cmd_scale_angular", 1.0)
        self.declare_parameter("safety.motor_min", -1.0)
        self.declare_parameter("safety.motor_max", 1.0)
        self.declare_parameter("safety.max_linear_x", 0.5)
        self.declare_parameter("safety.max_angular_z", 1.0)
        self.declare_parameter("safety.max_linear_delta", 0.10)
        self.declare_parameter("safety.max_angular_delta", 0.20)
        self.declare_parameter("safety.sensor_timeout_sec", 0.50)
        self.declare_parameter("safety.stop_on_stale_sensor", True)

        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        input_dim = self.get_parameter("input_dim").get_parameter_value().integer_value

        self.cmd_scale_linear = self.get_parameter("cmd_scale_linear").value
        self.cmd_scale_angular = self.get_parameter("cmd_scale_angular").value

        safety_cfg = SafetyConfig(
            motor_min=float(self.get_parameter("safety.motor_min").value),
            motor_max=float(self.get_parameter("safety.motor_max").value),
            max_linear_x=float(self.get_parameter("safety.max_linear_x").value),
            max_angular_z=float(self.get_parameter("safety.max_angular_z").value),
            max_linear_delta=float(self.get_parameter("safety.max_linear_delta").value),
            max_angular_delta=float(self.get_parameter("safety.max_angular_delta").value),
            sensor_timeout_sec=float(self.get_parameter("safety.sensor_timeout_sec").value),
            stop_on_stale_sensor=bool(self.get_parameter("safety.stop_on_stale_sensor").value),
        )
        self.safety = ActuationSafetyLayer(safety_cfg)
        self._last_safety_reason = "ok"

        self.brain = EdgeRuntimeBrain(model_path=model_path, input_dim=input_dim)
        self.input_dim = int(input_dim)

        self.latest_obs = np.zeros(self.input_dim, dtype=np.float32)
        self.last_sensor_time = self.get_clock().now()

        self.sensor_sub = self.create_subscription(
            Float32MultiArray,
            "/brain/sensors",
            self.sensor_callback,
            10,
        )

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.motor_pub = self.create_publisher(Float32MultiArray, "/brain/motor_command", 10)

        self.timer = self.create_timer(0.05, self.control_loop)  # 20 Hz

        self.get_logger().info("RobotBrainNode started")

    def sensor_callback(self, msg: Float32MultiArray):
        data = np.array(msg.data, dtype=np.float32)
        if data.shape[0] >= self.input_dim:
            self.latest_obs = data[:self.input_dim]
        else:
            padded = np.zeros(self.input_dim, dtype=np.float32)
            padded[: data.shape[0]] = data
            self.latest_obs = padded
        self.last_sensor_time = self.get_clock().now()

    def control_loop(self):
        motor = self.brain.step(self.latest_obs)
        sensor_age_sec = (self.get_clock().now() - self.last_sensor_time).nanoseconds / 1e9
        sensor_fresh = sensor_age_sec <= self.safety.cfg.sensor_timeout_sec

        gated = self.safety.apply(
            motor,
            cmd_scale_linear=self.cmd_scale_linear,
            cmd_scale_angular=self.cmd_scale_angular,
            sensor_fresh=sensor_fresh,
        )

        if gated["reason"] != self._last_safety_reason and gated["reason"] != "ok":
            self.get_logger().warn(f"Actuation safety engaged: {gated['reason']}")
        self._last_safety_reason = gated["reason"]

        motor_msg = Float32MultiArray()
        motor_msg.data = gated["safe_motor"].astype(np.float32).tolist()
        self.motor_pub.publish(motor_msg)

        twist = Twist()
        twist.linear.x = float(gated["linear_x"])
        twist.angular.z = float(gated["angular_z"])
        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = RobotBrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
