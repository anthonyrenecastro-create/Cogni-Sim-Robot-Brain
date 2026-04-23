import json
import os
import sys

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import BatteryState
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class ExecutiveContext(Node):
    def __init__(self):
        super().__init__('executive_context')
        self.tick_hz = float(self.declare_parameter('tick_hz', 5.0).value)
        self.input_dim = int(self.declare_parameter('brain_input_dim', 32).value)
        self.domain = str(self.declare_parameter('domain', 'default').value)
        self.brain_python_path = str(
            self.declare_parameter('brain_python_path', '/workspaces/Cogni-Sim-Robot-Brain/CogniSeer1.0-main').value
        )
        self.brain_checkpoint = str(
            self.declare_parameter('brain_checkpoint', '/workspaces/Cogni-Sim-Robot-Brain/CogniSeer1.0-main/robot_brain_fp32.pt').value
        )

        self._backend_ready = False
        self._torch = None
        self._brain = None
        self._backend_error = ''
        self._last_infer_ok = False
        self._latest_obs = np.zeros(self.input_dim, dtype=np.float32)
        self._battery_pct = 1.0
        self._scan_min = 10.0
        self._speed_norm = 0.0

        self.sensor_sub = self.create_subscription(Float32MultiArray, '/sensor_input', self.on_sensor, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.battery_sub = self.create_subscription(BatteryState, '/battery_state', self.on_battery, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.on_scan, 10)

        self.pub = self.create_publisher(String, '/brain/executive_state', 10)
        self.backend_health_pub = self.create_publisher(String, '/brain/backend_health', 10)
        self.heartbeat_pub = self.create_publisher(String, '/brain/heartbeat/executive_context', 10)
        self.timer = self.create_timer(1.0 / max(self.tick_hz, 1.0), self.tick)
        self.heartbeat_timer = self.create_timer(1.0, self.publish_heartbeat)

        self._init_brain_backend()

    def _init_brain_backend(self):
        try:
            if self.brain_python_path and os.path.isdir(self.brain_python_path):
                if self.brain_python_path not in sys.path:
                    sys.path.insert(0, self.brain_python_path)

            import torch
            from torch_edge_robot_brain import TorchBrainConfig, TorchEdgeRobotBrain

            cfg = TorchBrainConfig(input_dim=self.input_dim)
            brain = TorchEdgeRobotBrain(cfg)

            if self.brain_checkpoint and os.path.exists(self.brain_checkpoint):
                state_dict = torch.load(self.brain_checkpoint, map_location='cpu', weights_only=False)
                if isinstance(state_dict, dict):
                    brain.load_state_dict(state_dict, strict=False)

            brain.eval()
            brain.reset_state(batch_size=1, device='cpu')

            self._torch = torch
            self._brain = brain
            self._backend_ready = True
            self._backend_error = ''
            self.get_logger().info('ExecutiveContext model backend ready')
        except Exception as exc:
            self._backend_ready = False
            self._backend_error = str(exc)
            self.get_logger().warn(f'ExecutiveContext backend unavailable, publishing safe fallback context: {exc}')

    def on_sensor(self, msg: Float32MultiArray):
        obs = np.asarray(msg.data, dtype=np.float32)
        if obs.size < self.input_dim:
            padded = np.zeros(self.input_dim, dtype=np.float32)
            padded[:obs.size] = obs
            obs = padded
        elif obs.size > self.input_dim:
            obs = obs[:self.input_dim]
        self._latest_obs = obs

    def on_odom(self, msg: Odometry):
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        self._speed_norm = float(min(2.0, np.sqrt(vx * vx + vy * vy)))

    def on_battery(self, msg: BatteryState):
        if msg.percentage >= 0.0:
            self._battery_pct = float(max(0.0, min(1.0, msg.percentage)))

    def on_scan(self, msg: LaserScan):
        if msg.ranges:
            valid = [r for r in msg.ranges if np.isfinite(r) and r > 0.0]
            if valid:
                self._scan_min = float(min(valid))

    def _sensor_blend(self):
        obs = np.array(self._latest_obs, copy=True)
        if self.input_dim >= 3:
            obs[0] = np.float32(self._battery_pct)
            obs[1] = np.float32(min(10.0, self._scan_min) / 10.0)
            obs[2] = np.float32(min(2.0, self._speed_norm) / 2.0)
        return obs

    def publish_heartbeat(self):
        out = String()
        out.data = json.dumps({'node': 'executive_context', 'alive': True})
        self.heartbeat_pub.publish(out)

    def _infer_context(self):
        if not self._backend_ready:
            self._last_infer_ok = False
            return {
                'mode': 'cautious',
                'stability': 0.0,
                'coherence': 0.0,
                'domain': self.domain,
                'priority_bias': 0.0,
            }

        try:
            with self._torch.no_grad():
                obs = self._sensor_blend()
                obs_t = self._torch.tensor(obs, dtype=self._torch.float32).view(1, self.input_dim)
                out = self._brain(obs_t)
            self._last_infer_ok = True
        except Exception as exc:
            self._last_infer_ok = False
            self._backend_error = str(exc)
            return {
                'mode': 'cautious',
                'stability': 0.0,
                'coherence': 0.0,
                'domain': self.domain,
                'priority_bias': 0.0,
            }

        stability = _clamp01(float(out['res_stability'].mean().item()))
        variance = max(0.0, float(out['res_variance'].mean().item()))
        coherence = _clamp01(1.0 / (1.0 + variance))
        confidence = _clamp01(float(out['action_confidence'].mean().item()))
        priority_bias = _clamp01(0.5 * stability + 0.5 * confidence)

        if stability < 0.65:
            mode = 'cautious'
        elif coherence < 0.7:
            mode = 'reorganizing'
        else:
            mode = 'nominal'

        return {
            'mode': mode,
            'stability': stability,
            'coherence': coherence,
            'domain': self.domain,
            'priority_bias': priority_bias,
        }

    def _backend_health_payload(self):
        return {
            'backend_available': bool(self._backend_ready and self._last_infer_ok),
            'backend_ready': bool(self._backend_ready),
            'last_infer_ok': bool(self._last_infer_ok),
            'backend_error': self._backend_error,
            'model_checkpoint': self.brain_checkpoint,
            'domain': self.domain,
        }

    def tick(self):
        ctx = self._infer_context()
        ctx_msg = String()
        ctx_msg.data = json.dumps(ctx)
        self.pub.publish(ctx_msg)

        health_msg = String()
        health_msg.data = json.dumps(self._backend_health_payload())
        self.backend_health_pub.publish(health_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ExecutiveContext()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()