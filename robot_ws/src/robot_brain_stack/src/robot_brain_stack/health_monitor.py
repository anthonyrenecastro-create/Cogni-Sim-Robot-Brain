import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class HealthMonitor(Node):
    def __init__(self):
        super().__init__('health_monitor')
        self.declare_parameter('stability_warn_threshold', 0.45)
        self.declare_parameter('stability_critical_threshold', 0.25)
        self.declare_parameter('heartbeat_timeout_sec', 3.0)

        self.stability_warn_threshold = float(self.get_parameter('stability_warn_threshold').value)
        self.stability_critical_threshold = float(self.get_parameter('stability_critical_threshold').value)
        self.heartbeat_timeout_sec = float(self.get_parameter('heartbeat_timeout_sec').value)

        self.last_heartbeats = {}
        self.latest_exec = {'stability': 1.0, 'coherence': 1.0}
        self.latest_backend = {'backend_available': False}

        self.exec_sub = self.create_subscription(String, '/brain/executive_state', self.on_exec_state, 10)
        self.backend_sub = self.create_subscription(String, '/brain/backend_health', self.on_backend, 10)
        self.hb_goal_sub = self.create_subscription(String, '/brain/heartbeat/goal_manager', self.on_heartbeat, 10)
        self.hb_exec_sub = self.create_subscription(String, '/brain/heartbeat/executive_context', self.on_heartbeat, 10)
        self.hb_bt_sub = self.create_subscription(String, '/brain/heartbeat/bt_executor', self.on_heartbeat, 10)

        self.health_pub = self.create_publisher(String, '/brain/health_status', 10)
        self.heartbeat_pub = self.create_publisher(String, '/brain/heartbeat/health_monitor', 10)

        self.timer = self.create_timer(1.0, self.tick)

    def on_exec_state(self, msg: String):
        try:
            self.latest_exec = json.loads(msg.data)
        except Exception:
            pass

    def on_backend(self, msg: String):
        try:
            self.latest_backend = json.loads(msg.data)
        except Exception:
            pass

    def on_heartbeat(self, msg: String):
        try:
            payload = json.loads(msg.data)
            node = str(payload.get('node', 'unknown'))
            self.last_heartbeats[node] = time.time()
        except Exception:
            pass

    def _stale_nodes(self):
        now = time.time()
        stale = []
        for node, ts in self.last_heartbeats.items():
            if (now - ts) > self.heartbeat_timeout_sec:
                stale.append(node)
        return stale

    def tick(self):
        now = time.time()
        stability = float(self.latest_exec.get('stability', 0.0))
        coherence = float(self.latest_exec.get('coherence', 0.0))
        backend_available = bool(self.latest_backend.get('backend_available', False))
        stale = self._stale_nodes()

        level = 'ok'
        reason = 'nominal'
        if not backend_available:
            level = 'critical'
            reason = 'backend_unavailable'
        elif stale:
            level = 'critical'
            reason = 'heartbeat_timeout'
        elif stability < self.stability_critical_threshold:
            level = 'critical'
            reason = 'stability_critical'
        elif stability < self.stability_warn_threshold:
            level = 'warn'
            reason = 'stability_low'

        out = String()
        out.data = json.dumps(
            {
                'ts': now,
                'level': level,
                'reason': reason,
                'stability': stability,
                'coherence': coherence,
                'backend_available': backend_available,
                'stale_nodes': stale,
            }
        )
        self.health_pub.publish(out)

        hb = String()
        hb.data = json.dumps({'node': 'health_monitor', 'alive': True})
        self.heartbeat_pub.publish(hb)


def main(args=None):
    rclpy.init(args=args)
    node = HealthMonitor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
