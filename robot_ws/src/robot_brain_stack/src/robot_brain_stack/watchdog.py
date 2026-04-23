import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class Watchdog(Node):
    def __init__(self):
        super().__init__('watchdog')
        self.declare_parameter('timeout_sec', 4.0)
        self.timeout_sec = float(self.get_parameter('timeout_sec').value)

        self.last_seen = {}
        self.heartbeat_topics = [
            '/brain/heartbeat/goal_manager',
            '/brain/heartbeat/executive_context',
            '/brain/heartbeat/bt_executor',
            '/brain/heartbeat/health_monitor',
        ]

        for topic in self.heartbeat_topics:
            self.create_subscription(String, topic, self.on_heartbeat, 10)

        self.fail_pub = self.create_publisher(String, '/brain/failure_event', 10)
        self.timer = self.create_timer(1.0, self.tick)

    def on_heartbeat(self, msg: String):
        try:
            payload = json.loads(msg.data)
            node = str(payload.get('node', 'unknown'))
            self.last_seen[node] = time.time()
        except Exception:
            pass

    def tick(self):
        now = time.time()
        for node, ts in list(self.last_seen.items()):
            if (now - ts) > self.timeout_sec:
                out = String()
                out.data = json.dumps({'failure_type': 'watchdog_timeout', 'node': node})
                self.fail_pub.publish(out)
                self.last_seen[node] = now


def main(args=None):
    rclpy.init(args=args)
    node = Watchdog()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
