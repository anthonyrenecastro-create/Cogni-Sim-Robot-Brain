import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class RecoveryManager(Node):
    def __init__(self):
        super().__init__('recovery_manager')
        self.fail_sub = self.create_subscription(String, '/brain/failure_event', self.on_failure, 10)
        self.rec_pub = self.create_publisher(String, '/brain/recovery_action', 10)
        self.get_logger().info('RecoveryManager started')

    def choose_recovery(self, failure_type: str):
        if failure_type == 'navigation_blocked':
            return ['clear_costmap', 'replan', 'backup', 'retry_navigation']
        if failure_type == 'localization_unstable':
            return ['stop', 'rotate_in_place', 'relocalize', 'retry_navigation']
        if failure_type == 'goal_timeout':
            return ['abort_current_goal', 'requeue_goal']
        if failure_type == 'low_battery':
            return ['cancel_goal', 'navigate_to_dock']
        return ['stop', 'request_operator_attention']

    def on_failure(self, msg: String):
        try:
            payload = json.loads(msg.data)
            failure_type = payload.get('failure_type', 'unknown')
            actions = self.choose_recovery(failure_type)
            out = String()
            out.data = json.dumps({
                'failure_type': failure_type,
                'actions': actions,
            })
            self.rec_pub.publish(out)
            self.get_logger().warn(f'Recovery for {failure_type}: {actions}')
        except Exception as e:
            self.get_logger().error(f'Recovery handling failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = RecoveryManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()