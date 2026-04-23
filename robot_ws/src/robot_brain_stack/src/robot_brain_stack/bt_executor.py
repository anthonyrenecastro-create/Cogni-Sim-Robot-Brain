import json
import math
from enum import Enum

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import BatteryState
from std_msgs.msg import String

from .failure_injection import should_inject_failure


class Status(str, Enum):
    RUNNING = 'running'
    SUCCESS = 'success'
    FAILURE = 'failure'


class BTExecutor(Node):
    def __init__(self):
        super().__init__('bt_executor')
        self.use_nav2 = bool(self.declare_parameter('use_nav2', True).value)
        self.nav_action_name = str(self.declare_parameter('nav_action_name', '/navigate_to_pose').value)
        self.nav_frame = str(self.declare_parameter('nav_frame', 'map').value)
        self.low_battery_threshold = float(self.declare_parameter('low_battery_threshold', 0.2).value)
        self.dock_x = float(self.declare_parameter('dock_x', 0.0).value)
        self.dock_y = float(self.declare_parameter('dock_y', 0.0).value)
        self.dock_yaw = float(self.declare_parameter('dock_yaw', 0.0).value)
        self.failure_injection_enabled = bool(self.declare_parameter('failure_injection_enabled', False).value)
        self.failure_injection_rate = float(self.declare_parameter('failure_injection_rate', 0.0).value)
        self.executive_state = {'mode': 'nominal'}
        self.backend_health = {'backend_available': False}
        self.recovery_actions = []
        self.active_goal = None
        self.battery_pct = 1.0
        self.is_charging = False
        self._nav_goal_in_progress = False
        self._nav_result_status = None

        self.goal_sub = self.create_subscription(String, '/brain/active_goal', self.on_active_goal, 10)
        self.exec_sub = self.create_subscription(String, '/brain/executive_state', self.on_exec_state, 10)
        self.backend_sub = self.create_subscription(String, '/brain/backend_health', self.on_backend_health, 10)
        self.recovery_sub = self.create_subscription(String, '/brain/recovery_action', self.on_recovery, 10)
        self.battery_sub = self.create_subscription(BatteryState, '/battery_state', self.on_battery, 10)
        self.health_sub = self.create_subscription(String, '/brain/health_status', self.on_health, 10)

        self.goal_result_pub = self.create_publisher(String, '/brain/goal_result', 10)
        self.failure_pub = self.create_publisher(String, '/brain/failure_event', 10)
        self.cmd_pub = self.create_publisher(Twist, '/brain/cmd_vel_raw', 10)
        self.heartbeat_pub = self.create_publisher(String, '/brain/heartbeat/bt_executor', 10)
        self.nav_client = ActionClient(self, NavigateToPose, self.nav_action_name)

        self.timer = self.create_timer(0.2, self.tick)
        self.heartbeat_timer = self.create_timer(1.0, self.publish_heartbeat)

    def on_active_goal(self, msg: String):
        try:
            incoming_goal = json.loads(msg.data)
            incoming_goal_id = incoming_goal.get('goal_id')
            current_goal_id = self.active_goal.get('goal_id') if self.active_goal else None
            if incoming_goal_id != current_goal_id:
                self._nav_goal_in_progress = False
                self._nav_result_status = None
            self.active_goal = incoming_goal
        except Exception as e:
            self.get_logger().error(f'Failed to parse active goal: {e}')

    def on_exec_state(self, msg: String):
        self.executive_state = json.loads(msg.data)

    def on_backend_health(self, msg: String):
        try:
            self.backend_health = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f'Failed to parse backend health: {e}')

    def on_recovery(self, msg: String):
        payload = json.loads(msg.data)
        self.recovery_actions = payload.get('actions', [])

    def on_battery(self, msg: BatteryState):
        if msg.percentage >= 0.0:
            self.battery_pct = float(max(0.0, min(1.0, msg.percentage)))
        self.is_charging = bool(msg.power_supply_status == BatteryState.POWER_SUPPLY_STATUS_CHARGING)

    def on_health(self, msg: String):
        try:
            payload = json.loads(msg.data)
            if payload.get('level') == 'critical':
                self.publish_failure('health_critical')
        except Exception:
            pass

    def publish_heartbeat(self):
        out = String()
        out.data = json.dumps({'node': 'bt_executor', 'alive': True})
        self.heartbeat_pub.publish(out)

    def publish_goal_result(self, goal_id: str, status: str):
        out = String()
        out.data = json.dumps({'goal_id': goal_id, 'status': status})
        self.goal_result_pub.publish(out)

    def publish_failure(self, failure_type: str):
        out = String()
        out.data = json.dumps({
            'failure_type': failure_type,
            'backend_error': self.backend_health.get('backend_error', ''),
        })
        self.failure_pub.publish(out)

    def do_patrol(self, target: dict):
        cmd = Twist()
        cmd.linear.x = 0.2 if self.executive_state.get('mode') != 'cautious' else 0.1
        cmd.angular.z = 0.2
        self.cmd_pub.publish(cmd)
        return Status.RUNNING

    def do_go_to_pose(self, target: dict):
        if 'x' not in target or 'y' not in target:
            return Status.FAILURE

        if not self.use_nav2:
            cmd = Twist()
            cmd.linear.x = 0.15 if self.executive_state.get('mode') == 'cautious' else 0.25
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            return Status.RUNNING

        if self._nav_result_status is not None:
            nav_status = self._nav_result_status
            self._nav_result_status = None
            if nav_status == GoalStatus.STATUS_SUCCEEDED:
                return Status.SUCCESS
            return Status.FAILURE

        if self._nav_goal_in_progress:
            return Status.RUNNING

        if not self.nav_client.wait_for_server(timeout_sec=0.25):
            self.get_logger().error('Nav2 action server not available: ' + self.nav_action_name)
            return Status.FAILURE

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = self.nav_frame
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(target['x'])
        goal_msg.pose.pose.position.y = float(target['y'])
        yaw = float(target.get('yaw', 0.0))
        goal_msg.pose.pose.orientation.z = math.sin(yaw * 0.5)
        goal_msg.pose.pose.orientation.w = math.cos(yaw * 0.5)

        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self._on_nav_goal_response)
        self._nav_goal_in_progress = True
        return Status.RUNNING

    def do_dock(self, target: dict):
        dock_target = {
            'x': float(target.get('x', self.dock_x)),
            'y': float(target.get('y', self.dock_y)),
            'yaw': float(target.get('yaw', self.dock_yaw)),
        }
        nav_status = self.do_go_to_pose(dock_target)
        if nav_status == Status.FAILURE:
            return Status.FAILURE
        if nav_status == Status.SUCCESS:
            if self.is_charging:
                return Status.SUCCESS
            return Status.RUNNING
        return Status.RUNNING

    def _inject_failure(self):
        return should_inject_failure(self.failure_injection_enabled, self.failure_injection_rate)

    def _on_nav_goal_response(self, future):
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().warn('Nav2 goal rejected')
                self._nav_goal_in_progress = False
                self._nav_result_status = GoalStatus.STATUS_ABORTED
                return
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self._on_nav_result)
        except Exception as e:
            self.get_logger().error(f'Nav2 goal submission failed: {e}')
            self._nav_goal_in_progress = False
            self._nav_result_status = GoalStatus.STATUS_ABORTED

    def _on_nav_result(self, future):
        try:
            result = future.result()
            self._nav_result_status = result.status
        except Exception as e:
            self.get_logger().error(f'Nav2 result handling failed: {e}')
            self._nav_result_status = GoalStatus.STATUS_ABORTED
        finally:
            self._nav_goal_in_progress = False

    def tick(self):
        if self.active_goal is None:
            return

        goal_id = self.active_goal['goal_id']
        goal_type = self.active_goal['goal_type']
        target = self.active_goal.get('target', {})

        if not bool(self.backend_health.get('backend_available', False)):
            self.publish_failure('model_unavailable')
            self.publish_goal_result(goal_id, 'aborted')
            self.active_goal = None
            self._nav_goal_in_progress = False
            self._nav_result_status = None
            return

        if self.recovery_actions:
            self.get_logger().warn(f'Executing recovery plan: {self.recovery_actions}')
            self.recovery_actions = []
            return

        if self._inject_failure():
            self.publish_failure('injected_failure')
            self.publish_goal_result(goal_id, 'aborted')
            self.active_goal = None
            self._nav_goal_in_progress = False
            self._nav_result_status = None
            return

        if self.battery_pct <= self.low_battery_threshold and goal_type != 'dock':
            self.publish_failure('low_battery')
            self.publish_goal_result(goal_id, 'aborted')
            self.active_goal = {
                'goal_id': f'dock-{goal_id}',
                'goal_type': 'dock',
                'target': {
                    'x': self.dock_x,
                    'y': self.dock_y,
                    'yaw': self.dock_yaw,
                },
            }
            self._nav_goal_in_progress = False
            self._nav_result_status = None
            return

        if goal_type == 'patrol':
            status = self.do_patrol(target)
        elif goal_type == 'go_to_pose':
            status = self.do_go_to_pose(target)
        elif goal_type == 'dock':
            status = self.do_dock(target)
        else:
            status = Status.FAILURE

        if status == Status.FAILURE:
            self.publish_failure('goal_execution_failed')
            self.publish_goal_result(goal_id, 'aborted')
            self.active_goal = None
            self._nav_goal_in_progress = False
            self._nav_result_status = None
        elif status == Status.SUCCESS:
            self.publish_goal_result(goal_id, 'done')
            self.active_goal = None
            self._nav_goal_in_progress = False
            self._nav_result_status = None


def main(args=None):
    rclpy.init(args=args)
    node = BTExecutor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()