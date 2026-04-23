import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from robot_brain_stack.utils import clamp


class SafetyLayer(Node):
    def __init__(self):
        super().__init__('safety_layer')
        self.max_linear = float(self.declare_parameter('max_linear', 0.5).value)
        self.max_angular = float(self.declare_parameter('max_angular', 1.2).value)
        self.max_linear_delta = float(self.declare_parameter('max_linear_delta', 0.05).value)
        self.max_angular_delta = float(self.declare_parameter('max_angular_delta', 0.1).value)
        self.last_linear = 0.0
        self.last_angular = 0.0

        self.sub = self.create_subscription(Twist, '/brain/cmd_vel_raw', self.on_cmd, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

    def slew(self, current: float, target: float, delta: float) -> float:
        if target > current + delta:
            return current + delta
        if target < current - delta:
            return current - delta
        return target

    def on_cmd(self, msg: Twist):
        lin = clamp(msg.linear.x, -self.max_linear, self.max_linear)
        ang = clamp(msg.angular.z, -self.max_angular, self.max_angular)

        lin = self.slew(self.last_linear, lin, self.max_linear_delta)
        ang = self.slew(self.last_angular, ang, self.max_angular_delta)

        self.last_linear = lin
        self.last_angular = ang

        out = Twist()
        out.linear.x = lin
        out.angular.z = ang
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = SafetyLayer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()