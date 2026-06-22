"""
ROS2 Integration example for LLM cognition layer.

Shows how to integrate the slow brain (LLM) with the fast brain (PyTorch)
in a real ROS2 robot system.
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import asyncio
import logging
from typing import Dict, Any

from std_msgs.msg import Float32MultiArray, String, Int32
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from llm_cognition import CognitionLayer
from llm_cognition.decision_router import DecisionType, Urgency

logger = logging.getLogger(__name__)


class SlowBrainNode(Node):
    """
    Slow brain node running the LLM-based planning layer.
    
    Runs asynchronously at low frequency (0.1-1 Hz) without blocking
    the fast brain's real-time control loop.
    """
    
    def __init__(self):
        super().__init__('slow_brain_node')
        
        # Initialize cognition layer
        self.cognition = CognitionLayer(
            provider="ollama",
            model="llama2",
            memory_persistence_dir=self.get_parameter('memory_dir').value
            if self.has_parameter('memory_dir')
            else None
        )
        
        # Subscriptions
        self.create_subscription(
            Float32MultiArray,
            '/brain/sensors',
            self.sensor_callback,
            10
        )
        self.create_subscription(
            Odometry,
            '/odom',
            self.odometry_callback,
            1
        )
        
        # Publishers
        self.plan_pub = self.create_publisher(String, '/brain/plan', 10)
        self.explanation_pub = self.create_publisher(String, '/brain/explanation', 10)
        self.status_pub = self.create_publisher(String, '/brain/status', 10)
        
        # Timer for slow brain planning loop (1 Hz)
        self.create_timer(1.0, self.slow_brain_loop)
        
        # State
        self.current_goal = None
        self.current_observation = None
        self.current_position = (0.0, 0.0)
        self.mission_active = False
        
        self.get_logger().info("Slow brain node initialized (LLM-based planning)")
    
    def sensor_callback(self, msg: Float32MultiArray) -> None:
        """Receive sensor observations."""
        self.current_observation = {
            "sensors": list(msg.data),
            "timestamp": self.get_clock().now().to_msg(),
        }
    
    def odometry_callback(self, msg: Odometry) -> None:
        """Receive odometry (position)."""
        self.current_position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        )
        
        # Add spatial context to memory
        self.cognition.add_observation(
            f"Position: {self.current_position[0]:.2f}, {self.current_position[1]:.2f}"
        )
    
    def slow_brain_loop(self) -> None:
        """Slow brain planning loop (runs at ~1 Hz)."""
        if not self.current_observation:
            return
        
        # Plan recovery if failure detected
        if self._failure_detected():
            self.get_logger().warn("Failure detected, planning recovery...")
            asyncio.create_task(self._plan_recovery())
        
        # Plan next steps for current goal
        if self.mission_active and self.current_goal:
            asyncio.create_task(
                self._plan_next_steps(self.current_goal)
            )
        
        # Periodically generate status report
        asyncio.create_task(self._generate_status())
    
    async def _plan_recovery(self) -> None:
        """Plan recovery from detected failure."""
        recovery = await self.cognition.handle_failure(
            failure="unknown_failure",  # Would come from fast brain
            context=f"Position: {self.current_position}"
        )
        
        if recovery:
            msg = String()
            msg.data = f"Recovery plan: {recovery.failure_type}"
            self.plan_pub.publish(msg)
    
    async def _plan_next_steps(self, goal: str) -> None:
        """Plan next steps toward goal."""
        plan = await self.cognition.plan_mission(goal)
        
        if plan:
            msg = String()
            msg.data = f"Plan: {plan.goal}, {len(plan.steps)} steps, "
            msg.data += f"success prob: {plan.success_probability:.1%}"
            self.plan_pub.publish(msg)
    
    async def _generate_status(self) -> None:
        """Generate status report."""
        status = await self.cognition.generate_status_report()
        
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
    
    def _failure_detected(self) -> bool:
        """Check if failure has been detected by fast brain."""
        # This would subscribe to fast brain failure signals
        return False  # Placeholder


class FastBrainIntegration(Node):
    """
    Fast brain node (PyTorch control) with slow brain coordination.
    
    Demonstrates how the fast brain (20 Hz) coordinates with slow brain (1 Hz).
    Fast brain makes reactive decisions; slow brain does planning.
    """
    
    def __init__(self, cognition: CognitionLayer):
        super().__init__('fast_brain_node')
        
        self.cognition = cognition
        self.control_frequency = 20  # Hz
        
        # Subscriptions
        self.create_subscription(
            Float32MultiArray,
            '/brain/sensors',
            self.sensor_callback,
            10
        )
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Fast control loop (20 Hz)
        self.create_timer(1.0 / self.control_frequency, self.control_loop)
        
        self.current_observation = None
        self.last_decision = None
        
        self.get_logger().info("Fast brain node initialized (PyTorch control, 20Hz)")
    
    def sensor_callback(self, msg: Float32MultiArray) -> None:
        """Receive sensor observations."""
        self.current_observation = {
            "sensors": list(msg.data),
            "timestamp": self.get_clock().now().to_msg(),
        }
    
    def control_loop(self) -> None:
        """Fast control loop running at 20 Hz."""
        if not self.current_observation:
            return
        
        # Make fast decision (should return immediately from PyTorch)
        decision = self.cognition.decide(
            observation=self.current_observation,
            decision_type=DecisionType.OBSTACLE_AVOIDANCE,
            urgency=Urgency.HIGH,
        )
        
        # Execute decision
        self._execute_decision(decision)
        self.last_decision = decision
    
    def _execute_decision(self, decision) -> None:
        """Execute decision by sending control commands."""
        twist = Twist()
        
        # Parse decision action and convert to velocity command
        if "stop" in decision.action.lower():
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        elif "turn_right" in decision.action.lower():
            twist.linear.x = 0.1
            twist.angular.z = -0.5
        elif "forward" in decision.action.lower():
            twist.linear.x = 0.3
            twist.angular.z = 0.0
        
        self.cmd_vel_pub.publish(twist)


def main():
    """Run integrated fast+slow brain system."""
    rclpy.init()
    
    # Shared cognition layer
    cognition = CognitionLayer(provider="ollama", model="llama2")
    
    # Create both nodes
    fast_brain = FastBrainIntegration(cognition)
    slow_brain = SlowBrainNode()
    
    # Use multithreaded executor to run both at their natural frequencies
    executor = MultiThreadedExecutor()
    executor.add_node(fast_brain)
    executor.add_node(slow_brain)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
