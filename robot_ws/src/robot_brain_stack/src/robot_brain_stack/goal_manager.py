import json
import sqlite3
import uuid
from typing import List

import rclpy

from rclpy.node import Node
from std_msgs.msg import String

from .mission_types import Goal


class GoalManager(Node):
    def __init__(self):
        super().__init__('goal_manager')
        self.declare_parameter('mission_db_path', '/tmp/robot_brain_missions.db')
        db_path = str(self.get_parameter('mission_db_path').value)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS missions(
                goal_id TEXT PRIMARY KEY,
                goal_type TEXT NOT NULL,
                priority INTEGER NOT NULL,
                target TEXT NOT NULL,
                created_at REAL NOT NULL,
                status TEXT NOT NULL,
                retries INTEGER NOT NULL
            )
            '''
        )
        self.conn.commit()

        self.goals: List[Goal] = []
        self.active_goal = None

        self.goal_sub = self.create_subscription(String, '/brain/add_goal', self.on_add_goal, 10)
        self.done_sub = self.create_subscription(String, '/brain/goal_result', self.on_goal_result, 10)
        self.active_pub = self.create_publisher(String, '/brain/active_goal', 10)
        self.heartbeat_pub = self.create_publisher(String, '/brain/heartbeat/goal_manager', 10)
        self.timer = self.create_timer(0.5, self.tick)
        self.heartbeat_timer = self.create_timer(1.0, self.publish_heartbeat)

        self._load_goals()

        self.get_logger().info('GoalManager started')

    def _load_goals(self):
        rows = self.conn.execute(
            'SELECT goal_id, goal_type, priority, target, created_at, status, retries FROM missions ORDER BY priority DESC, created_at ASC'
        ).fetchall()
        self.goals = [
            Goal(
                goal_id=r[0],
                goal_type=r[1],
                priority=int(r[2]),
                target=json.loads(r[3]),
                created_at=float(r[4]),
                status=r[5],
                retries=int(r[6]),
            )
            for r in rows
            if r[5] not in ('done', 'aborted')
        ]

    def _save_goal(self, goal: Goal):
        self.conn.execute(
            '''
            INSERT INTO missions(goal_id, goal_type, priority, target, created_at, status, retries)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(goal_id) DO UPDATE SET
                goal_type=excluded.goal_type,
                priority=excluded.priority,
                target=excluded.target,
                created_at=excluded.created_at,
                status=excluded.status,
                retries=excluded.retries
            ''',
            (
                goal.goal_id,
                goal.goal_type,
                int(goal.priority),
                json.dumps(goal.target),
                float(goal.created_at),
                goal.status,
                int(goal.retries),
            ),
        )
        self.conn.commit()

    def publish_heartbeat(self):
        out = String()
        out.data = json.dumps({'node': 'goal_manager', 'alive': True})
        self.heartbeat_pub.publish(out)

    def on_add_goal(self, msg: String):
        try:
            payload = json.loads(msg.data)
            goal = Goal(
                goal_id=payload.get('goal_id', str(uuid.uuid4())),
                goal_type=payload['goal_type'],
                priority=int(payload.get('priority', 100)),
                target=payload.get('target', {}),
            )
            self.goals.append(goal)
            self.goals.sort(key=lambda g: (-g.priority, g.created_at))
            self._save_goal(goal)
            self.get_logger().info(f'Added goal {goal.goal_id} type={goal.goal_type} priority={goal.priority}')
        except Exception as e:
            self.get_logger().error(f'Failed to add goal: {e}')

    def on_goal_result(self, msg: String):
        try:
            payload = json.loads(msg.data)
            goal_id = payload['goal_id']
            status = payload['status']
            for g in self.goals:
                if g.goal_id == goal_id:
                    g.status = status
                    self._save_goal(g)
            self.goals = [g for g in self.goals if g.status not in ('done', 'aborted')]
            if self.active_goal and self.active_goal.goal_id == goal_id:
                self.active_goal = None
        except Exception as e:
            self.get_logger().error(f'Failed to process goal result: {e}')

    def select_goal(self):
        candidates = [g for g in self.goals if g.status in ('pending', 'running')]
        if not candidates:
            return None
        candidates.sort(key=lambda g: (-g.priority, g.created_at))
        return candidates[0]

    def tick(self):
        selected = self.select_goal()
        if selected is None:
            return
        selected.status = 'running'
        self.active_goal = selected
        self._save_goal(selected)
        out = String()
        out.data = json.dumps({
            'goal_id': selected.goal_id,
            'goal_type': selected.goal_type,
            'priority': selected.priority,
            'target': selected.target,
        })
        self.active_pub.publish(out)

    def destroy_node(self):
        self.conn.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GoalManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()