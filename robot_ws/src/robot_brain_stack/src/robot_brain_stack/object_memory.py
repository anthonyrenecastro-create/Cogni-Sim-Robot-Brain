import json
import sqlite3
import time

import rclpy

from rclpy.node import Node
from std_msgs.msg import String


class ObjectMemory(Node):
    def __init__(self):
        super().__init__('object_memory')

        self.declare_parameter('db_path', '/tmp/robot_brain_objects.db')
        db_path = str(self.get_parameter('db_path').value)

        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS objects(
                object_id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                confidence REAL NOT NULL,
                timestamp REAL NOT NULL,
                metadata TEXT NOT NULL
            )
            '''
        )
        self.conn.commit()

        self.upsert_sub = self.create_subscription(String, '/brain/object_upsert', self.on_upsert, 10)
        self.query_sub = self.create_subscription(String, '/brain/object_query', self.on_query, 10)
        self.query_pub = self.create_publisher(String, '/brain/object_query_result', 10)

        self.get_logger().info(f'ObjectMemory using {db_path}')

    def on_upsert(self, msg: String):
        try:
            payload = json.loads(msg.data)
            self.conn.execute(
                '''
                INSERT INTO objects(object_id, label, x, y, confidence, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(object_id) DO UPDATE SET
                    label=excluded.label,
                    x=excluded.x,
                    y=excluded.y,
                    confidence=excluded.confidence,
                    timestamp=excluded.timestamp,
                    metadata=excluded.metadata
                ''',
                (
                    payload['object_id'],
                    payload['label'],
                    float(payload['x']),
                    float(payload['y']),
                    float(payload.get('confidence', 1.0)),
                    float(payload.get('timestamp', time.time())),
                    json.dumps(payload.get('metadata', {})),
                ),
            )
            self.conn.commit()
        except Exception as e:
            self.get_logger().error(f'Object upsert failed: {e}')

    def on_query(self, msg: String):
        try:
            payload = json.loads(msg.data)
            label = payload.get('label')
            rows = []
            if label:
                rows = self.conn.execute(
                    'SELECT object_id, label, x, y, confidence, timestamp, metadata FROM objects WHERE label = ? ORDER BY timestamp DESC LIMIT 20',
                    (label,),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    'SELECT object_id, label, x, y, confidence, timestamp, metadata FROM objects ORDER BY timestamp DESC LIMIT 20'
                ).fetchall()

            result = []
            for r in rows:
                result.append({
                    'object_id': r[0],
                    'label': r[1],
                    'x': r[2],
                    'y': r[3],
                    'confidence': r[4],
                    'timestamp': r[5],
                    'metadata': json.loads(r[6]),
                })

            out = String()
            out.data = json.dumps({'results': result})
            self.query_pub.publish(out)
        except Exception as e:
            self.get_logger().error(f'Object query failed: {e}')

    def destroy_node(self):
        self.conn.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ObjectMemory()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()