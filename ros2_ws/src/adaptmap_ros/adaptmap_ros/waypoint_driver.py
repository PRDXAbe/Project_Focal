"""Scripted repeatable demo lap: pure-pursuit-ish drive through waypoints using /odom."""
import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

WAYPOINTS = [(20, -1.0), (40, -1.0), (60, -1.0), (80, -1.0), (100, -1.0), (115, -1.0)]


class Driver(Node):
    def __init__(self):
        super().__init__("waypoint_driver")
        self.declare_parameter("speed", 1.5)
        self.speed = self.get_parameter("speed").value
        self.pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.i = 0

    def on_odom(self, msg):
        cmd = Twist()
        if self.i < len(WAYPOINTS):
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            gx, gy = WAYPOINTS[self.i]
            dx, dy = gx - p.x, gy - p.y
            if math.hypot(dx, dy) < 1.0:
                self.i += 1
                self.get_logger().info(f"waypoint {self.i}/{len(WAYPOINTS)}")
            err = math.atan2(math.sin(math.atan2(dy, dx) - yaw), math.cos(math.atan2(dy, dx) - yaw))
            cmd.angular.z = max(-1.0, min(1.0, 2.0 * err))
            cmd.linear.x = self.speed * max(0.0, math.cos(err))
        self.pub.publish(cmd)


def main():
    rclpy.init()
    try:
        rclpy.spin(Driver())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
