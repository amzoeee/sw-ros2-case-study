#!/usr/bin/env python3
"""Starter node for the Lunabotics ROS 2 case study.

Fill in TASKS 1-3 here. See README.md for the full description of each task.

Run it with:

    ros2 run move publisher

As shipped this node starts, spins, and does nothing -- that is intentional. Use it to
confirm your workspace is built and sourced before you write any logic.

Each task is marked with a TASK n.n comment matching the README. Commented-out lines are
deliberate: uncomment and complete them.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float64


class RobotController(Node):
    """Drives the robot, tracks its position error, and flags obstacles."""

    # ---- TASK 1.3 tuning constants -----------------------------------------
    # Robot starts at the origin facing +x, square to the barrier (x = 6,
    # spanning y -3..3). The route is a 90 deg pivot to -y, then a half circle
    # left about centre (ARC_RADIUS, 0):
    #     (x, y) = R(1 + cos t), R sin t,   t: pi -> 2pi
    # It leaves (0, 0) heading -y along the near face, bottoms out at (R, -R)
    # past the barrier's -y end, and finishes at (2R, 0) heading +y along the
    # far face. R = 5.0 keeps the path 4.8 m clear of the barrier's end.
    ARC_RADIUS = 5.0        # [m] half-circle radius
    LINEAR_SPEED = 0.5      # [m/s] forward speed along the arc
    PIVOT_RATE = 0.5        # [rad/s] yaw rate of the in-place pivot
    CMD_PERIOD = 0.1        # [s] command period

    def __init__(self):
        super().__init__('robot_controller')

        # Publish to /error when the position delta exceeds this (TASK 2.3).
        # This is a starting value -- justify whatever you settle on.
        self.error_thresh = 0.5

        # ---- TASK 1.2: publisher that drives the robot ---------------------
        self.move_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.move_timer = self.create_timer(self.CMD_PERIOD, self.send_move_cmd)

        # ---- TASK 1.3: the path you chose ----------------------------------
        # Segments driven open loop, in order: (v [m/s], omega [rad/s], t [s]).
        # Negative omega turns right, positive turns left.
        self.path = [
            # Pivot in place 90 deg right, from facing +x to facing -y.
            (0.0, -self.PIVOT_RATE, (math.pi / 2.0) / self.PIVOT_RATE),
            # Half circle left around the barrier's -y end.
            (self.LINEAR_SPEED,
             self.LINEAR_SPEED / self.ARC_RADIUS,
             (math.pi * self.ARC_RADIUS) / self.LINEAR_SPEED),
        ]

        self.segment_index = 0
        self.segment_elapsed = 0.0

        # ---- TASK 2.2: subscriber for the robot's 6D pose ------------------
        # One of the two onboard sensors reports 6D data. Find it (TASK 2.1).
        #
        # self.robot_pos_sub = self.create_subscription(
        #     <TODO: msg type>,
        #     '<TODO: topic name>',
        #     self.on_robot_pos,
        #     qos_profile_sensor_data,
        # )

        # ---- TASK 2.3: where the measured-vs-actual error goes -------------
        # self.error_pub = self.create_publisher(Float64, '/error', 10)
        #
        # Hint: ground truth for "actual" is published by the simulator on the
        # robot's odometry topic (nav_msgs/Odometry). Deciding what to compare,
        # and in which frame, is part of the task.

        # ---- TASK 3: lidar in, filtered obstacles out ----------------------
        # The lidar has a single vertical sample, so this cloud is one flat
        # row of points at the sensor's height -- not a 3D volume.
        #
        # self.lidar_sub = self.create_subscription(
        #     PointCloud2,
        #     '/lidar/points',
        #     self.on_lidar,
        #     qos_profile_sensor_data,
        # )
        # self.obstacle_pub = self.create_publisher(
        #     PointCloud2, '/obstacle_cloud', 10)

        self.get_logger().info(
            f'robot_controller started: pivot 90 deg right, then a '
            f'{math.pi * self.ARC_RADIUS:.1f} m half circle of radius '
            f'{self.ARC_RADIUS:.1f} m at {self.LINEAR_SPEED:.2f} m/s')

    # -----------------------------------------------------------------------
    # TASK 1.2 -- publish a velocity command
    # -----------------------------------------------------------------------
    def send_move_cmd(self):
        """Publish one Twist for the current segment of self.path."""
        cmd = Twist()

        if self.segment_index < len(self.path):
            linear, angular, duration = self.path[self.segment_index]
            cmd.linear.x = linear
            cmd.angular.z = angular

            self.segment_elapsed += self.CMD_PERIOD
            if self.segment_elapsed >= duration:
                self.segment_index += 1
                self.segment_elapsed = 0.0

        self.move_pub.publish(cmd)

    # -----------------------------------------------------------------------
    # TASK 2.3 -- compare reported position against ground truth
    # -----------------------------------------------------------------------
    def on_robot_pos(self, msg):
        """Compare the sensor's idea of where we are against the truth.

        Publish a Float64 on self.error_pub when the delta exceeds
        self.error_thresh.

        TODO: decide what "delta" means here and justify it in a comment.
        """
        raise NotImplementedError('TASK 2.3')

    # -----------------------------------------------------------------------
    # TASK 3.3 -- classify a single lidar point
    # -----------------------------------------------------------------------
    def is_obstacle(self, point):
        """Return True if `point` is something we must avoid.

        The barrier is passable -- treat it like dust in the air. The poles are
        not. `point` is an (x, y, z) tuple in the lidar's frame.

        TODO: decide what separates a pole from the barrier and implement it.
        """
        raise NotImplementedError('TASK 3.3')

    # -----------------------------------------------------------------------
    # TASK 3.2 -- filter the scan and republish what matters
    # -----------------------------------------------------------------------
    def on_lidar(self, msg):
        """Filter incoming points through is_obstacle and republish.

        point_cloud2.read_points(msg, field_names=('x', 'y', 'z')) iterates the
        cloud; point_cloud2.create_cloud_xyz32(msg.header, pts) builds the
        outgoing one.

        TODO: keep only the obstacle points and publish on self.obstacle_pub.
        """
        raise NotImplementedError('TASK 3.2')


def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
