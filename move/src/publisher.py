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
    # Arc from the origin facing +x, centre (0, -R): x = R sin(t),
    # y = -R(1 - cos(t)), t = s / R. Defaults clear the barrier corner
    # (x 5.75, y -3) by ~1.3 m against a 0.6 m chassis half-width.
    ARC_LENGTH = 9.0        # [m] total arc length to drive
    ARC_RADIUS = 6.0        # [m] turning radius; smaller = tighter arc
    TURN_SIGN = -1.0        # -1 turns toward -y (away from the pillar at +2.5)
    LINEAR_SPEED = 0.5      # [m/s] forward speed along the arc
    CMD_PERIOD = 0.1        # [s] command period; 10 Hz is plenty for a diff drive

    def __init__(self):
        super().__init__('robot_controller')

        # Publish to /error when the position delta exceeds this (TASK 2.3).
        # This is a starting value -- justify whatever you settle on.
        self.error_thresh = 0.5

        # ---- TASK 1.2: publisher that drives the robot ---------------------
        # model.sdf's diff-drive plugin subscribes to /cmd_vel; sim.launch.py
        # bridges it. Reliable QoS: commands, not sensor samples.
        self.move_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.move_timer = self.create_timer(self.CMD_PERIOD, self.send_move_cmd)

        # ---- TASK 1.3: the path you chose ----------------------------------
        # A single constant-curvature arc, open loop, turning toward -y.
        #
        # Why: driving +x hits the barrier at x = 6 (spans y -3..3), so the
        # robot has to round one end; the pillar at (4, 2.5) makes +y the
        # cluttered side. A diff drive holds an arc exactly when v and omega
        # are constant, so this costs no per-step control effort, and chaining
        # arcs (omega = 0 gives a straight) covers any longer route.
        #
        # Why arc length, not time: curvature 1/R fixes the shape and
        # ARC_LENGTH fixes how far along it we go, both independent of speed.
        # One arc rounds the -y end but stops short of crossing x = 6.25;
        # a second segment would be needed to finish (radii that do both in
        # one arc span ~0.05 m, too brittle to tune).
        #
        # Open loop: s += v * dt dead-reckons off our own command and ignores
        # slip. TASK 2's odometry is what closes that gap.
        self.path = {
            'radius': self.ARC_RADIUS,
            'turn_sign': self.TURN_SIGN,
            'length': self.ARC_LENGTH,
        }

        # Commanded arc length so far [m].
        self.distance_travelled = 0.0

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
            'robot_controller started: driving a %.1f m arc of radius %.1f m '
            '(%.0f deg, %s) at %.2f m/s',
            self.ARC_LENGTH, self.ARC_RADIUS,
            math.degrees(self.ARC_LENGTH / self.ARC_RADIUS),
            'right' if self.TURN_SIGN < 0 else 'left', self.LINEAR_SPEED)

    # -----------------------------------------------------------------------
    # TASK 1.2 -- publish a velocity command
    # -----------------------------------------------------------------------
    def send_move_cmd(self):
        """Publish one Twist that moves the robot along self.path."""
        cmd = Twist()

        if self.distance_travelled < self.path['length']:
            # omega = v / R: one radian of heading per R metres travelled.
            cmd.linear.x = self.LINEAR_SPEED
            cmd.angular.z = (
                self.path['turn_sign'] * self.LINEAR_SPEED / self.path['radius'])

            self.distance_travelled += self.LINEAR_SPEED * self.CMD_PERIOD

        # Past ARC_LENGTH, cmd stays zeroed; keep publishing so the plugin
        # never falls back on the last non-zero command.
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
