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
    # ARC_LENGTH is the knob to tune: how far the robot drives along the arc,
    # in metres, and so how far around the barrier it gets. The sweep angle is
    # ARC_LENGTH / ARC_RADIUS = 9.0 / 6.0 = 1.5 rad (~86 deg).
    #
    # Geometry the defaults come from (robot starts at the origin facing +x,
    # turn centre is therefore at (0, -R)): the arc is x = R sin(theta),
    # y = -R(1 - cos(theta)). At the barrier's near face, x = 5.75, that puts
    # the robot at y = -4.3 -- about 1.3 m clear of the barrier's corner at
    # y = -3, comfortable for a chassis half-width of 0.6 m. The arc ends at
    # roughly (6.0, -5.6): around the -y end of the barrier, but still on the
    # near side of the barrier plane. Getting all the way past and back onto
    # the x axis wants a second segment chained on (see send_move_cmd) -- a
    # single arc that both clears the corner and crosses x = 6.25 only exists
    # in a very narrow band of radii, which is a poor thing to tune against.
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
        # The diff-drive plugin in model.sdf subscribes to /cmd_vel, and the
        # parameter_bridge in sim.launch.py forwards ROS Twist -> gz Twist on
        # that same name. Depth 10, reliable (the default): these are commands,
        # not sensor samples, so we do not want them silently dropped.
        self.move_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        #
        # Then drive it on a timer:
        self.move_timer = self.create_timer(self.CMD_PERIOD, self.send_move_cmd)

        # ---- TASK 1.3: the path you chose ----------------------------------
        # Path: a single constant-curvature arc, driven open loop.
        #
        # WHY THIS PATH. The robot starts at the origin facing +x. The barrier
        # (`wall_front`) sits at x = 6 and spans y = -3..3, so driving straight
        # down +x runs into it; the robot has to leave the x axis and come back
        # around one end. The pillar sits at (4, +2.5), so the +y end of the
        # wall is the cluttered side and we swing toward -y instead.
        #
        # WHY A CONSTANT-CURVATURE ARC. A differential drive holds a circular
        # arc exactly when v and omega are both constant, so this path is the
        # one a diff drive can track with no per-step control effort at all --
        # two numbers held flat for the whole run. It is also the smallest
        # useful primitive: longer routes are just arcs chained together (an
        # arc with omega = 0 is a straight line), so self.path generalizes to a
        # list of these segments later without changing the shape of the code.
        #
        # HOW IT IS REPRESENTED. By arc length, not by time or by waypoints.
        # The path is (radius, direction, length): curvature kappa = 1/R fixes
        # the shape, and ARC_LENGTH fixes how far along that shape we go. Arc
        # length is the natural parameter for a curve -- it is a geometric
        # property of the path, independent of how fast we choose to drive it,
        # so retuning LINEAR_SPEED does not change where the robot ends up.
        # Time only enters when we convert: t_total = ARC_LENGTH / LINEAR_SPEED.
        #
        # This is deliberately open loop (TASK 1 has no feedback). We integrate
        # the *commanded* distance, s += v * dt, which is dead reckoning off our
        # own command -- it ignores wheel slip and the actuator's response, so
        # real and commanded travel will drift apart. Closing that gap is what
        # the odometry in TASK 2 is for.
        self.path = {
            'radius': self.ARC_RADIUS,
            'turn_sign': self.TURN_SIGN,
            'length': self.ARC_LENGTH,
        }

        # Commanded arc length travelled so far, in metres.
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
        """Publish one Twist that moves the robot along self.path.

        The arc is constant curvature, so the command itself never changes:
        every tick publishes the same (v, omega) until the commanded arc length
        is used up, and zeros after that. All the shaping lives in the
        constants, not in this function -- which is also why chaining segments
        later is just a matter of popping the next (radius, sign, length) off
        self.path instead of rewriting this logic.
        """
        cmd = Twist()

        if self.distance_travelled < self.path['length']:
            # omega = v / R is the exact relation that makes a unicycle
            # (equivalently, a differential drive) trace a circle of radius R:
            # the body turns through one radian for every R metres travelled.
            # The sign picks the direction of the turn.
            cmd.linear.x = self.LINEAR_SPEED
            cmd.angular.z = (
                self.path['turn_sign'] * self.LINEAR_SPEED / self.path['radius'])

            # Dead reckoning off our own command: with no feedback, the
            # commanded speed is the only estimate of travel we have.
            self.distance_travelled += self.LINEAR_SPEED * self.CMD_PERIOD
        else:
            # Twist() is already all zeros; publishing it explicitly stops the
            # robot. We keep publishing so the diff-drive plugin does not fall
            # back on its last non-zero command if it has a command timeout.
            pass

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
