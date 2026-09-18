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
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float64


def stamp_to_sec(stamp):
    """Flatten a builtin_interfaces/Time into float seconds."""
    return stamp.sec + stamp.nanosec * 1e-9


def rotate_by_quat(q, v):
    """Rotate vector v by quaternion q (body frame -> world frame)."""
    # v' = v + w * t + q_vec x t, where t = 2 * (q_vec x v)
    tx = 2.0 * (q.y * v[2] - q.z * v[1])
    ty = 2.0 * (q.z * v[0] - q.x * v[2])
    tz = 2.0 * (q.x * v[1] - q.y * v[0])
    return (v[0] + q.w * tx + q.y * tz - q.z * ty,
            v[1] + q.w * ty + q.z * tx - q.x * tz,
            v[2] + q.w * tz + q.x * ty - q.y * tx)


class RobotController(Node):
    """Drives the robot, tracks its position error, and flags obstacles."""

    # ---- TASK 1.3 tuning constants -----------------------------------------
    # Robot starts at the origin facing +x, square to the barrier (x = 6,
    # spanning y -3..3). The route is a 90 deg pivot to -y, then a half circle
    # left about centre (ARC_RADIUS, 0):
    #     (x, y) = R(1 + cos t), R sin t,   t: pi -> 2pi
    # It leaves (0, 0) heading -y along the near face, bottoms out at (R, -R)
    # past the barrier's -y end, and finishes at (2R, 0) heading +y along the
    # far face. 
    ARC_RADIUS = 4.5       # [m] half-circle radius
    LINEAR_SPEED = 2.0      # [m/s] forward speed along the arc
    PIVOT_RATE = 1.0        # [rad/s] yaw rate of the in-place pivot
    CMD_PERIOD = 0.1        # [s] command period

    # ---- TASK 2.3 constants -------------------------------------------------
    GRAVITY = 9.8           # [m/s^2] SDF default, the world sets no <gravity>

    # ---- TASK 3 constants ---------------------------------------------------
    # gz maps <laser_retro> to point intensity: the poles ship 2000, the
    # barrier 0, so anything well inside that gap separates them.
    OBSTACLE_INTENSITY = 100.0
    # Beams diverge by 0.25 deg, so neighbours on one surface land ~2 cm apart
    # at the ranges we care about. 0.25 m tolerates a few dropped beams without
    # merging separate objects.
    CLUSTER_GAP = 0.25      # [m]
    MIN_CLUSTER_POINTS = 3  # a pole spans tens of beams; fewer is speckle

    def __init__(self):
        super().__init__('robot_controller')

        # ensure we are using sim time to time the path. 
        # --> consisrency across diff. devices running diff. sims
        if not self.get_parameter('use_sim_time').value:
            self.set_parameters(
                [Parameter('use_sim_time', Parameter.Type.BOOL, True)])
            self.get_logger().warn('use_sim_time was off -- forced on')

        # Publish to /error when the position delta exceeds this (TASK 2.3).
        # 1.0m is 1/2 of robot chassis length -- quite an error
        # however note that position from imu is quitee inaccurate
        # due to imu drift and noise :( 
        # so /error will still likely be published to a every loop. 
        self.error_thresh = 1.0

        # ---- TASK 1.2: publisher that drives the robot ---------------------
        self.move_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # note: to test lidar, comment out this line. 
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

        # rationale: 
        # a half circle is super easy to parametrize, and even easier to tune.  
        # since this is dead reckoning and we have to tune it, simple is better. 
        # note: b/c the robot is diffy drive we'll need to rotate at the start. 
        # unfortunately this is not very time efficient but 
        # its better that it works than it not working :) 

        # Route time is measured from the node's first tick, not from the sim
        # clock's zero 
        # Captured on the first callback because with use_sim_time the clock
        # reads 0 until the first /clock message lands.
        self.start_time = None

        # ---- TASK 2.2: subscriber for the robot's 6D pose ------------------
        # note /imu does not give position directly
        # need to double integrate accel. to get pos
        # which is often veryyyyyy noisy and not particularly reliable 
        self.robot_pos_sub = self.create_subscription(
            Imu,
            '/imu',
            self.on_robot_pos,
            qos_profile_sensor_data,
        )

        # ---- TASK 2.3: where the measured-vs-actual error goes -------------
        # actual here is the sim's odometry
        self.odom_sub = self.create_subscription(
            Odometry,
            '/model/vehicle_blue/odometry',
            self.on_odom,
            10,
        )
        self.error_pub = self.create_publisher(Float64, '/error', 10)

        # The robot spawns at the world origin and the IMU reports 
        # orientation relative to its own start, so the two frames 
        # coincide and need no startup offset.
        self.est_pos = [0.0, 0.0, 0.0]
        self.est_vel = [0.0, 0.0, 0.0]
        self.last_imu_stamp = None
        self.latest_odom_pos = None # note: sim's odom might refreshes at a diff time

        # ---- TASK 3: lidar in, filtered obstacles out ----------------------
        # The lidar has a single vertical sample, so this cloud is one flat
        # row of points at the sensor's height -- not a 3D volume.
        #
        self.lidar_sub = self.create_subscription(
            PointCloud2,
            '/lidar/points',
            self.on_lidar,
            qos_profile_sensor_data,
        )
        self.obstacle_pub = self.create_publisher(
            PointCloud2, '/obstacle_cloud', 10)

        self.get_logger().info(
            f'robot_controller started: pivot 90 deg right, then a '
            f'{math.pi * self.ARC_RADIUS:.1f} m half circle of radius '
            f'{self.ARC_RADIUS:.1f} m at {self.LINEAR_SPEED:.2f} m/s')

    # -----------------------------------------------------------------------
    # TASK 1.2 -- publish a velocity command
    # -----------------------------------------------------------------------
    def send_move_cmd(self):
        """Publish one Twist for the current segment of self.path."""
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.start_time is None:
            self.start_time = now
        elapsed = now - self.start_time

        # Walk the segments to find the one this instant falls in. Reading the
        # clock beats counting ticks: the timer can fire late and the segment
        # boundaries still land where the path expects them.
        cmd = Twist()
        for linear, angular, duration in self.path:
            if elapsed < duration:
                cmd.linear.x = linear
                cmd.angular.z = angular
                break
            elapsed -= duration

        self.move_pub.publish(cmd)

    # -----------------------------------------------------------------------
    # TASK 2.3 -- compare reported position against ground truth
    # -----------------------------------------------------------------------
    def on_odom(self, msg):
        """Cache the latest ground truth position."""
        p = msg.pose.pose.position
        self.latest_odom_pos = (p.x, p.y)

    def on_robot_pos(self, msg):
        """Double integrate position from the IMU and publish the error vs truth."""
        stamp = stamp_to_sec(msg.header.stamp)
        if self.last_imu_stamp is None:
            self.last_imu_stamp = stamp
            return

        dt = stamp - self.last_imu_stamp # cannot rely on /imu to pub. at rate of exactly 1 hz 
        self.last_imu_stamp = stamp
        if dt <= 0.0:
            return

        # gz reports specific force, so a stationary IMU reads +g on its z
        # axis. Rotate into the world frame and take gravity back out before
        # anything gets integrated.
        a = msg.linear_acceleration
        acc = list(rotate_by_quat(msg.orientation, (a.x, a.y, a.z)))
        acc[2] -= self.GRAVITY

        # Constant acceleration over the interval. At the sensor's 1 Hz that
        # assumption is coarse, and the resulting drift is what TASK 2 measures.
        for i in range(3):
            self.est_pos[i] += self.est_vel[i] * dt + 0.5 * acc[i] * dt * dt
            self.est_vel[i] += acc[i] * dt

        if self.latest_odom_pos is None:
            return

        # Planar distance: the robot is ground bound, so x-y carries the error
        # that matters and fits the single Float64 the task asks for.
        truth_x, truth_y = self.latest_odom_pos
        delta = math.hypot(self.est_pos[0] - truth_x,
                           self.est_pos[1] - truth_y)
        if delta > self.error_thresh:
            self.error_pub.publish(Float64(data=delta))

    # -----------------------------------------------------------------------
    # TASK 3.3 -- classify a single lidar point
    # -----------------------------------------------------------------------
    def is_obstacle(self, point):
        """Return True if `point` is something we must avoid.

        The barrier is passable -- treat it like dust in the air. The poles are
        not. `point` is an (x, y, z, intensity) tuple in the lidar's frame.

        Only the poles are retroreflective, so intensity is what tells the two
        apart. Nothing about where they sit or how big they are is assumed.
        """
        return point[3] > self.OBSTACLE_INTENSITY

    # -----------------------------------------------------------------------
    # TASK 3.2 -- filter the scan and republish what matters
    # -----------------------------------------------------------------------
    def on_lidar(self, msg):
        """Filter incoming points through is_obstacle and republish."""
        # Pass 1: per point gate.
        hits = []
        for p in point_cloud2.read_points(
                msg,
                field_names=('x', 'y', 'z', 'intensity'),
                skip_nans=True):
            # read_points hands back numpy records, so flatten to plain floats.
            point = (float(p[0]), float(p[1]), float(p[2]), float(p[3]))
            # Beams that hit nothing come back inf, which poisons anything
            # downstream that tries to do arithmetic on the cloud.
            if not all(math.isfinite(v) for v in point[:3]):
                continue
            if self.is_obstacle(point):
                hits.append(point[:3])

        # Pass 2: the scan arrives in angular order and pass 1 preserves it, so
        # one walk down the survivors segments them -- cut a cluster wherever
        # consecutive points jump. No neighbour search needed.
        keep = []
        cluster = []
        for point in hits:
            if cluster and math.dist(point, cluster[-1]) > self.CLUSTER_GAP:
                if len(cluster) >= self.MIN_CLUSTER_POINTS:
                    keep.extend(cluster)
                cluster = []
            cluster.append(point)
        if len(cluster) >= self.MIN_CLUSTER_POINTS:
            keep.extend(cluster)

        self.obstacle_pub.publish(
            point_cloud2.create_cloud_xyz32(msg.header, keep))


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
