import csv
import os
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Float64
from slam_nav import math_utils as mu


class EvalLogger(Node):

    def __init__(self):
        super().__init__('eval_logger')

        self.declare_parameter('ground_truth_topic', '/ground_truth')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('slam_topic', '/slam/pose')
        self.declare_parameter('cov_topic', '/slam/cov_trace')
        self.declare_parameter('landmark_topic', '/slam/landmarks')
        self.declare_parameter('output_dir', '/tmp/slam_eval')
        self.declare_parameter('flush_period', 5.0)

        gp = self.get_parameter
        self.out_dir = gp('output_dir').value
        os.makedirs(self.out_dir, exist_ok=True)

       
        self.traj = []  
        self.cov = []    
        self.lms = []   

        self.last_gt = None
        self.last_odom = None
        self.start_t = None

        self.create_subscription(
            Odometry, gp('ground_truth_topic').value, self.gt_cb, 20)
        self.create_subscription(
            Odometry, gp('odom_topic').value, self.odom_cb, 20)
        self.create_subscription(
            PoseStamped, gp('slam_topic').value, self.slam_cb, 20)
        self.create_subscription(
            Float64, gp('cov_topic').value, self.cov_cb, 20)
        self.create_subscription(
            MarkerArray, gp('landmark_topic').value, self.lm_cb, 10)

        self.create_timer(gp('flush_period').value, self.flush)
        self.get_logger().info(
            f'Eval logger started. Writing to {self.out_dir}')

    def t_now(self):
        t = self.get_clock().now().nanoseconds * 1e-9
        if self.start_t is None:
            self.start_t = t
        return t - self.start_t

    def gt_cb(self, msg):
        self.last_gt = self.pose_from_odom(msg)

    def odom_cb(self, msg):
        self.last_odom = self.pose_from_odom(msg)

    def slam_cb(self, msg):
  
        t = self.t_now()
        sx = msg.pose.position.x
        sy = msg.pose.position.y
        syaw = mu.yaw_from_quaternion(msg.pose.orientation)
        gt = self.last_gt if self.last_gt else (float('nan'),) * 3
        od = self.last_odom if self.last_odom else (float('nan'),) * 3
        self.traj.append([t, gt[0], gt[1], gt[2],
                          od[0], od[1], od[2], sx, sy, syaw])

    def cov_cb(self, msg):
        self.cov.append([self.t_now(), msg.data])

    def lm_cb(self, msg):
        t = self.t_now()
        for m in msg.markers:
            self.lms.append([t, m.id, m.pose.position.x, m.pose.position.y])

    @staticmethod
    def pose_from_odom(msg):
        return (msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                mu.yaw_from_quaternion(msg.pose.pose.orientation))

    def flush(self):
        self.write(os.path.join(self.out_dir, 'trajectory.csv'),
                   ['t', 'gt_x', 'gt_y', 'gt_yaw', 'odom_x', 'odom_y',
                    'odom_yaw', 'slam_x', 'slam_y', 'slam_yaw'], self.traj)
        self.write(os.path.join(self.out_dir, 'covariance.csv'),
                   ['t', 'cov_trace'], self.cov)
        self.write(os.path.join(self.out_dir, 'landmarks.csv'),
                   ['t', 'landmark_id', 'x', 'y'], self.lms)

    @staticmethod
    def write(path, header, rows):
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)


def main(args=None):
    rclpy.init(args=args)
    node = EvalLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.flush()
        node.get_logger().info('Final CSVs written.')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
