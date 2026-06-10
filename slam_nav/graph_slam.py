import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float64
from tf2_ros import TransformBroadcaster
from slam_nav import math_utils as mu
from slam_nav.icp import icp
from slam_nav.pose_graph import PoseGraph


class GraphSlam(Node):

    def __init__(self):
        super().__init__('graph_slam')

       
        self.declare_parameter('keyframe_trans_threshold', 0.3)  
        self.declare_parameter('keyframe_rot_threshold', 0.3)    
       
        self.declare_parameter('icp_max_correspondence_distance', 0.5) 
        self.declare_parameter('icp_max_iterations', 30)
        self.declare_parameter('icp_min_inliers', 30)
        self.declare_parameter('icp_max_fitness', 0.20)          
        
        self.declare_parameter('loop_radius', 1.5)               
        self.declare_parameter('loop_min_keyframe_gap', 10)
        self.declare_parameter('loop_max_fitness', 0.10)         
        self.declare_parameter('loop_min_inliers', 50)
      
        self.declare_parameter('odom_info', [200.0, 200.0, 500.0])
        self.declare_parameter('icp_info',  [100.0, 100.0, 200.0])
        self.declare_parameter('loop_info', [ 50.0,  50.0, 100.0])
       
        self.declare_parameter('optimize_iterations', 10)
        self.declare_parameter('max_scan_range', 3.3)            
        
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')

        gp = self.get_parameter
        self.kf_trans = gp('keyframe_trans_threshold').value
        self.kf_rot   = gp('keyframe_rot_threshold').value
        self.icp_max_dist= gp('icp_max_correspondence_distance').value
        self.icp_iters = gp('icp_max_iterations').value
        self.icp_min_in = gp('icp_min_inliers').value
        self.icp_max_fit  = gp('icp_max_fitness').value
        self.loop_radius = gp('loop_radius').value
        self.loop_min_gap= gp('loop_min_keyframe_gap').value
        self.loop_max_fit = gp('loop_max_fitness').value
        self.loop_min_in  = gp('loop_min_inliers').value
        self.odom_info  = np.diag(gp('odom_info').value)
        self.icp_info   = np.diag(gp('icp_info').value)
        self.loop_info = np.diag(gp('loop_info').value)
        self.opt_iters = gp('optimize_iterations').value
        self.max_range = gp('max_scan_range').value
        self.map_frame = gp('map_frame').value
        self.odom_frame= gp('odom_frame').value
        self.base_frame = gp('base_frame').value

       
        self.pg = PoseGraph(backend='auto')
        self.get_logger().info(f'PoseGraph backend: {self.pg.backend}')

        self.keyframe_scans = {}      
        self.next_kf_id = 0
        self.last_kf_id = None
        self.last_kf_pose = (0.0, 0.0, 0.0)
        self.last_kf_odom = None

        self.curr_odom = None
        self.curr_pose = (0.0, 0.0, 0.0)

        self.create_subscription(Odometry, '/odom', self.odom_cb, 20)
        self.create_subscription(
            LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)

        self.pose_pub = self.create_publisher(PoseStamped, '/slam/pose', 10)
        self.kf_pub = self.create_publisher(MarkerArray, '/slam/keyframes', 10)
        self.cov_pub = self.create_publisher(Float64, '/slam/cov_trace', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info('Graph SLAM node started.')

   
    def odom_cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        th = mu.yaw_from_quaternion(msg.pose.pose.orientation)
        self.curr_odom = (x, y, th)

    def scan_cb(self, msg):
        if self.curr_odom is None:
            return
        points = self._scan_to_points(msg)
        if len(points) < 30:
            return

        if self.last_kf_id is None:
            self._add_keyframe(0.0, 0.0, 0.0, points, parent_id=None)
            self.last_kf_odom = self.curr_odom
            self._publish_all(msg.header.stamp)
            return

        odom_rel = self._odom_relative()

        last_scan = self.keyframe_scans[self.last_kf_id]
        x, y, th, fit, n_in = icp(
            points, last_scan, init=odom_rel,
            max_iterations=self.icp_iters,
            max_correspondence_distance=self.icp_max_dist)

        icp_ok = n_in >= self.icp_min_in and fit < self.icp_max_fit
        if icp_ok:
            rel = (x, y, th)
            edge_info = self.icp_info
        else:
        
            rel = odom_rel
            edge_info = self.odom_info

        self.curr_pose = self._compose(self.last_kf_pose, rel)

        if math.hypot(rel[0], rel[1]) > self.kf_trans or abs(rel[2]) > self.kf_rot:
            self._add_keyframe(rel[0], rel[1], rel[2], points,
                               parent_id=self.last_kf_id, edge_info=edge_info)
            self.last_kf_odom = self.curr_odom

            n_loops = self._check_loop_closures()
            if n_loops > 0:
              
                self.pg.optimize(iterations=self.opt_iters)
                self.last_kf_pose = self.pg.get_pose(self.last_kf_id)
                self.curr_pose = self.last_kf_pose

        self._publish_all(msg.header.stamp)

  
    def _scan_to_points(self, scan):
        ranges = np.asarray(scan.ranges)
        n = len(ranges)
        angles = scan.angle_min + np.arange(n) * scan.angle_increment
        max_r = min(self.max_range, scan.range_max)
        mask = np.isfinite(ranges) & \
            (ranges > scan.range_min) & (ranges < max_r)
        rs = ranges[mask]
        a = angles[mask]
        return np.column_stack([rs * np.cos(a), rs * np.sin(a)])

    def _odom_relative(self):
       
        x0, y0, t0 = self.last_kf_odom
        x1, y1, t1= self.curr_odom
        dx = x1 - x0
        dy = y1 - y0
        c, s = math.cos(t0), math.sin(t0)
        return (c * dx + s * dy,
                -s * dx + c * dy,
                mu.angle_diff(t1, t0))

    @staticmethod
    def _compose(a, b):
       
        ax, ay, at = a
        bx, by, bt =b
        c, s = math.cos(at), math.sin(at)
        return (ax + c * bx - s * by,
                ay + s * bx + c * by,
                mu.normalize_angle(at + bt))

    @staticmethod
    def _relative(a, b):
       
        ax, ay, at = a
        bx, by, bt = b
        dx = bx - ax
        dy = by - ay
        c, s = math.cos(at), math.sin(at)
        return (c * dx + s * dy,
                -s * dx + c * dy,
                mu.angle_diff(bt, at))

    def _add_keyframe(self, dx, dy, dth, points, parent_id, edge_info=None):
        new_id = self.next_kf_id
        self.next_kf_id += 1
        if parent_id is None:
           
            new_pose = (dx, dy, dth)
            self.pg.add_node(new_id, *new_pose, fixed=True)
        else:
            parent_pose = self.pg.get_pose(parent_id)
            new_pose = self._compose(parent_pose, (dx, dy, dth))
            self.pg.add_node(new_id, *new_pose)
            info = edge_info if edge_info is not None else self.icp_info
            self.pg.add_edge(parent_id, new_id, dx, dy, dth, information=info)
        self.keyframe_scans[new_id] = points
        self.last_kf_id = new_id
        self.last_kf_pose = new_pose

    def _check_loop_closures(self):
       
        if self.last_kf_id is None or self.last_kf_id < self.loop_min_gap:
            return 0
        new_pose = self.pg.get_pose(self.last_kf_id)
        new_scan = self.keyframe_scans[self.last_kf_id]
        added = 0

        max_candidate = self.last_kf_id - self.loop_min_gap
        for kf_id in range(max_candidate + 1):
            old_pose = self.pg.get_pose(kf_id)
            d = math.hypot(new_pose[0] - old_pose[0],
                           new_pose[1] - old_pose[1])
            if d > self.loop_radius:
                continue
           
            init = self._relative(old_pose, new_pose)
            old_scan = self.keyframe_scans[kf_id]
            x, y, th, fit, n_in = icp(
                new_scan, old_scan, init=init,
                max_iterations=self.icp_iters,
                max_correspondence_distance=self.icp_max_dist)
            if n_in >= self.loop_min_in and fit < self.loop_max_fit:
                self.pg.add_edge(kf_id, self.last_kf_id,
                                 x, y, th, information=self.loop_info)
                added += 1
                self.get_logger().info(
                    f'Loop closure: kf {kf_id} <-> kf {self.last_kf_id}  '
                    f'd={d:.2f}m  fit={fit:.3f}  inliers={n_in}')
        return added

   
    def _publish_all(self, stamp):
        self._publish_pose(stamp)
        self._publish_tf(stamp)
        self._publish_keyframes(stamp)
        self._publish_chi2()

    def _publish_pose(self, stamp):
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.map_frame
        msg.pose.position.x =float(self.curr_pose[0])
        msg.pose.position.y = float(self.curr_pose[1])
        qx, qy, qz, qw = mu.quaternion_from_yaw(self.curr_pose[2])
        msg.pose.orientation.x= qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z= qz
        msg.pose.orientation.w = qw
        self.pose_pub.publish(msg)

    def _publish_tf(self, stamp):
        if self.curr_odom is None:
            return
        T_mb = mu.pose_to_matrix(*self.curr_pose)
        T_ob = mu.pose_to_matrix(*self.curr_odom)
        T_mo = T_mb @ np.linalg.inv(T_ob)
        x_mo = float(T_mo[0, 2])
        y_mo = float(T_mo[1, 2])
        th_mo = math.atan2(T_mo[1, 0], T_mo[0, 0])

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.odom_frame
        t.transform.translation.x = x_mo
        t.transform.translation.y = y_mo
        qx, qy, qz, qw = mu.quaternion_from_yaw(th_mo)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t)

    def _publish_keyframes(self, stamp):
        arr = MarkerArray()
        for kf_id, pose in self.pg.get_all_poses().items():
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = self.map_frame
            m.ns = 'keyframes'
            m.id = int(kf_id)
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(pose[0])
            m.pose.position.y = float(pose[1])
            m.pose.position.z = 0.05
            m.pose.orientation.w = 1.0
            m.scale.x = 0.10
            m.scale.y = 0.10
            m.scale.z = 0.10
            m.color.r = 0.0
            m.color.g = 0.8
            m.color.b = 1.0
            m.color.a = 1.0
            arr.markers.append(m)
        self.kf_pub.publish(arr)

    def _publish_chi2(self):
        msg = Float64()
        msg.data =float(self.pg.chi2())
        self.cov_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GraphSlam()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
