import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from slam_nav import math_utils as mu


class LocalPlanner(Node):

    def __init__(self):
        super().__init__('local_planner')

       
        self.declare_parameter('k_att', 1.0)            
        self.declare_parameter('k_rep', 0.6)           
        self.declare_parameter('d_threshold', 0.6)     
        self.declare_parameter('att_clip', 1.0)         
        self.declare_parameter('lookahead_distance', 0.4)  
        self.declare_parameter('goal_tolerance', 0.15) 
        self.declare_parameter('max_linear_vel', 0.22) 
        self.declare_parameter('max_angular_vel', 1.5)  
        self.declare_parameter('k_angular', 1.5)       
        self.declare_parameter('heading_turn_thresh', 1.0)  
        self.declare_parameter('control_period', 0.1)   
        self.declare_parameter('stuck_speed', 0.03)     
        self.declare_parameter('stuck_time', 2.0)       
        self.declare_parameter('escape_time', 1.0)      
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')

        gp = self.get_parameter
        self.k_att = gp('k_att').value
        self.k_rep = gp('k_rep').value
        self.d_thresh = gp('d_threshold').value
        self.att_clip = gp('att_clip').value
        self.lookahead = gp('lookahead_distance').value
        self.goal_tol = gp('goal_tolerance').value
        self.max_lin = gp('max_linear_vel').value
        self.max_ang = gp('max_angular_vel').value
        self.k_ang = gp('k_angular').value
        self.turn_thresh = gp('heading_turn_thresh').value
        self.stuck_speed = gp('stuck_speed').value
        self.stuck_time = gp('stuck_time').value
        self.escape_time = gp('escape_time').value
        self.map_frame = gp('map_frame').value
        self.base_frame = gp('base_frame').value

      
        self.path = []           
        self.scan =None
        self.stuck_since = None
        self.escaping_until= None
        self.escape_dir = 1.0

      
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(Path, '/planned_path', self.path_cb, 10)
        self.create_subscription(
            LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(gp('control_period').value, self.control_loop)

        self.get_logger().info('Potential field local planner started.')

   
    def path_cb(self, msg):
        self.path = [(p.pose.position.x, p.pose.position.y)
                     for p in msg.poses]
        self.stuck_since = None
        self.escaping_until = None
        self.get_logger().info(f'Received path with {len(self.path)} points.')

    def scan_cb(self, msg):
        self.scan = msg

  
    def control_loop(self):
        if not self.path or self.scan is None:
            return

        pose = self.robot_pose()
        if pose is None:
            return
        rx, ry, rth = pose

        gx, gy = self.path[-1]
        if math.hypot(gx - rx, gy - ry) < self.goal_tol:
            self.stop()
            self.path = []
            self.get_logger().info('Goal reached.')
            return

        now = self.now_sec()
        if self.escaping_until is not None:
            if now < self.escaping_until:
                self.publish(0.0, self.escape_dir * self.max_ang * 0.6)
                return
            self.escaping_until = None
            self.stuck_since = None

        lx, ly = self.lookahead_point(rx, ry)
       
        dx, dy = lx - rx, ly - ry
        local_x = math.cos(rth) * dx + math.sin(rth) * dy
        local_y = -math.sin(rth) * dx + math.cos(rth) * dy
        f_att = np.array([local_x, local_y]) * self.k_att
        mag = np.linalg.norm(f_att)
        if mag > self.att_clip:          
            f_att = f_att / mag * self.att_clip

       
        f_rep = self.repulsive_force()

        f_total = f_att + f_rep

        
        force_mag = np.linalg.norm(f_total)
        if force_mag < 1e-6:
            self.handle_stuck(now, 0.0)
            self.publish(0.0, 0.0)
            return

        heading_err = math.atan2(f_total[1], f_total[0])
        angular = mu.clamp(self.k_ang * heading_err, -self.max_ang, self.max_ang)

        if abs(heading_err) > self.turn_thresh:
            linear = 0.0
        else:
            align = math.cos(heading_err)
            linear = mu.clamp(self.max_lin * align, 0.0, self.max_lin)

        self.handle_stuck(now, linear)
        self.publish(linear, angular)

    
    def repulsive_force(self):
       
        scan = self.scan
        ranges = np.asarray(scan.ranges)
        n = len(ranges)
        angles = scan.angle_min + np.arange(n) * scan.angle_increment

        fx = fy = 0.0
        for i in range(n):
            d = ranges[i]
            if not math.isfinite(d) or d <= scan.range_min:
                continue
            if d >= self.d_thresh:
                continue
        
            ox = d * math.cos(angles[i])
            oy = d * math.sin(angles[i])
           
            strength = self.k_rep * (1.0 / d - 1.0 / self.d_thresh) / (d * d)
            
            fx += strength * (-ox / d)
            fy += strength * (-oy / d)
        return np.array([fx, fy])

   
    def lookahead_point(self, rx, ry):
        
        dists = [math.hypot(px - rx, py - ry) for (px, py) in self.path]
        closest = int(np.argmin(dists))

        acc = 0.0
        for i in range(closest, len(self.path) - 1):
            x0, y0 = self.path[i]
            x1, y1 = self.path[i + 1]
            seg = math.hypot(x1 - x0, y1 - y0)
            acc += seg
            if acc >= self.lookahead:
                return x1, y1
        return self.path[-1]

    def handle_stuck(self, now, linear):
        if linear > self.stuck_speed:
            self.stuck_since = None
            return
        if self.stuck_since is None:
            self.stuck_since = now
        elif now - self.stuck_since > self.stuck_time:
            self.escaping_until = now + self.escape_time
            self.escape_dir = 1.0 if np.random.rand() > 0.5 else -1.0
            self.get_logger().warn('Local minimum detected; escaping.')

   
    def robot_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time(),
                timeout=Duration(seconds=0.1))
        except Exception:
            return None
        x = tf.transform.translation.x
        y = tf.transform.translation.y
        yaw = mu.yaw_from_quaternion(tf.transform.rotation)
        return x, y, yaw

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def publish(self, linear, angular):
        cmd = Twist()
        cmd.linear.x = float(linear)
        cmd.angular.z = float(angular)
        self.cmd_pub.publish(cmd)

    def stop(self):
        self.publish(0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = LocalPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
