import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from slam_nav import math_utils as mu


class GridMapper(Node):

    def __init__(self):
        super().__init__('grid_mapper')

        
        self.declare_parameter('resolution', 0.05)   
        self.declare_parameter('width', 400)           
        self.declare_parameter('height', 400)         
        self.declare_parameter('log_occ', 0.85)       
        self.declare_parameter('log_free', 0.40)      
        self.declare_parameter('log_min', -5.0)       
        self.declare_parameter('log_max', 5.0)
        self.declare_parameter('max_range', 3.3)      
        self.declare_parameter('publish_period', 0.5) 
        self.declare_parameter('map_frame', 'map')

        gp = self.get_parameter
        self.res = gp('resolution').value
        self.width = gp('width').value
        self.height = gp('height').value
        self.l_occ = gp('log_occ').value
        self.l_free = gp('log_free').value
        self.l_min = gp('log_min').value
        self.l_max = gp('log_max').value
        self.max_range = gp('max_range').value
        self.map_frame = gp('map_frame').value

        self.origin_x = -(self.width * self.res) / 2.0
        self.origin_y = -(self.height * self.res) / 2.0

        self.grid = np.zeros((self.height, self.width), dtype=np.float32)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)
        self.create_timer(gp('publish_period').value, self.publish_map)

        self.get_logger().info('Occupancy grid mapper started.')

   
    def scan_cb(self, scan):
       
        pose = self.lookup_pose(scan.header.frame_id, scan.header.stamp)
        if pose is None:
            return
        rx, ry, rth = pose

        r_row, r_col = mu.world_to_grid(
            rx, ry, self.origin_x, self.origin_y, self.res)
        if not mu.in_bounds(r_row, r_col, self.height, self.width):
            return

        ranges = np.asarray(scan.ranges)
        n = len(ranges)
        angles = scan.angle_min + np.arange(n) * scan.angle_increment

        for i in range(n):
            r = ranges[i]
            hit = True
           
            if not math.isfinite(r) or r <= scan.range_min:
                continue
           
            if r >= min(self.max_range, scan.range_max):
                r = min(self.max_range, scan.range_max)
                hit = False

            ex = rx + r * math.cos(rth + angles[i])
            ey = ry + r * math.sin(rth + angles[i])
            e_row, e_col = mu.world_to_grid(
                ex, ey, self.origin_x, self.origin_y, self.res)

            self.cast_ray(r_row, r_col, e_row, e_col, hit)

    def cast_ray(self, r0, c0, r1, c1, hit):
        """Mark cells along a beam free, and the endpoint occupied if it was
        a real hit (not a max-range fall-off)."""
        cells = mu.bresenham(r0, c0, r1, c1)
       
        for (rr, cc) in cells[:-1]:
            if mu.in_bounds(rr, cc, self.height, self.width):
                self.grid[rr, cc] = mu.clamp(
                    self.grid[rr, cc] - self.l_free, self.l_min, self.l_max)
       
        er, ec = cells[-1]
        if mu.in_bounds(er, ec, self.height, self.width):
            if hit:
                self.grid[er, ec] = mu.clamp(
                    self.grid[er, ec] + self.l_occ, self.l_min, self.l_max)
            else:
                self.grid[er, ec] = mu.clamp(
                    self.grid[er, ec] - self.l_free, self.l_min, self.l_max)

   
    def lookup_pose(self, laser_frame, stamp):
       
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, laser_frame, stamp,
                timeout=Duration(seconds=0.1))
        except Exception:
            
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.map_frame, laser_frame, rclpy.time.Time())
            except Exception as e:
                self.get_logger().warn(
                    f'TF lookup failed: {e}', throttle_duration_sec=2.0)
                return None
        t = tf.transform.translation
        yaw = mu.yaw_from_quaternion(tf.transform.rotation)
        return t.x, t.y, yaw

   
    def publish_map(self):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.info.resolution = self.res
        msg.info.width = self.width
        msg.info.height = self.height
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.orientation.w = 1.0

       
        prob = 1.0 - 1.0 / (1.0 + np.exp(self.grid))
        occ = (prob * 100.0).astype(np.int8)
        occ[self.grid == 0.0] = -1           
        msg.data = occ.flatten().tolist()
        self.map_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GridMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
