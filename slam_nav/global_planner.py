import heapq
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from slam_nav import math_utils as mu
class GlobalPlanner(Node):

    def __init__(self):
        super().__init__('global_planner')

        self.declare_parameter('inflation_radius', 0.22)  
        self.declare_parameter('occupied_threshold', 50)   
        self.declare_parameter('allow_unknown', True)      
        self.declare_parameter('smooth_weight_data', 0.5)
        self.declare_parameter('smooth_weight_smooth', 0.3)
        self.declare_parameter('smooth_iterations', 50)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')

        gp = self.get_parameter
        self.inflation_radius = gp('inflation_radius').value
        self.occ_thresh = gp('occupied_threshold').value
        self.allow_unknown = gp('allow_unknown').value
        self.w_data = gp('smooth_weight_data').value
        self.w_smooth = gp('smooth_weight_smooth').value
        self.smooth_iters = gp('smooth_iterations').value
        self.map_frame = gp('map_frame').value
        self.base_frame = gp('base_frame').value

        self.map_msg = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

      
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, 10)
        self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_cb, 10)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)

        self.get_logger().info('A* global planner started.')

  
    def map_cb(self, msg):
        self.map_msg = msg

    def goal_cb(self, goal):
        if self.map_msg is None:
            self.get_logger().warn('No map yet so cannot plan.')
            return

        start = self.robot_cell()
        if start is None:
            return

        info = self.map_msg.info
        gx, gy = goal.pose.position.x, goal.pose.position.y
        goal_cell = mu.world_to_grid(
            gx, gy, info.origin.position.x,
            info.origin.position.y, info.resolution)

        height, width = info.height, info.width
        if not mu.in_bounds(*goal_cell, height, width):
            self.get_logger().warn('Goal is outside the map.')
            return

        grid = self.inflated_grid()

        if self.is_blocked(grid, *start):
            self.get_logger().warn('Robot is inside an inflated obstacle.')
            return
        if self.is_blocked(grid, *goal_cell):
            self.get_logger().warn('Goal is inside an inflated obstacle.')
            return

        cell_path = self.astar(grid, start, goal_cell)
        if cell_path is None:
            self.get_logger().warn('A* found no path to the goal.')
            return

        world_path = self.cells_to_world(cell_path)
        world_path = self.smooth(world_path, grid)
        self.publish_path(world_path)
        self.get_logger().info(
            f'Planned a path with {len(world_path)} waypoints.')

   
    def inflated_grid(self):
        """Return a 2D int array: -1 unknown, 0 free, 100 occupied/inflated."""
        info = self.map_msg.info
        h, w = info.height, info.width
        data = np.array(self.map_msg.data, dtype=np.int16).reshape((h, w))

        inflated = data.copy()
        radius_cells = int(math.ceil(self.inflation_radius / info.resolution))

       
        offsets = []
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                if dr * dr + dc * dc <= radius_cells * radius_cells:
                    offsets.append((dr, dc))

        occ_rows, occ_cols = np.where(data >= self.occ_thresh)
        for r, c in zip(occ_rows, occ_cols):
            for dr, dc in offsets:
                rr, cc = r + dr, c + dc
                if mu.in_bounds(rr, cc, h, w):
                    inflated[rr, cc] = 100
        return inflated

    def is_blocked(self, grid, row, col):
        v = grid[row, col]
        if v >= self.occ_thresh:
            return True
        if v < 0 and not self.allow_unknown:
            return True
        return False

    def robot_cell(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time(),
                timeout=Duration(seconds=0.2))
        except Exception as e:
            self.get_logger().warn(f'Cannot get robot pose: {e}')
            return None
        info = self.map_msg.info
        return mu.world_to_grid(
            tf.transform.translation.x, tf.transform.translation.y,
            info.origin.position.x, info.origin.position.y, info.resolution)

   
    def astar(self, grid, start, goal):
        h, w = grid.shape
       
        moves = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
            (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
        ]

        def heuristic(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        open_heap = []
        counter = 0
        heapq.heappush(open_heap, (0.0, counter, start))
        g_score = {start: 0.0}
        came_from = {}
        closed = set()

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current == goal:
                return self.reconstruct(came_from, current)
            if current in closed:
                continue
            closed.add(current)

            cr, cc = current
            for dr, dc, cost in moves:
                nr, nc = cr + dr, cc + dc
                if not mu.in_bounds(nr, nc, h, w):
                    continue
                if self.is_blocked(grid, nr, nc):
                    continue
               
                if dr != 0 and dc != 0:
                    if self.is_blocked(grid, cr + dr, cc) and \
                       self.is_blocked(grid, cr, cc + dc):
                        continue
                neighbor = (nr, nc)
                if neighbor in closed:
                    continue
                tentative = g_score[current] + cost
                if tentative < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    f = tentative + heuristic(neighbor, goal)
                    counter += 1
                    heapq.heappush(open_heap, (f, counter, neighbor))
        return None

    def reconstruct(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

  
    def cells_to_world(self, cell_path):
        info = self.map_msg.info
        pts = []
        for (r, c) in cell_path:
            x, y = mu.grid_to_world(
                r, c, info.origin.position.x,
                info.origin.position.y, info.resolution)
            pts.append([x, y])
        return np.array(pts)

    def smooth(self, path, grid):
       
        if len(path) < 3:
            return path
        info = self.map_msg.info
        new = path.copy()
        for _ in range(self.smooth_iters):
            for i in range(1, len(path) - 1):
                for j in range(2):
                    proposed = new[i, j]
                    proposed += self.w_data * (path[i, j] - new[i, j])
                    proposed += self.w_smooth * (
                        new[i - 1, j] + new[i + 1, j] - 2.0 * new[i, j])
                    saved = new[i, j]
                    new[i, j] = proposed
                   
                    r, c = mu.world_to_grid(
                        new[i, 0], new[i, 1], info.origin.position.x,
                        info.origin.position.y, info.resolution)
                    if not mu.in_bounds(r, c, grid.shape[0], grid.shape[1]) \
                            or self.is_blocked(grid, r, c):
                        new[i, j] = saved
        return new

   
    def publish_path(self, world_path):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        for (x, y) in world_path:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GlobalPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
