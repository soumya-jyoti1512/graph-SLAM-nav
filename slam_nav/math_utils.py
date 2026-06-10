import math
import numpy as np


def normalize_angle(angle):
   
    return math.atan2(math.sin(angle), math.cos(angle))


def angle_diff(a, b):
    
    return normalize_angle(a -b)


def euclidean_distance(p1, p2):
    
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def yaw_from_quaternion(q):
   
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp,cosy_cosp)


def quaternion_from_yaw(yaw):
   
    half = yaw * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


def pose_to_matrix(x, y, theta):
    
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [c, -s, x],
        [s,  c, y],
        [0,  0, 1],
    ])


def transform_point(matrix, px, py):
  
    p = np.array([px, py, 1.0])
    out = matrix @ p
    return out[0], out[1]


def world_to_grid(x, y, origin_x, origin_y, resolution):
   
    col = int((x - origin_x) / resolution)
    row = int((y - origin_y) /resolution)
    return row, col


def grid_to_world(row, col, origin_x, origin_y, resolution):
   
    x = origin_x + (col + 0.5) * resolution
    y = origin_y + (row + 0.5) * resolution
    return x, y


def in_bounds(row, col, height, width):
   
    return 0 <= row < height and 0 <= col < width


def bresenham(r0, c0, r1, c1):
    
    cells= []
    dr = abs(r1 - r0)
    dc = abs(c1 -c0)
    sr = 1 if r0< r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr
    r, c = r0, c0
    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr
    return cells


def clamp(value, low, high):
    
    return max(low, min(high, value))
