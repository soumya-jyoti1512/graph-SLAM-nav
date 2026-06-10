import math
import numpy as np
from scipy.spatial import cKDTree


def transform_points(points, x, y, theta):
    
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s],[s, c]])
    return points @ R.T + np.array([x, y])


def icp(source, target, init=(0.0,0.0, 0.0),
        max_iterations=30, tolerance=1e-4,
        max_correspondence_distance=0.5):
   
    src = np.asarray(source, dtype=float)
    tgt = np.asarray(target,dtype=float)
    if len(src) < 3 or len(tgt) < 3:
        return init[0], init[1], init[2], float('inf'), 0

    tree = cKDTree(tgt)
    x, y, theta = init

    for _ in range(max_iterations):
       
        transformed = transform_points(src, x, y, theta)

        distances, indices = tree.query(transformed)

        mask = distances < max_correspondence_distance
        if mask.sum() < 3:
            break    

        src_in = src[mask]                 
        tgt_in = tgt[indices[mask]]        

        src_centroid = src_in.mean(axis=0)
        tgt_centroid = tgt_in.mean(axis=0)
        H = (src_in - src_centroid).T @ (tgt_in - tgt_centroid)  
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
          
            Vt[-1, :] *= -1.0
            R = Vt.T @ U.T
        t = tgt_centroid - R @ src_centroid
        new_x = float(t[0])
        new_y =float(t[1])
        new_theta = math.atan2(R[1, 0], R[0, 0])

       
        delta = abs(new_x - x) + abs(new_y - y) + abs(new_theta - theta)
        x, y, theta = new_x, new_y, new_theta
        if delta < tolerance:
            break

    final = transform_points(src, x, y, theta)
    distances, _ = tree.query(final)
    inlier_mask = distances < max_correspondence_distance
    n_inliers = int(inlier_mask.sum())
    fitness = float(distances[inlier_mask].mean()) \
        if n_inliers > 0 else float('inf')

    return x, y, theta, fitness, n_inliers
