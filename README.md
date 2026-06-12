# Indoor Autonomous Navigation via Custom Graph SLAM Stack

> A ROS 2 implementation of 2D LiDAR SLAM and navigation, built from scratch with an ICP scan-matching front end, a pose-graph back end with loop closure, and an A\* + potential field navigation stack.

---

## Overview

This project implements a complete SLAM and navigation pipeline from scratch. Every component is written directly so the full data flow is visible end to end:

- **SLAM front end** -> point-to-point ICP scan matching between consecutive keyframes.
- **SLAM back end** -> pose graph with relative-pose edges, loop closure detection, and optimization via g2o (or a built-in Gauss-Newton solver as a fallback).
- **Mapping** -> log-odds occupancy grid built with Bresenham ray casting.
- **Navigation** -> A\* global planner with obstacle inflation and gradient-descent path smoothing, plus a potential field local planner.


---

## This project targets two goals:

- **Foundational SLAM component** - implement scan matching, pose-graph optimization, and loop closure from first principles, not as library calls
- **End-to-end integration** - connect the SLAM output cleanly to a working navigation stack on a TurtleBot3 in a Gazebo.



---

## Sensor Suite

| Sensor    | Modality       |
|-----------|----------------
| 2D LiDAR  | Geometric      |
| Wheel Odometry | Proprioceptive |

---

## System Architecture

<img width="1083" height="1646" alt="Image" src="https://github.com/user-attachments/assets/28d85edb-7542-4cf6-83e5-31cab5eefdca" />

---

## Methodology

### Front End - ICP Scan Matching

The ICP front end aligns each new LiDAR scan to the most recent keyframe scan using point-to-point Iterative Closest Point. A KD-tree accelerates nearest-neighbor correspondence search, distance-based outlier rejection removes spurious matches, and a closed-form SVD-based solve recovers the rigid transform between scans.

Initialization handling: odometry provides the initial pose guess for each ICP call, keeping the optimization in the basin of convergence even when relative motion between scans is non-trivial. When ICP fails its quality gate (too few inliers or residual too high), the system falls back cleanly to the odometry estimate with a downgraded edge information matrix, so the pipeline never stalls.

### Back End - Pose Graph + Loop Closure

Each keyframe is a node in the pose graph. ICP results become relative-pose edges between consecutive keyframes. When a new keyframe is added, the system searches past keyframes within a Euclidean radius for loop closure candidates. ICP between the candidate's scan and the current scan, initialized from the current graph estimate, verifies the match. Accepted loops add a graph edge and trigger optimization.

Two interchangeable optimization backends are supported:

- g2o (preferred, auto-detected) 
- Built-in Gauss-Newton (always available) 


### Mapping and Navigation

Mapping uses a log-odds occupancy grid updated via Bresenham ray casting, with max-range fall-off handled separately from real hits to prevent free-space rays from being mistaken for obstacles.

Global planning runs A\* on the inflated grid with a Euclidean heuristic and corner-cutting prevention. The raw grid path is then smoothed via gradient descent with obstacle rejection on each candidate update.

Local control uses an artificial potential field: attractive force toward a moving lookahead point on the path, repulsion from live LiDAR points, turn-in-place when badly misaligned, and a timed escape rotation to break out of local minima.

---

## Evaluation Scenarios

The pipeline is evaluated across four conditions:

| Condition | Description |
|-----------|-------------|
| Closed-loop traversal | Robot drives a loop and returns to start tests loop closure |
| Open-trajectory navigation | Point-to-point goals without loop closure opportunities |
| Noisy odometry | Injected scale and bias errors to stress the front end |
| Sparse-feature regions | Stretches of bare wall to stress scan matching robustness |

The environment is the standard `turtlebot3_world`  which provides a mix of corners, walls, and isolated obstacles for scan matching.

---

## Results

| Metric | Value |
|--------|-------|
| Loop closure detection | ICP fitness 0.029 m, 196 inliers, accepted |
| Graph chi² before / after optimization | 27.6 → 0.68 |
| Final endpoint error after optimization | 1.2 cm |
| Mean trajectory error (26 keyframes) | 11.0 cm |
| Pure odometry endpoint error (baseline) | 18.3 cm |

https://github.com/user-attachments/assets/573e025c-0fb9-4084-9609-da0c1c11628b 

https://github.com/user-attachments/assets/021eb50f-ae72-4bcb-bdb9-2d678472e673


---

## Tech Stack

| Category    | Tools |
|-------------|-------|
| Languages   | Python |
| Middleware  | ROS 2 Jazzy |
| Simulation  | Gazebo Harmonic, TurtleBot3 |
| Optimization | g2o (with built-in Gauss-Newton fallback) |
| Math/Linear Algebra | NumPy, SciPy |
| Visualization | RViz2, Matplotlib |

---



## Build and Run

**Requirements:** Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic, TurtleBot3 packages, Python 3 with `numpy` and `scipy`. Optional: `pip install g2o-python` for the g2o backend (system falls back to built-in Gauss-Newton if unavailable).

```bash
cd ~/ros2_ws/src
git clone https://github.com/soumya-jyoti1512/slam-nav.git
cd ~/ros2_ws
colcon build --packages-select slam_nav
source install/setup.bash
```

```bash
# Terminal 1 - simulation
export TURTLEBOT3_MODEL=waffle
ros2 launch slam_nav simulation.launch.py

# Terminal 2 - SLAM + navigation
ros2 launch slam_nav slam_nav.launch.py
```

Send a goal via the **2D Goal Pose** button in RViz, or publish to `/goal_pose` directly. Drive the robot manually first (`ros2 run turtlebot3_teleop teleop_keyboard`) so the map builds before navigating.

---

