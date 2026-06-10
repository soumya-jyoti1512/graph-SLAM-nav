import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('slam_nav')
    slam_params = os.path.join(share, 'config', 'slam_params.yaml')
    nav_params = os.path.join(share, 'config', 'nav_params.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    sim = {'use_sim_time': use_sim_time}

    graph_slam = Node(
        package='slam_nav', executable='graph_slam', name='graph_slam',
        parameters=[slam_params, sim], output='screen',
    )
    grid_mapper = Node(
        package='slam_nav', executable='grid_mapper', name='grid_mapper',
        parameters=[nav_params, sim], output='screen',
    )
    global_planner = Node(
        package='slam_nav', executable='global_planner', name='global_planner',
        parameters=[nav_params, sim], output='screen',
    )
    local_planner = Node(
        package='slam_nav', executable='local_planner', name='local_planner',
        parameters=[nav_params, sim], output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        graph_slam,
        grid_mapper,
        global_planner,
        local_planner,
    ])
