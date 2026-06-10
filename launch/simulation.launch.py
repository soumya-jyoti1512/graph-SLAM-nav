import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    slam_nav_share = get_package_share_directory('slam_nav')
    tb3_gazebo_share = get_package_share_directory('turtlebot3_gazebo')

    rviz_config = os.path.join(slam_nav_share, 'slam_nav.rviz')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    set_model = SetEnvironmentVariable('TURTLEBOT3_MODEL', 'waffle')

    world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            tb3_gazebo_share, 'launch', 'turtlebot3_world.launch.py')),
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        set_model,
        world,
        rviz,
    ])
