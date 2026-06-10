import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'slam_nav'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch files
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        # Config files
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        # RViz config
        (os.path.join('share', package_name), glob('*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    description='Custom 2D LiDAR SLAM and navigation stack for ROS2 Jazzy.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'graph_slam = slam_nav.graph_slam:main',
            'grid_mapper = slam_nav.grid_mapper:main',
            'global_planner = slam_nav.global_planner:main',
            'local_planner = slam_nav.local_planner:main',
            'eval_logger = slam_nav.eval_logger:main',
        ],
    },
)
