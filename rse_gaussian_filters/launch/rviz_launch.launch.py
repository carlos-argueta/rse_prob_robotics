from launch import LaunchDescription
from launch_ros.actions import Node

from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    urdf_file_path = os.path.join(get_package_share_directory('rse_gaussian_filters'), 'shl_robot.urdf')
    with open(urdf_file_path, 'r') as urdf_file:
        robot_description = urdf_file.read()



    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
    )
    
    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', PathJoinSubstitution([FindPackageShare("rse_gaussian_filters"), "kalman.rviz"]),],
        parameters=[{'use_sim_time': True}]
    )
    
    return LaunchDescription([
        robot_state_publisher,
        rviz2
    ])


