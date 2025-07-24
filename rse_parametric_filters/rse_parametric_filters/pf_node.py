# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.


from rclpy.node import Node

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry

import numpy as np

from rse_common_utils.sensor_utils import odom_to_pose2D, get_normalized_pose2D, rotate_pose2D, get_yaw_from_quaternion
from rse_common_utils.helper_utils import normalize_angle
from rse_common_utils.visualization import Visualizer

from .filters.pf import ParticleFilter

class ParticleFilterBaseNode(Node):
    def __init__(self, pf):
        super().__init__('particle_filter_node')

        self.odom_gt_subscription = self.create_subscription(
            Odometry,
            'odom',  # 'wheel_odom',  # Ground Truth
            self.odom_gt_callback,
            10)

        self.odom_raw_subscription = self.create_subscription(
            Odometry,
            'odom_raw',  # 'odom_raw',         # For controls
            self.odom_raw_callback,
            10)

        self.imu_subscriber = self.create_subscription(
            Imu,
            '/imu/data_raw',  # '/imu',
            self.imu_callback,
            10)

        self.pf = pf
        
        self.visualizer = Visualizer()

        # Create a ROS 2 timer for the visualizer updates
        self.visualizer_timer = self.create_timer(0.1, self.update_visualizer)

        self.mu = None
        self.Sigma = None
        self.u = None
        self.z = None
        self.prev_time = None  # previous prediction time, used to compute the delta_t

        # Variables to normalize the pose (always start at the origin)
        self.initial_pose = None
        self.normalized_pose = (0.0, 0.0, 0.0) 

        # IMU data
        self.initial_imu_theta = None
        self.normalized_imu_theta = 0.0
        self.imu_w = 0.0
        self.imu_a_x = 0.0
        self.imu_a_y = 0.0

        self.prev_normalized_pose = (0.0, 0.0, 0.0)
        self.prev_pose_set = None 

        self.initial_gt_pose = None
        self.normalized_gt_pose = (0.0, 0.0, 0.0)

        self.odom_count = 0
        self.odom_skip = 30

        print("PF ready!")

    def update_visualizer(self):
        # Call the visualizer update asynchronously
        if self.mu is not None and self.Sigma is not None:
            if self.z is not None:
                self.visualizer.update(self.normalized_gt_pose, self.mu, self.Sigma, self.z, step="update")
            else:
                self.visualizer.update(self.normalized_gt_pose, self.mu, self.Sigma, step="predict")

    def odom_raw_callback(self, msg):

        # Set the initial pose
        if not self.initial_pose:
            initial_pose = odom_to_pose2D(msg)  

            self.initial_pose = rotate_pose2D(initial_pose, -90)

        # Get and normalize the pose
        current_pose = odom_to_pose2D(msg)
        rotated_pose = rotate_pose2D(current_pose, -90)        
        self.normalized_pose = np.array(get_normalized_pose2D(self.initial_pose, rotated_pose))

        self.control = msg.twist.twist

        # Get the control inputs for velocity model
        self.set_control()

        # Compute dt
        curr_time = self.get_clock().now().nanoseconds
        if self.prev_time:
            dt = (curr_time - self.prev_time) / 1e9  # Convert nanoseconds to seconds
        else:
            dt = 0.01

        self.mu, self.Sigma = self.pf.predict(self.u, dt)
        
        self.prev_time = curr_time

        self.set_observation()

        self.odom_count += 1
        if self.odom_count % self.odom_skip == 0:
            self.mu, self.Sigma = self.pf.update(self.z, dt)
            
        print(f"mu: {self.mu}, Sigma: {self.Sigma}")
       
        self.prev_normalized_pose = self.normalized_pose

    def imu_callback(self, msg):

        self.imu_msg = msg

        # Extract the linear acceleration data from the IMU message
        self.imu_a_x = msg.linear_acceleration.x
        self.imu_a_y = msg.linear_acceleration.y

        # Extract the angular velocity data from the IMU message
        self.imu_w = msg.angular_velocity.z

        # Compute the yaw from the IMU quaternion
        imu_theta = get_yaw_from_quaternion(msg.orientation)

        # Calculate the fake theta
        if not self.initial_imu_theta:
            self.initial_imu_theta = imu_theta
        else:
            # Calculate the difference in yaw
            delta_theta = imu_theta - self.initial_imu_theta

            # Unwrap the delta_yaw to avoid issues with angles wrapping around at 2*pi radians
            self.normalized_imu_theta = normalize_angle(delta_theta)

    def odom_gt_callback(self, msg):
        # Set the initial pose
        if not self.initial_gt_pose:

            initial_pose = odom_to_pose2D(msg)  

            self.initial_gt_pose = initial_pose

        # Get and normalize the pose
        current_pose = odom_to_pose2D(msg)
        self.normalized_gt_pose = np.array(get_normalized_pose2D(self.initial_gt_pose, current_pose))

    def set_control(self):
        raise NotImplementedError("This function has to be implemented by a child class")

    def set_observation(self):
        raise NotImplementedError("This function has to be implemented by a child class")


class ParticleFilterNode(ParticleFilterBaseNode):
    def set_control(self):
        self.u = np.asarray([self.control.linear.x + 0.1, self.control.angular.z])

        # Get the control inputs for odometry model
        ''' if self.prev_pose_set:
            self.u = np.asarray([self.prev_normalized_pose, self.normalized_pose])
        else:
            self.prev_normalized_pose = self.normalized_pose
            self.prev_pose_set = True
            return
        '''

    def set_observation(self):
        self.z = self.normalized_pose


class ParticleFilterFusionNode(ParticleFilterNode):

    def set_observation(self):
        self.z = np.array([[self.normalized_pose[0]], [self.normalized_pose[1]], [self.normalized_pose[2]], [self.normalized_imu_theta], [self.imu_w], [self.imu_a_x], [self.imu_a_y]])