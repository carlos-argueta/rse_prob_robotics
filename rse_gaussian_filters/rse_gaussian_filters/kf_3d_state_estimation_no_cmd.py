# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

import numpy as np

from rse_common_utils.sensor_utils import odom_to_pose2D, get_normalized_pose2D, rotate_pose2D, get_yaw_from_quaternion
from rse_common_utils.helper_utils import normalize_angle
from rse_common_utils.visualization import Visualizer
from .filters.kf_vel_and_odom import KalmanFilter 

class KalmanFilterNode(Node):
    def __init__(self):
        super().__init__('kalman_filter_node')
        
        self.odom_gt_subscription = self.create_subscription(
            Odometry,
            'odom', #'wheel_odom',  # Ground Truth
            self.odom_gt_callback,
            10)

        self.odom_raw_subscription = self.create_subscription(
            Odometry,
            'odom_raw', #'odom_raw',         # For controls
            self.odom_raw_callback,
            10)


        # Initialize the Kalman Filter
        mu0 = np.zeros(3)
        Sigma0 = np.eye(3)
        #proc_noise_std = [1000.02, 1000.02, 1000.01] 
        #obs_noise_std = [0.002, 0.002, 0.001]
        proc_noise_std = [0.002, 0.002, 0.001]
        obs_noise_std = [1000.02, 1000.02, 1000.01]
        
        self.kf = KalmanFilter(mu0, Sigma0, proc_noise_std, obs_noise_std)
        # Initialize the visualizer to see the results
        self.visualizer = Visualizer()
        
        self.u = np.zeros(2) # Initial controls (linear and angular velocities) [v, omega]

        self.prev_time = None # previous prediction time, used to compute the delta_t

        # Variables to normalize the pose (always start at the origin)
        self.initial_pose = None
        self.normalized_pose = (0.0, 0.0, 0.0) 
        
        # IMU data
        self.initial_imu_theta = None
        self.normalized_imu_theta = 0.0
        self.imu_w = 0.0

        self.prev_normalized_pose = (0.0, 0.0, 0.0)
        self.prev_pose_set = None 

        self.initial_gt_pose = None
        self.normalized_gt_pose = (0.0, 0.0, 0.0)

    def odom_raw_callback(self, msg):
        
        # Set the initial pose
        if not self.initial_pose:
            initial_pose = odom_to_pose2D(msg)  

            self.initial_pose = rotate_pose2D(initial_pose, -90)

        # Get and normalize the pose
        current_pose = odom_to_pose2D(msg)
        rotated_pose = rotate_pose2D(current_pose, -90)        
        self.normalized_pose = np.array(get_normalized_pose2D(self.initial_pose, rotated_pose))

        # Get the control inputs for velocity model
        self.u = np.asarray([msg.twist.twist.linear.x, msg.twist.twist.angular.z]) 
        # Get the control inputs for odometry model
        ''' if self.prev_pose_set:
            self.u = np.asarray([self.prev_normalized_pose, self.normalized_pose])
        else:
            self.prev_normalized_pose = self.normalized_pose
            self.prev_pose_set = True
            return
        '''

        # Compute dt
        curr_time = self.get_clock().now().nanoseconds
        if self.prev_time:
            dt = (curr_time - self.prev_time) / 1e9  # Convert nanoseconds to seconds
        else:
            dt = 0.01

        # Prediction step  ---------------------------------------------------------------
        mu, Sigma = self.kf.predict(self.u, dt)
        # Prediction step  ---------------------------------------------------------------

        # View the results
        self.visualizer.update(self.normalized_gt_pose, mu, Sigma, step="predict")
        print("Predicted", mu)
       
        self.prev_time = curr_time

        # Update step ---------------------------------------------------------------------------------
        z = self.normalized_pose # Set the normalized pose as our observation
        # Include them in the measurement vector `z`
        # z =  np.array([[self.normalized_pose[0]], [self.normalized_pose[1]], [self.normalized_pose[2]], [self.normalized_imu_theta], [self.imu_w]])

        mu, Sigma = self.kf.update(z, dt)
        # Update step ---------------------------------------------------------------------------------

        # View the results
        print("Updated", mu)
        self.visualizer.update(self.normalized_gt_pose, mu, Sigma, z, step="update")

        self.prev_normalized_pose = self.normalized_pose

    def imu_callback(self, msg):

        self.imu_msg = msg

        # Extract the linear acceleration data from the IMU message
        a_x = msg.linear_acceleration.x
        a_y = msg.linear_acceleration.y

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
            
        
        # You may also want to update the measurement noise covariance matrix R with the variance of the acceleration measurements
        # This is an optional step and will depend on the specifics of your application
        # For example, you might do something like this:
        #self.R[3][3] = np.var(a_x)
        #self.R[4][4] = np.var(a_y)    

        
    def odom_gt_callback(self, msg):
        # Set the initial pose
        if not self.initial_gt_pose:

            initial_pose = odom_to_pose2D(msg)  

            self.initial_gt_pose = initial_pose # rotate_pose2D(initial_pose, 70)

        # Get and normalize the pose
        current_pose = odom_to_pose2D(msg)  

        # rotated_pose = rotate_pose2D(current_pose, 70)

        self.normalized_gt_pose = np.array(get_normalized_pose2D(self.initial_gt_pose, current_pose))
       
def main(args=None):
    rclpy.init(args=args)
    kalman_filter_node = KalmanFilterNode()
    rclpy.spin(kalman_filter_node)
    kalman_filter_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
