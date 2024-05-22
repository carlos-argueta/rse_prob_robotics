# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np

from .sensor_utils import odom_to_pose2D, get_normalized_pose2D, Odom2DDriftSimulator
from .visualization import Visualizer
from .filters.kalman_filter import KalmanFilter 

class KalmanFilterNode(Node):
    def __init__(self):
        super().__init__('kalman_filter_node')
        
        self.odom_subscription = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10)

        self.cmd_vel_subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10)

        # Initialize the Kalman Filter
        self.kf = KalmanFilter(np.zeros(3), np.eye(3), proc_noise_std = [0.02, 0.02, 0.01], obs_noise_std = [2.0, 2.0, 2.0])
        # Initialize the drift simulator, to corrupt the perfect simulation odometry
        self.odom_simulator = Odom2DDriftSimulator()    
        # Initialize the visualizer to see the results
        self.visualizer = Visualizer()
        
        self.u = np.zeros(2) # Initial controls (linear and angular velocities) [v, omega]

        self.prev_time = None # previous prediction time, used to compute the delta_t

        self.first_prediction_done = False # We only want to update after we have done an initial prediction

        # Variables to normalize the pose (always start at the origin)
        self.initial_pose = None
        self.normalized_pose = (0.0, 0.0, 0.0) 

    def cmd_vel_callback(self, msg):
        self.u = np.asarray([msg.linear.x, msg.angular.z]) # Get the control inputs

        # Compute dt
        curr_time = self.get_clock().now().nanoseconds
        if self.prev_time:
            dt = (curr_time - self.prev_time) / 1e9  # Convert nanoseconds to seconds
        else:
            dt = 0.0

        # Prediction step  ---------------------------------------------------------------
        mu, Sigma = self.kf.predict(self.u, dt)
        self.first_prediction_done = True
        # Prediction step  ---------------------------------------------------------------

        # View the results
        self.visualizer.update(self.normalized_pose, mu, Sigma, step="predict")
        print("Predicted", mu)
       
        self.prev_time = curr_time

    def odom_callback(self, msg):

        # Set the initial pose
        if not self.initial_pose:
            self.initial_pose = odom_to_pose2D(msg)

        # Get and normalize the pose
        current_pose = odom_to_pose2D(msg)
        self.normalized_pose = np.array(get_normalized_pose2D(self.initial_pose, current_pose))

        if self.first_prediction_done:
            # Update step ---------------------------------------------------------------------------------
            # Corrupt and set the normalized pose as our observation
            curr_time_secs = self.get_clock().now().nanoseconds / 1e9
            z = self.odom_simulator.add_drift(self.normalized_pose, curr_time_secs) 
            mu, Sigma = self.kf.update(z)
            # Update step ---------------------------------------------------------------------------------

            # View the results
            self.visualizer.update(self.normalized_pose, mu, Sigma, step="update")
            print("Updated", mu)
            print("Odometry", self.normalized_pose)

def main(args=None):
    rclpy.init(args=args)
    kalman_filter_node = KalmanFilterNode()
    rclpy.spin(kalman_filter_node)
    kalman_filter_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
