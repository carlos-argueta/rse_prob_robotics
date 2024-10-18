# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import rclpy

import numpy as np

from rse_motion_models.acceleration_motion_models import acceleration_motion_model_linearized_1
from rse_observation_models.odometry_imu_observation_models import odometry_imu_observation_model_with_acceleration_motion_model_linearized_1

from .filters.ekf import ExtendedKalmanFilter
from .kf_node import KalmanFilterFusionNode as ExtendedKalmanFilterFusionNode


def main(args=None):
    # Initialize the Kalman Filter

    mu0 = np.zeros(7)
    Sigma0 = np.eye(7)
    proc_noise_std = [0.1, 0.1, 0.05, 0.1, 0.1, 0.1, 0.1] # [x, y, theta, v, w, a_x, a_y]
    obs_noise_std = [100.0, 100.0, 1000.0, 6.853891945200942e-06, 1.0966227112321507e-06, 0.0015387262937311438, 0.0015387262937311438] #[x, y, theta, theta_imu, w, a_x, a_y]

    ekf = ExtendedKalmanFilter(mu0, Sigma0, 
                               acceleration_motion_model_linearized_1,
                               odometry_imu_observation_model_with_acceleration_motion_model_linearized_1,
                               proc_noise_std = proc_noise_std,
                               obs_noise_std = obs_noise_std)

    rclpy.init(args=args)
    kalman_filter_node = ExtendedKalmanFilterFusionNode(ekf)
    rclpy.spin(kalman_filter_node)
    kalman_filter_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()