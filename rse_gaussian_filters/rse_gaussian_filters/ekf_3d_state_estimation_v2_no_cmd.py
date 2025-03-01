
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import rclpy

import numpy as np

from rse_motion_models.velocity_motion_models import velocity_motion_model_linearized_2
from rse_observation_models.odometry_observation_models import odometry_observation_model_linearized

from .filters.ekf import ExtendedKalmanFilter
from .kf_node import KalmanFilterNode as ExtendedKalmanFilterNode


def main(args=None):
    # Initialize the Kalman Filter
    mu0 = np.zeros(3)
    Sigma0 = np.eye(3)
    proc_noise_std = [0.002, 0.002, 0.001]
    obs_noise_std = [0.02, 0.02, 1000.01]

    ekf = ExtendedKalmanFilter(mu0, Sigma0, 
                               velocity_motion_model_linearized_2,
                               odometry_observation_model_linearized,
                               proc_noise_std = proc_noise_std,
                               obs_noise_std = obs_noise_std)

    rclpy.init(args=args)
    kalman_filter_node = ExtendedKalmanFilterNode(ekf)
    rclpy.spin(kalman_filter_node)
    kalman_filter_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()