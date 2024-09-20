# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import rclpy

import numpy as np

from rse_motion_models.velocity_motion_models import velocity_motion_model_linearized_1
from rse_observation_models.odometry_observation_models import odometry_observation_model_linearized

from .filters.ukf import UnscentedKalmanFilter 
from .ukf_node import UnscentedKalmanFilterNode

def main(args=None):
    # Initialize the Kalman Filter
    mu0 = np.zeros(3)
    Sigma0 = np.eye(3)
    proc_noise_std = [0.002, 0.002, 0.001] 
    obs_noise_std = [1000.02, 1000.02, 1000.01]

    '''
    beta = 2 is a good choice for Gaussian problems, 
    kappa = 3 - n where n is the dimension of X is a good choice for kappa
    and 0 < alpha <= 1 is an appropriate choice for alpha 
    where a larger value for alpha spreads the sigma points further from the mean.
    '''
    alpha = 1.0
    beta = 2.0
    kappa = 0.0

    kf = UnscentedKalmanFilter(mu0, Sigma0, 
                velocity_motion_model_linearized_1, 
                odometry_observation_model_linearized, 
                proc_noise_std = proc_noise_std, 
                obs_noise_std = obs_noise_std,
                alpha = alpha,
                beta = beta,
                kappa = kappa)

    rclpy.init(args=args)
    kalman_filter_node = UnscentedKalmanFilterNode(kf)
    rclpy.spin(kalman_filter_node)
    kalman_filter_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
