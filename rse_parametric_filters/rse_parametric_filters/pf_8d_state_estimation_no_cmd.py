

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import rclpy

import numpy as np

from rse_motion_models.acceleration_motion_models import acceleration_motion_model_particles
from rse_observation_models.odometry_imu_observation_models import odometry_imu_observation_model_particles

from .filters.pf import ParticleFilter
from .pf_node import ParticleFilterFusionNode


def main(args=None):
    # Initialize the Kalman Filter
    mu0 = np.zeros(8)
    mu0[0] = 1.0
    mu0[1] = -2.0 
    # Sigma0 = np.eye(3)
    proc_noise_std = [0.1, 0.1, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1] # [x, y, theta, v_x, v_y, w, a_x, a_y]
    obs_noise_std = [100.0, 100.0, 1000.0, 6.853891945200942e-06, 1.0966227112321507e-06, 0.0015387262937311438, 0.0015387262937311438] #[x, y, theta, theta_imu, w, a_x, a_y]

    # Alphas for noise on a_x, a_y, w based on v_cmd, w_cmd
    alphas = [0.01, 0.001, 0.01, 0.001, 0.001, 0.01]

    pf = ParticleFilter(mu0, #Sigma0, 
                               acceleration_motion_model_particles,
                               odometry_imu_observation_model_particles,
                               num_particles=1000,
                               resampling_method="stratified",
                               proc_noise_std = proc_noise_std,
                               obs_noise_std = obs_noise_std,
                               alphas=alphas)

    rclpy.init(args=args)
    particle_filter_node = ParticleFilterFusionNode(pf)
    rclpy.spin(particle_filter_node)
    particle_filter_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
