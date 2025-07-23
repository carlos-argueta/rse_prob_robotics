

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import rclpy

import numpy as np

from rse_motion_models.velocity_motion_models import velocity_motion_model_particles

from .filters.pf import ParticleFilter
from .pf_node import ParticleFilterNode


def main(args=None):
    # Initialize the Kalman Filter
    mu0 = np.zeros(3)
    # Sigma0 = np.eye(3)
    proc_noise_std = [0.002, 0.002, 0.001] 
    obs_noise_std = [0.02, 0.02, 1000.01]
    alphas = [0.1, 0.01, 0.01, 0.1, 0.01, 0.01]

    pf = ParticleFilter(mu0, #Sigma0, 
                               velocity_motion_model_particles,
                               None,
                               num_particles=1000,
                               resampling_method="multinomial",
                               proc_noise_std = proc_noise_std,
                               obs_noise_std = obs_noise_std,
                               alphas=alphas)

    rclpy.init(args=args)
    particle_filter_node = ParticleFilterNode(pf)
    rclpy.spin(particle_filter_node)
    particle_filter_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
