import rclpy

import numpy as np

from rse_motion_models.velocity_motion_models import velocity_motion_model_particles
from rse_observation_models.odometry_observation_models import odometry_observation_model_particles

from .filters.pf import ParticleFilter
from .pf_node import ParticleFilterNode


def main(args=None):
    # Initialize the Kalman Filter
    mu0 = np.zeros(3)
    mu0[0] = 1.0
    mu0[1] = -2.0 
    # Sigma0 = np.eye(3)
    proc_noise_std = [0.002, 0.002, 0.001] 
    obs_noise_std = [5.0, 5.0, 1000.0]
    alphas = [0.1, 0.01, 0.01, 0.1, 0.01, 0.01]

    pf = ParticleFilter(mu0, #Sigma0, 
                               velocity_motion_model_particles,
                               odometry_observation_model_particles,
                               num_particles=1000,
                               resampling_method="stratified",
                               proc_noise_std = proc_noise_std,
                               obs_noise_std = obs_noise_std,
                               motion_model_params=alphas)

    rclpy.init(args=args)
    particle_filter_node = ParticleFilterNode(pf)
    rclpy.spin(particle_filter_node)
    particle_filter_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
