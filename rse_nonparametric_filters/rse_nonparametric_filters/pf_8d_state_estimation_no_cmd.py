import rclpy

import numpy as np

from rse_motion_models.acceleration_motion_models import acceleration_motion_model_particles
from rse_observation_models.odometry_imu_observation_models import odometry_imu_observation_model_particles_1
from rse_observation_models.odometry_imu_observation_models import odometry_imu_observation_model_particles_2

from .filters.pf import ParticleFilter
from .pf_node import ParticleFilterFusionNode


def main(args=None):
    # Initialize the Kalman Filter
    mu0 = np.zeros(8)
    mu0[0] = 1.0
    mu0[1] = -2.0 
    # Sigma0 = np.eye(3)
    proc_noise_std = [0.1, 0.1, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1] # [x, y, theta, v_x, v_y, w, a_x, a_y]
    obs_noise_std = [5.0, 5.0, 1000.0, 0.05, 0.05, 1.0, 1.0] #[x, y, theta, theta_imu, w, a_x, a_y]

    # Parameters for the motion model
    tau_v  = 0.2    # speed response
    tau_w  = 0.7    # yaw response
    tau_vy = 0.2    # lateral velocity damping
    tau_ay = 0.6    # lateral accel damping

    # Process noise (per sqrt(sec))
    sig_ax =  0.4   # accel noise x  (m/s^2)
    sig_ay =  0.4   # accel noise y  (m/s^2)
    sig_w  = 0.05   # yaw-rate noise (rad/s)
    sig_vx = 0.02   # direct vel noise (m/s)
    sig_vy = 0.02
    sig_p  = 0.003  # position noise (m)
    sig_th = 0.001  # heading integration noise (rad)

    motion_model_params = [
        tau_v, tau_w, tau_vy, tau_ay,
        sig_ax, sig_ay, sig_w, sig_vx, sig_vy, sig_p, sig_th
    ]

    pf = ParticleFilter(mu0, #Sigma0, 
                               acceleration_motion_model_particles,
                               odometry_imu_observation_model_particles_1,
                               num_particles=1000,
                               resampling_method="systematic",
                               proc_noise_std = proc_noise_std,
                               obs_noise_std = obs_noise_std,
                               motion_model_params=motion_model_params)

    rclpy.init(args=args)
    particle_filter_node = ParticleFilterFusionNode(pf)
    rclpy.spin(particle_filter_node)
    particle_filter_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
