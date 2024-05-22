# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import numpy as np 

from rse_motion_models.acceleration_motion_models import acceleration_motion_model_linearized_2
from rse_observation_models.odometry_imu_observation_models import odometry_imu_observation_model_with_acceleration_motion_model_linearized_2

class KalmanFilter:

	def __init__(self, initial_state, initial_covariance, proc_noise_std = [0.02, 0.02, 0.01], obs_noise_std = [0.02, 0.02, 0.01, 0.01, 0.01]):

		self.mu = initial_state # Initial state estimate [x, y, theta]
		self.Sigma = initial_covariance # Initial uncertainty

		self.g, self.G, self.V = acceleration_motion_model_linearized_2() # The action model to use.
		
		# Standard deviations for the noise in x, y, and theta (process or action model noise)
		self.proc_noise_std = np.array(proc_noise_std)
		# Process noise covariance (R)
		self.R = np.diag(self.proc_noise_std ** 2)  # process noise covariance

		# Observation model (C)
		self.h, self.H = odometry_imu_observation_model_with_acceleration_motion_model_linearized_2() # The observation model to use

		# Standard deviations for the noise in x, y, and theta (observation or sensor model noise)
		self.obs_noise_std = np.array(obs_noise_std)
		# Observation noise covariance (Q)
		self.Q = np.diag(self.obs_noise_std ** 2)

		self.prev_theta = 0.0
		

	def predict(self, u, dt):
		# Predict state estimate (mu) 
		self.mu = self.g(self.mu, u, dt)
		# Predict covariance (Sigma)
		self.Sigma = self.G(self.mu, u, dt) @ self.Sigma @ self.G(self.mu, u, dt).T + self.R # V_t @ M_t @ V_t.T

		self.prev_theta = self.mu[2]
		
		return self.mu, self.Sigma

	def update(self, z, dt):
		# Compute the Kalman gain (K)
		K = self.Sigma @ self.H().T @ np.linalg.inv(self.H() @ self.Sigma @ self.H().T + self.Q)
		
		# Update state estimate (mu) 
		innovation = z - self.h(self.mu)
		# print("innovation", innovation.shape)
		self.mu = self.mu + (K @ innovation).reshape((self.mu.shape[0],))
		# print("upmu", self.mu.shape)

		# Update covariance (Sigma)
		I = np.eye(len(K)) # I = np.identity(len(self.mu_t))
		self.Sigma = (I - K @ self.H()) @ self.Sigma

		self.prev_theta = self.mu[2]

		return self.mu, self.Sigma