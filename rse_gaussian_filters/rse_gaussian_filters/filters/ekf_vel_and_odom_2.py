# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import numpy as np 

from rse_motion_models.velocity_motion_models import velocity_motion_model_linearized_2
from rse_observation_models.odometry_observation_models import odometry_observation_model_linearized

class KalmanFilter:

	def __init__(self, initial_state, initial_covariance, proc_noise_std = [0.02, 0.02, 0.01], obs_noise_std = [0.02, 0.02, 0.01]):

		self.mu = initial_state # Initial state estimate 
		self.Sigma = initial_covariance # Initial uncertainty

		self.g, self.G, self.V = velocity_motion_model_linearized_2() # The action model to use.
		
		# Standard deviations of the process or action model noise
		self.proc_noise_std = np.array(proc_noise_std)
		# Process noise covariance (R)
		self.R = np.diag(self.proc_noise_std ** 2) 

		self.h, self.H = odometry_observation_model_linearized() # The observation model to use

		# Standard deviations for the observation or sensor model noise
		self.obs_noise_std = np.array(obs_noise_std)
		# Observation noise covariance (Q)
		self.Q = np.diag(self.obs_noise_std ** 2)
		

	def predict(self, u, dt):
		# Predict state estimate (mu) 
		self.mu = self.g(self.mu, u, dt)
		# Predict covariance (Sigma)
		self.Sigma = self.G(self.mu, u, dt) @ self.Sigma @ self.G(self.mu, u, dt).T + self.R # V_t @ M_t @ V_t.T

		return self.mu, self.Sigma

	def update(self, z, dt):
		# Compute the Kalman gain (K)
		K = self.Sigma @ self.H(self.mu).T @ np.linalg.inv(self.H(self.mu) @ self.Sigma @ self.H(self.mu).T + self.Q)
		
		# Update state estimate (mu) 
		self.mu = self.mu + K @ (z - self.h(self.mu))

		# Update covariance (Sigma)
		I = np.eye(len(K)) 
		self.Sigma = (I - K @ self.H(self.mu)) @ self.Sigma

		return self.mu, self.Sigma