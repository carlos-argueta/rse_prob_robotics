# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import numpy as np 

from rse_motion_models.motion_models import odometry_motion_model
from rse_observation_models.observation_models import odometry_observation_model

class KalmanFilter:

	def __init__(self, initial_state, initial_covariance, proc_noise_std = [0.02, 0.02, 0.01], obs_noise_std = [0.02, 0.02, 0.01]):

		self.mu = initial_state # Initial state estimate [x, y, theta]
		self.Sigma = initial_covariance # Initial uncertainty

		self.A, self.B = odometry_motion_model() # The action model to use. Returns A and B matrices
		
		# Standard deviations for the noise in x, y, and theta (process or action model noise)
		self.proc_noise_std = np.array(proc_noise_std)
		# Process noise covariance (R)
		self.R = np.diag(self.proc_noise_std ** 2)  # process noise covariance

		# Observation model (C)
		self.C = odometry_observation_model() # The observation model to use

		# Standard deviations for the noise in x, y, and theta (observation or sensor model noise)
		self.obs_noise_std = np.array(obs_noise_std)
		# Observation noise covariance (Q)
		self.Q = np.diag(self.obs_noise_std ** 2)
		

	def predict(self, u, dt):
		# Predict state estimate (mu) 
		self.mu = self.A().dot(self.mu) + self.B(self.mu, u)
		# Predict covariance (Sigma)
		self.Sigma = self.A().dot(self.Sigma).dot(self.A().T) + self.R

		return self.mu, self.Sigma

	def update(self, z):
		# Compute the Kalman gain (K)
		K = self.Sigma.dot(self.C.T).dot(np.linalg.inv(self.C.dot(self.Sigma).dot(self.C.T) + self.Q))
		# Update state estimate (mu) 
		self.mu = self.mu + K.dot(z - self.C.dot(self.mu))
		# Update covariance (Sigma)
		self.Sigma = (np.eye(len(K)) - K.dot(self.C)).dot(self.Sigma)

		return self.mu, self.Sigma