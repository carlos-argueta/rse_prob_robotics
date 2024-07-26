
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import numpy as np 
import math

from scipy.linalg import cholesky

from rse_common_utils.helper_utils import normalize_angle

def state_residual(a, b):
	y = a - b
	y[2] = normalize_angle(y[2])

	return y

def state_add(a, b):
	y = a + b
	y[2] = normalize_angle(y[2])

	return y

def state_mean(Wm, sigmas):
	x = np.zeros(3)

	sum_sin = np.sum(np.dot(np.sin(sigmas[:, 2]), Wm))
	sum_cos = np.sum(np.dot(np.cos(sigmas[:, 2]), Wm))
	x[0] = np.sum(np.dot(sigmas[:, 0], Wm))
	x[1] = np.sum(np.dot(sigmas[:, 1], Wm))
	x[2] = math.atan2(sum_sin, sum_cos)
	
	return x

class UnscentedKalmanFilter:

	def __init__(self, initial_state, initial_covariance, motion_model, observation_model, **kwargs):
		# Process arguments
		proc_noise_std = kwargs.get('proc_noise_std', [0.02, 0.02, 0.01])
		obs_noise_std = kwargs.get('obs_noise_std', [0.02, 0.02, 0.01])
	
		self.alpha = kwargs.get('alpha', 1.0)
		self.beta = kwargs.get('beta', 2)
		self.kappa = kwargs.get('kappa', 0)
		self.subtract_fn = kwargs.get('subtract_fn', np.subtract)
		self.state_add_fn = kwargs.get('state_add_fn', np.add)
		self.state_mean_fn = kwargs.get('state_mean_fn', np.dot)
		self.observation_mean_fn = kwargs.get('observation_mean_fn', np.dot)
		self.state_residual_fn = kwargs.get('state_residual_fn', np.subtract)
		self.observation_residual_fn = kwargs.get('observation_residual_fn', np.subtract)
		self.sqrt_fn = kwargs.get('sqrt_fn', cholesky)

		self.mu = initial_state  # Initial state estimate 
		self.Sigma = initial_covariance  # Initial uncertainty

		self.g, _, _ = motion_model()  # The action model to use.
		
		# Standard deviations of the process or action model noise
		self.proc_noise_std = np.array(proc_noise_std)
		# Process noise covariance (R)
		self.R = np.diag(self.proc_noise_std ** 2) 

		self.h, _ = observation_model()  # The observation model to use

		# Standard deviations for the observation or sensor model noise
		self.obs_noise_std = np.array(obs_noise_std)
		# Observation noise covariance (Q)
		self.Q = np.diag(self.obs_noise_std ** 2)

		# UKF specific variables
		self.n = np.size(self.mu)
		self.lambda_ = self.alpha ** 2 * (self.n + self.kappa) - self.n
		self.num_sigmas = 2 * self.n + 1

		# Compute associated weights
		weight = .5 / (self.n + self.lambda_)
		self.Wm = np.full(self.num_sigmas, weight)
		self.Wc = np.full(self.num_sigmas, weight)
		self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
		self.Wc[0] = self.lambda_ / (self.n + self.lambda_) + (1 - self.alpha ** 2 + self.beta)

	def compute_sigma_points(self):
		sigmas = np.zeros((self.num_sigmas, self.n))
		sigmas[0] = self.mu
		for k in range(self.n):
			sigmas[k + 1]   = self.subtract_fn(self.mu, -self.sqrt_fn((self.n + self.lambda_) * self.Sigma)[k]) # k chooses the kth row vector of the matrix
			sigmas[self.n + k + 1] = self.subtract_fn(self.mu, self.sqrt_fn((self.n + self.lambda_) * self.Sigma)[k])

		return sigmas

	def predict(self, u, dt):
		# Step 1: Compute Sigma points and their weights
		# Compute Sigma points
		sigmas = self.compute_sigma_points()
		
		# Step 2: Pass Sigma points through non-linear function g
		sigmas_star = np.zeros((self.num_sigmas, self.n))
		for i, s in enumerate(sigmas):
			sigmas_star[i] = self.g(s, u, dt)

		# Step 3: Predict state estimate (mu) 
		self.mu = self.state_mean_fn(self.Wm, sigmas_star)
		
		# Step 4: Predict covariance (Sigma)
		if self.state_residual_fn is np.subtract or self.state_residual_fn is None:
			y = sigmas_star - self.mu[np.newaxis, :]
			self.Sigma = np.dot(y.T, np.dot(np.diag(self.Wc), y))
		else:
			self.Sigma = np.zeros((self.n, self.n))
			for k in range(self.num_sigmas):
				y = self.state_residual_fn(sigmas_star[k], self.mu)
				self.Sigma += self.Wc[k] * np.outer(y, y)

		self.Sigma += self.R

		return self.mu, self.Sigma

	def update(self, z, dt):

		# Some observation models will return an (n,1) vector instead of (n,)
		# Let's drop that 2nd dimension as it is not necessary and causes some 
		# errors below
		z = z.squeeze()
		
		# Step 1: Update Sigma points
		sigmas = self.compute_sigma_points()

		# Step 2: Pass Sigma points through non-linear function h
		sigmas_h = np.zeros((self.num_sigmas, np.size(z)))
		for i, s in enumerate(sigmas):
			sigmas_h[i] = self.h(s).squeeze() # As above, let's drop the 2nd dimension

		# Step 3: Predict the mean of the prediction passed through unscented transform
		z_hat = self.observation_mean_fn(self.Wm, sigmas_h)
		
		# Step 4: Compute covariance matrix S
		if self.observation_residual_fn is np.subtract or self.observation_residual_fn is None:
			y = sigmas_h - z_hat[np.newaxis, :]
			S = np.dot(y.T, np.dot(np.diag(self.Wc), y))
		else:
			S = np.zeros((self.n, self.n))
			for k in range(self.num_sigmas):
				y = self.observation_residual_fn(sigmas_h[k], z_hat)
				S += self.Wc[k] * np.outer(y, y)

		S += self.Q

		# Step 5: Compute cross variance
		Sigma_xz = np.zeros((self.n, np.size(z)))

		for i in range(self.num_sigmas):
			dx = self.state_residual_fn(sigmas[i], self.mu)
			dz = self.observation_residual_fn(sigmas_h[i], z)	
			Sigma_xz += self.Wc[i] * np.outer(dx, dz)

		# Step 6: Compute the Kalman gain (K)
		K = np.dot(Sigma_xz, np.linalg.inv(S))     

		# Step 7: Update the state mean
		# self.mu = self.mu + K @ self.observation_residual_fn(z , z_hat)
		self.mu = self.state_add_fn(self.mu, np.dot(K, self.observation_residual_fn(z , z_hat)))

		# Step 8: Update the state covariance
		# self.Sigma = self.Sigma - K @ S @ K.T
		self.Sigma = self.Sigma - np.dot(K, np.dot(S, K.T))

		return self.mu, self.Sigma