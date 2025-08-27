import numpy as np
from numpy import random
from numpy.random import randn
from numpy.random import uniform

from rse_observation_models.odometry_imu_observation_models import odometry_imu_observation_model_particles_1
from rse_observation_models.odometry_imu_observation_models import odometry_imu_observation_model_particles_2

import time


class ParticleFilter:

	def __init__(self, initial_state, motion_model, observation_model, num_particles=1000, resampling_method="multinomial", **kwargs):
		# Arguments
		proc_noise_std = kwargs.get('proc_noise_std', [0.02, 0.02, 0.01])
		obs_noise_std = kwargs.get('obs_noise_std', [0.02, 0.02, 0.01])
		self.motion_model_params = kwargs.get('motion_model_params', None)
		
		# Create particles and weights
		self.num_particles = num_particles
		if initial_state is not None:
			self.particles = self.create_gaussian_particles(initial_state, proc_noise_std, self.num_particles)
		else:
			self.particles = self.create_uniform_particles([[-10, 10], [-10, 10], [0, 2 * np.pi]], self.num_particles, angle_dims=[2])
		self.weights = np.ones(self.num_particles) / self.num_particles  # Uniform initial weights

		self.g = motion_model() # The action model to use.
		
		self.h = observation_model() # The observation model to use
		
		# Standard deviations for the observation or sensor model noise
		self.obs_noise_std = np.asarray(obs_noise_std, dtype=np.float64)
	
		# Observation noise covariance (Q)
		self.Q = np.diag(self.obs_noise_std ** 2)
		
		# Resampling method
		if resampling_method == "multinomial":
			self.resample = self.multinomial_resample
		elif resampling_method == "residual":
			self.resample = self.residual_resample
		elif resampling_method == "stratified":
			self.resample = self.stratified_resample
		elif resampling_method == "systematic":
			self.resample = self.systematic_resample
		else:
			raise ValueError("Unknown resampling method: {}".format(resampling_method))
	
		self.exec_times_pred = []
		self.exec_times_upd = []

	def predict(self, u, dt):
		start_time = time.time()
		
		self.particles = self.g(self.particles, u, dt, self.motion_model_params)

		mu, Sigma = self.estimate()

		end_time = time.time()
		execution_time = end_time - start_time
		self.exec_times_pred.append(execution_time)
		print(f"Execution time prediction: {execution_time} seconds")

		print("Average exec time pred: ", sum(self.exec_times_pred) / len(self.exec_times_pred))

		return mu, Sigma

	def update(self, z, dt):
		start_time = time.time()

		# Compute likehood of each particle given the observation
		likelihood = self.h(self.particles, z, self.Q)

		# Update the weights
		self.weights *= likelihood

		# Normalize weights
		self.weights += 1.e-300      # Avoid round-off to zero
		self.weights /= sum(self.weights) # Make weights sum to 1

		# Resample particles based on the updated weights
		if self.neff() < self.num_particles / 2:
			indexes = self.resample()
			self.particles = self.particles[indexes]
			# The code below keeps the relative weights of the particles,
			# another option is to have a uniform distribution like in the original code.
			self.weights = np.ones(self.num_particles) / self.num_particles
			# self.weights = self.weights[indexes]
			# self.weights /= sum(self.weights)
			# assert np.allclose(self.weights, 1/self.num_particles)

		mu, Sigma = self.estimate()
			
		end_time = time.time()
		execution_time = end_time - start_time
		self.exec_times_upd.append(execution_time)
		print(f"Execution time update: {execution_time} seconds")

		print("Average exec time update: ", sum(self.exec_times_upd) / len(self.exec_times_upd))

		return mu, Sigma
	
	def estimate(self):
		# Calculate the weighted mean
		mean = np.average(self.particles, weights=self.weights, axis=0)

		# For the angle (heading), a simple average is incorrect.
		# We average the sin/cos components to correctly handle wrapping.
		sum_sin = np.sum(self.weights * np.sin(self.particles[:, 2]))
		sum_cos = np.sum(self.weights * np.cos(self.particles[:, 2]))
		mean[2] = np.arctan2(sum_sin, sum_cos)

		# Calculate deviations from the mean
		dev = self.particles - mean
		# Wrap the angle deviations to the range [-pi, pi]
		dev[:, 2] = (dev[:, 2] + np.pi) % (2 * np.pi) - np.pi

		# Calculate the weighted covariance matrix
		cov = dev.T @ (dev * self.weights[:, np.newaxis])
		
		return mean, cov
	
	# Implementing the effective sample size N (Neff) calculation
	def neff(self):
		return 1. / np.sum(np.square(self.weights))
	
	def multinomial_resample(self):
		cumulative_sum = np.cumsum(self.weights)
		cumulative_sum[-1] = 1.  # avoid round-off errors
		return np.searchsorted(cumulative_sum, random.rand(len(self.weights)))  
	
	def residual_resample(self):
		N = len(self.weights)
		indexes = np.zeros(N, 'i')

		# take int(N*w) copies of each weight
		num_copies = (N*np.asarray(self.weights)).astype(int)
		k = 0
		for i in range(N):
			for _ in range(num_copies[i]): # make n copies
				indexes[k] = i
				k += 1

		# use multinomial resample on the residual to fill up the rest.
		residual = self.weights - num_copies     # get fractional part
		residual /= sum(residual)     # normalize
		cumulative_sum = np.cumsum(residual)
		cumulative_sum[-1] = 1. # ensures sum is exactly one
		indexes[k:N] = np.searchsorted(cumulative_sum, random.rand(N-k))

		return indexes
	
	def stratified_resample(self):
		N = len(self.weights)
		# make N subdivisions, chose a random position within each one
		positions = (random.rand(N) + range(N)) / N

		indexes = np.zeros(N, 'i')
		cumulative_sum = np.cumsum(self.weights)
		i, j = 0, 0
		while i < N:
			if positions[i] < cumulative_sum[j]:
				indexes[i] = j
				i += 1
			else:
				j += 1
		return indexes
	
	def systematic_resample(self):
		N = len(self.weights)

		# make N subdivisions, choose positions 
		# with a consistent random offset
		positions = (np.arange(N) + random.rand()) / N

		indexes = np.zeros(N, 'i')
		cumulative_sum = np.cumsum(self.weights)
		i, j = 0, 0
		while i < N:
			if positions[i] < cumulative_sum[j]:
				indexes[i] = j
				i += 1
			else:
				j += 1
		return indexes
	
	def create_uniform_particles(self, ranges, N, angle_dims=None):
		# Ensure ranges is a NumPy array for calculations
		ranges = np.asarray(ranges)
		
		# Validate the shape of the ranges array
		if ranges.ndim != 2 or ranges.shape[1] != 2:
			raise ValueError("The 'ranges' parameter must be an array-like object with shape (D, 2), where D is the number of dimensions.")

		# Get the dimensionality of the state
		dim = ranges.shape[0]

		# Extract the lower and upper bounds for each dimension
		lows = ranges[:, 0]
		highs = ranges[:, 1]

		# Generate N particles. The `uniform` function can take arrays for `low`
		# and `high`, which allows us to generate all particles for all dimensions
		# in a single, efficient operation. The size is specified as (N, dim)
		# to get the desired output shape.
		particles = uniform(low=lows, high=highs, size=(N, dim))

		# Wrap the angle dimensions if any are specified
		if angle_dims:
			for i in angle_dims:
				if i < dim:
					particles[:, i] %= 2 * np.pi
				else:
					# Optional: Warn if an invalid index is provided
					print(f"Warning: angle_dims index {i} is out of bounds for state dimension {dim}.")

		return particles

	def create_gaussian_particles(self, mean, std, N, angle_dims=None):
		# Ensure mean and std are NumPy arrays for calculations
		mean = np.asarray(mean)
		std = np.asarray(std)

		# Validate that mean and std have the same number of dimensions
		if mean.shape != std.shape:
			raise ValueError("The shape of 'mean' and 'std' must be the same.")

		# Get the dimensionality of the state from the length of the mean vector
		dim = len(mean)

		# Generate N random samples from a standard normal distribution (mean=0, std=1)
		# The shape will be (N, dim), perfect for creating N particles of dimension `dim`.
		particles = randn(N, dim)

		# Scale by the standard deviation and shift by the mean.
		# NumPy's broadcasting applies the operations element-wise.
		# `std` is broadcast across the N rows, and `mean` is broadcast across the N rows.
		particles = particles * std + mean

		# Wrap the angle dimensions if any are specified
		if angle_dims:
			for i in angle_dims:
				if i < dim:
					particles[:, i] %= 2 * np.pi
				else:
					# Optional: Warn if an invalid index is provided
					print(f"Warning: angle_dims index {i} is out of bounds for state dimension {dim}.")

		return particles