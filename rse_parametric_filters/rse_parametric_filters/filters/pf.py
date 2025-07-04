import numpy as np
from numpy.random import randn
from numpy.random import uniform


class ParticleFilter:

	def __init__(self, initial_state, motion_model, observation_model, num_particles=1000, **kwargs):
		# Process arguments
		proc_noise_std = kwargs.get('proc_noise_std', [0.02, 0.02, 0.01])
		obs_noise_std = kwargs.get('obs_noise_std', [0.02, 0.02, 0.01])

		# Create particles and weights
		self.num_particles = num_particles
		if initial_state is not None:
			particles = self.create_gaussian_particles(initial_state, proc_noise_std, self.num_particles)
		else:
			particles = self.create_uniform_particles([[-10, 10], [-10, 10], [0, 2 * np.pi]], self.num_particles, angle_dims=[2])
		self.weights = np.ones(self.num_particles) / self.num_particles  # Uniform initial weights

		self.mu = initial_state # Initial state estimate 
		self.Sigma = initial_covariance # Initial uncertainty

		self.g, self.G, self.V = motion_model() # The action model to use.
		
		# Standard deviations of the process or action model noise
		self.proc_noise_std = np.array(proc_noise_std)
		# Process noise covariance (R)
		self.R = np.diag(self.proc_noise_std ** 2)

		self.h, self.H = observation_model() # The observation model to use

		# Standard deviations for the observation or sensor model noise
		self.obs_noise_std = np.array(obs_noise_std)
		# Observation noise covariance (Q)
		self.Q = np.diag(self.obs_noise_std ** 2)

		self.exec_times_pred = []
		self.exec_times_upd = []

	def create_uniform_particles(self, ranges, N, angle_dims=None):
		"""
		Creates N particles with a uniform distribution for a state of any dimension.

		This function generates particles where each state variable is drawn from an
		independent uniform distribution defined by a specified range.

		Parameters
		----------
		ranges : array_like
			A list or array of ranges for each state variable. The shape should be
			(D, 2), where D is the dimensionality of the state. Each inner list
			or tuple should contain the minimum and maximum values for that variable.
			Example: [[x_min, x_max], [y_min, y_max], [hdg_min, hdg_max]]

		N : int
			The number of particles to generate.

		angle_dims : list of int, optional
			A list of indices for state variables that are angles. These angles will
			be wrapped to the range [0, 2*pi]. For example, if the third state
			variable (index 2) is a heading, you would pass `angle_dims=[2]`.
			If None, no wrapping is performed.

		Returns
		-------
		numpy.ndarray
			An array of particles of shape (N, D), where D is the dimensionality
			of the state. Each row is a particle, and each column is a state variable.
			
		Raises
		------
		ValueError
			If the `ranges` array does not have the shape (D, 2).
		"""
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
		"""
		Creates N particles with a Gaussian distribution for a state of any dimension.

		This function generates particles where each state variable is drawn from an
		independent Gaussian distribution defined by the corresponding mean and standard
		deviation.

		Parameters
		----------
		mean : array_like
			The mean of the state variables. The length of this array determines the
			dimensionality of the state.
			Example: [x_mean, y_mean, heading_mean]

		std : array_like
			The standard deviation for each state variable. Must be the same length
			as `mean`.
			Example: [x_std, y_std, heading_std]

		N : int
			The number of particles to generate.

		angle_dims : list of int, optional
			A list of indices for state variables that are angles. These angles will
			be wrapped to the range [0, 2*pi]. For example, if the third state
			variable (index 2) is a heading, you would pass `angle_dims=[2]`.
			If None, no wrapping is performed.

		Returns
		-------
		numpy.ndarray
			An array of particles of shape (N, D), where D is the dimensionality
			of the state (i.e., len(mean)). Each row is a particle, and each column
			is a state variable.

		Raises
		------
		ValueError
			If the lengths of `mean` and `std` are not equal.
		"""
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


	def predict(self, u, dt):
		start_time = time.time()
		# Predict state estimate (mu) 
		self.mu = self.g(self.mu, u, dt)
		# Predict covariance (Sigma)
		self.Sigma = self.G(self.mu, u, dt) @ self.Sigma @ self.G(self.mu, u, dt).T + self.R 

		end_time = time.time()
		execution_time = end_time - start_time
		self.exec_times_pred.append(execution_time)
		print(f"Execution time prediction: {execution_time} seconds")

		print("Average exec time pred: ", sum(self.exec_times_pred) / len(self.exec_times_pred))


		return self.mu, self.Sigma

	def update(self, z, dt):
		start_time = time.time()

		# Compute the Kalman gain (K)
		K = self.Sigma @ self.H(self.mu).T @ np.linalg.inv(self.H(self.mu) @ self.Sigma @ self.H(self.mu).T + self.Q)
		
		# Update state estimate (mu) 
		innovation = z - self.h(self.mu)
		self.mu = self.mu + (K @ innovation).reshape((self.mu.shape[0],))

		# Update covariance (Sigma)
		I = np.eye(len(K))
		self.Sigma = (I - K @ self.H(self.mu)) @ self.Sigma

		end_time = time.time()
		execution_time = end_time - start_time
		self.exec_times_upd.append(execution_time)
		print(f"Execution time update: {execution_time} seconds")

		print("Average exec time update: ", sum(self.exec_times_upd) / len(self.exec_times_upd))

		return self.mu, self.Sigma