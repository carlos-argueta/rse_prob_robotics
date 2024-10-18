import numpy as np
import time

class ExtendedInformationFilter:

	def __init__(self, initial_state, initial_covariance, motion_model, observation_model, **kwargs):
		# Process arguments
		proc_noise_std = kwargs.get('proc_noise_std', [0.02, 0.02, 0.01])
		obs_noise_std = kwargs.get('obs_noise_std', [0.02, 0.02, 0.01])
	
		self.inf_vector = np.linalg.inv(initial_covariance) @ initial_state  # Initial information vector
		self.inf_matrix = np.linalg.inv(initial_covariance)  # Initial information matrix
		self.mu = initial_state
		
		self.g, self.G, self.V = motion_model() # The action model to use. Returns A and B matrices
		
		# Standard deviations for the noise in x, y, and theta (process or action model noise)
		self.proc_noise_std = np.array(proc_noise_std)
		# Process noise covariance (R)
		self.R = np.diag(self.proc_noise_std ** 2)  # process noise covariance

		# Observation model (C)
		self.h, self.H = observation_model() # The observation model to use

		# Standard deviations for the noise in x, y, and theta (observation or sensor model noise)
		self.obs_noise_std = np.array(obs_noise_std)
		# Observation noise covariance (Q)
		self.Q = np.diag(self.obs_noise_std ** 2)
		self.Q_inv = np.linalg.inv(self.Q) 

		self.exec_times_pred = []
		self.exec_times_upd = []
				
	def predict(self, u, dt):
		
		# Perform the prediction step
		start_time = time.time()
		Sigma = np.linalg.inv(self.inf_matrix)
		self.mu = Sigma @ self.inf_vector
		
		self.inf_matrix = np.linalg.inv(self.G(self.mu, u, dt) @ Sigma @ self.G(self.mu, u, dt).T + self.R)
		self.mu = self.g(self.mu, u, dt)
		self.inf_vector = self.inf_matrix @ self.mu

		# This should be part of the update step, but we do it here
		# because the update step does not have the control u as input
		self.mu = self.g(self.mu, u, dt)

		end_time = time.time()
		execution_time = end_time - start_time
		self.exec_times_pred.append(execution_time)
		print(f"Execution time prediction: {execution_time} seconds")

		print("Average exec time pred: ", sum(self.exec_times_pred) / len(self.exec_times_pred))

		return self.inf_vector, self.inf_matrix

	def update(self, z, dt):
		# Some observation models will return an (n,1) vector instead of (n,)
		# Let's drop that 2nd dimension as it is not necessary and causes some 
		# errors below
		z = z.squeeze()

		start_time = time.time()

		# Q_inv = np.linalg.inv(self.Q) 
		self.inf_matrix = self.inf_matrix + self.H(self.mu).T @ self.Q_inv @ self.H(self.mu)
		self.inf_vector = self.inf_vector + self.H(self.mu).T @ self.Q_inv @ (z - self.h(self.mu).squeeze() + self.H(self.mu) @ self.mu)

		end_time = time.time()
		execution_time = end_time - start_time
		self.exec_times_upd.append(execution_time)
		print(f"Execution time update: {execution_time} seconds")

		print("Average exec time update: ", sum(self.exec_times_upd) / len(self.exec_times_upd))
		
		return self.inf_vector, self.inf_matrix
