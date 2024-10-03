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

		print("prediction step")
		print("mu shape: ", self.mu.shape)
		print("inf_matrix shape: ", self.inf_matrix.shape)
		print("inf_vector shape: ", self.inf_vector.shape)

		end_time = time.time()
		execution_time = end_time - start_time
		self.exec_times_pred.append(execution_time)
		print(f"Execution time prediction: {execution_time} seconds")

		print("Average exec time pred: ", sum(self.exec_times_pred) / len(self.exec_times_pred))

		# Predict state estimate (self.mu) 
		# self.self.mu = self.A() @ self.self.mu + self.B(self.self.mu, dt) @ u
		# Predict covariance (Sigma)
		# self.Sigma = self.A() @ self.Sigma @ self.A().T + self.R

		return self.inf_vector, self.inf_matrix

	def update(self, z, dt):
		# Some observation models will return an (n,1) vector instead of (n,)
		# Let's drop that 2nd dimension as it is not necessary and causes some 
		# errors below
		print("z shape: ", z.shape)
		z = z.squeeze()
		print("z shape: ", z.shape)

		start_time = time.time()

		Q_inv = np.linalg.inv(self.Q) 
		self.inf_matrix = self.inf_matrix + self.H(self.mu).T @ Q_inv @ self.H(self.mu)
		self.inf_vector = self.inf_vector + self.H(self.mu).T @ Q_inv @ (z - self.h(self.mu).squeeze() + self.H(self.mu) @ self.mu)

		print("update step")
		print("mu shape: ", self.mu.shape)
		print("inf_matrix shape: ", self.inf_matrix.shape)
		print("inf_vector shape: ", self.inf_vector.shape)
		print("h(mu) shape", self.h(self.mu).shape)
		print("H(mu) shape", self.H(self.mu).shape)


		end_time = time.time()
		execution_time = end_time - start_time
		self.exec_times_upd.append(execution_time)
		print(f"Execution time update: {execution_time} seconds")

		print("Average exec time update: ", sum(self.exec_times_upd) / len(self.exec_times_upd))
		
		# Compute the Kalman gain (K)
		# K = self.Sigma @ self.C().T @ np.linalg.inv(self.C() @ self.Sigma @ self.C().T + self.Q)
		# Update state estimate (mu) 
		# self.mu = self.mu + K @ (z - self.C() @ self.mu)
		# Update covariance (Sigma)
		# self.Sigma = (np.eye(len(K)) - K @ self.C() @ self.Sigma)

		return self.inf_vector, self.inf_matrix
	


'''
Execution time prediction: 0.00041556358337402344 seconds
Average exec time pred:  0.00024603640203510886
Execution time update: 0.00013828277587890625 seconds
Average exec time update:  0.00016492452376928084

Execution time prediction: 6.0558319091796875e-05 seconds
Average exec time pred:  0.00010180254995604575
Execution time update: 0.0002455711364746094 seconds
Average exec time update:  0.00028385194666656383

'''