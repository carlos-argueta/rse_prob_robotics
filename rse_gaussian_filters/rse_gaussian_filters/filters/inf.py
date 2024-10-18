import numpy as np
import time

class InformationFilter:

	def __init__(self, initial_state, initial_covariance, motion_model, observation_model, **kwargs):
		# Process arguments
		proc_noise_std = kwargs.get('proc_noise_std', [0.02, 0.02, 0.01])
		obs_noise_std = kwargs.get('obs_noise_std', [0.02, 0.02, 0.01])
	
		self.inf_vector = np.linalg.inv(initial_covariance) @ initial_state  # Initial information vector
		self.inf_matrix = np.linalg.inv(initial_covariance)  # Initial information matrix
		
		self.A, self.B = motion_model() # The action model to use. Returns A and B matrices
		
		# Standard deviations for the noise in x, y, and theta (process or action model noise)
		self.proc_noise_std = np.array(proc_noise_std)
		# Process noise covariance (R)
		self.R = np.diag(self.proc_noise_std ** 2)  # process noise covariance

		self.R_inv = np.linalg.inv(self.R)  # Inverse of process noise covariance

		# Observation model (C)
		self.C = observation_model() # The observation model to use

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
		'''
		Sigma = np.linalg.inv(self.inf_matrix)
		mu = Sigma @ self.inf_vector
		
		self.inf_matrix = np.linalg.inv(self.A() @ Sigma @ self.A().T + self.R)
		self.inf_vector = self.inf_matrix @ (self.A() @ Sigma @ self.inf_vector + self.B(mu, dt) @ u)
		'''

		
		Sigma = np.linalg.inv(self.inf_matrix)  # Inverse of the information matrix to get Sigma
		mu = Sigma @ self.inf_vector  # State mean

		A = self.A()  # State transition matrix
		
		# Apply the matrix inversion lemma
		temp = np.linalg.inv(Sigma + A.T @ self.R_inv @ A)  # (Omega_t-1 + A_t^T R_t^{-1} A_t)^{-1}
		self.inf_matrix = self.R_inv - self.R_inv @ A @ temp @ A.T @ self.R_inv  # Matrix inversion lemma

		# Update the information vector
		self.inf_vector = self.inf_matrix @ (A @ Sigma @ self.inf_vector + self.B(mu, dt) @ u)
		

		end_time = time.time()
		execution_time = end_time - start_time
		self.exec_times_pred.append(execution_time)
		print(f"Execution time prediction: {execution_time} seconds")

		print("Average exec time pred: ", sum(self.exec_times_pred) / len(self.exec_times_pred))

		return self.inf_vector, self.inf_matrix

	def update(self, z, dt):
		start_time = time.time()
		
		self.inf_matrix = self.inf_matrix + self.C().T @ self.Q_inv @ self.C()
		self.inf_vector = self.inf_vector + self.C().T @ self.Q_inv @ z

		end_time = time.time()
		execution_time = end_time - start_time
		self.exec_times_upd.append(execution_time)
		print(f"Execution time update: {execution_time} seconds")

		print("Average exec time update: ", sum(self.exec_times_upd) / len(self.exec_times_upd))
		
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