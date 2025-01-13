import numpy as np

def odometry_observation_model():
	def observation_matrix_C():
		return np.eye(3)

	return observation_matrix_C

def odometry_observation_model_linearized():
	def observation_function_h(mu):
		return mu
	
	def jacobian_of_h_wrt_state_H(mu):
		return np.eye(3)

	return observation_function_h, jacobian_of_h_wrt_state_H