import numpy as np
import scipy.stats

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


def odometry_observation_model_particles():
	def observation_model(particles, z, Q):
		# Calculate the error between the measured pose and each particle's pose
		error = particles - z
		
		# Handle angle wrapping for the heading error. This is crucial.
		# We subtract the angles and then wrap the result to the [-pi, pi] range.
		error[:, 2] = (error[:, 2] - z[2] + np.pi) % (2 * np.pi) - np.pi
		
		# Calculate the likelihood of the error using a multivariate normal PDF.
		# The mean of the error is [0, 0, 0].
		# This gives a high likelihood to particles with low error.
		likelihood = scipy.stats.multivariate_normal(mean=np.zeros(3), cov=Q).pdf(error)

		return likelihood
	
	return observation_model