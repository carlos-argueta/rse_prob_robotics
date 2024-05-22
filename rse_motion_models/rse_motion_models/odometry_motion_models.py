import numpy as np 

from rse_common_utils.helper_utils import normalize_angle
from rse_common_utils.sampling_utils import sample_normal_distribution

def odometry_motion_model():
	def state_transition_matrix_A():
		A = np.eye(3)

		return A

	def control_input_matrix_B(mu, u, alpha_1 = 0.01, alpha_2 = 0.001, alpha_3 = 0.001, alpha_4 = 0.01):
		theta = mu[2]
		robot_pose_t_minus_1 = u[0]
		robot_pose_t = u[1]

		x_hat = robot_pose_t_minus_1[0]
		y_hat = robot_pose_t_minus_1[1]
		theta_hat = robot_pose_t_minus_1[2]

		x_hat_prime = robot_pose_t[0]
		y_hat_prime = robot_pose_t[1]
		theta_hat_prime = robot_pose_t[2]

		# Compute parameters from robot odometry
		d_rot1 = np.arctan2(y_hat_prime - y_hat, x_hat_prime - x_hat) - theta_hat
		d_rot1 = normalize_angle(d_rot1)

		d_trans = np.sqrt((x_hat - x_hat_prime) ** 2 + (y_hat - y_hat_prime) ** 2)

		d_rot2 = theta_hat_prime - theta_hat - d_rot1
		d_rot2 = normalize_angle(d_rot2)

		d_rot1 = d_rot1 - sample_normal_distribution(alpha_1 * d_rot1 ** 2 + alpha_2 * d_trans ** 2)
		d_trans = d_trans - sample_normal_distribution( alpha_3 * d_trans ** 2 + alpha_4 * d_rot1 ** 2 + alpha_4 * d_rot2 ** 2)
		d_rot2 = d_rot2 - sample_normal_distribution(alpha_1 * d_rot2 ** 2 + alpha_2 * d_trans ** 2)

		B = np.array([
			d_trans * np.cos(theta + d_rot1),
			d_trans * np.sin(theta + d_rot1),
			d_rot1 + d_rot2
		])

		return B

	return state_transition_matrix_A, control_input_matrix_B