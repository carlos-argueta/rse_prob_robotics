import numpy as np
import scipy.stats

def odometry_imu_observation_model_linearized():
	def observation_function_h_v1(mu, u):
		# mu is [x, y, theta]
		# Predict sensor measurements purely based on the state estimate and perhaps physical models
		predicted_imu_theta = mu[2]  # Predicts IMU's theta should match the state's theta
		predicted_imu_omega = u[1]  # If omega from control is assumed to reflect actual omega

		# [x, y, theta, theta, w]
		return np.array([mu[0], mu[1], mu[2], predicted_imu_theta, predicted_imu_omega])


	def observation_function_h_v2(mu, previous_theta, delta_t):
		# Compute omega from theta
		delta_theta = mu[2] - previous_theta
		predicted_imu_omega = delta_theta / delta_t
		
		# Assume imu_theta measurement aligns with theta from state
		predicted_imu_theta = mu[2]  # This would be aligned with an actual IMU measurement for theta

		return np.array([[mu[0]], [mu[1]], [mu[2]], [predicted_imu_theta], [predicted_imu_omega]])


	
	def jacobian_of_h_wrt_state_H_v1():
		# Jacobian with respect to the state; note additional rows for IMU data
		# Here we assume theta from IMU aligns directly with theta from state
		H = np.zeros((5, 3))
		H[0:3, :] = np.eye(3)  # Odometry part affects x, y, theta
		H[3, 2] = 1  # IMU theta part affects state theta
		H[4, 2] = 0  # IMU omega does not depend on state directly in this simple model
		return H

	def jacobian_of_h_wrt_state_H_v2(delta_t):
		# Jacobian with respect to the state; note additional rows for IMU data
		# Here we assume theta from IMU aligns directly with theta from state
		H = np.zeros((5, 3))
		H[0:3, :] = np.eye(3)  # Odometry part affects x, y, theta
		H[3, 2] = 1  # IMU theta part affects state theta
		H[4, 2] = 1 / delta_t  # IMU omega does not depend on state directly in this simple model
		return H

	return observation_function_h_v2, jacobian_of_h_wrt_state_H_v2

def odometry_imu_observation_model_with_acceleration_motion_model_linearized_1():
	def observation_function_h(mu):
		x, y, theta, v, w, ax, ay = mu
		return np.array([[x], [y], [theta], [theta], [w], [ax], [ay]])
	
	def jacobian_of_h_wrt_state_H(mu):
		return np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],   	     
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],   	     
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  	     
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],   	     
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],   	     
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])   	 
                    

	return observation_function_h, jacobian_of_h_wrt_state_H

def odometry_imu_observation_model_with_acceleration_motion_model_linearized_2():
	def observation_function_h(mu):
		x, y, theta, v_x, v_y, w, ax, ay = mu
		return np.array([[x], [y], [theta], [theta], [w], [ax], [ay]])
	
	def jacobian_of_h_wrt_state_H(mu):
		return np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],       
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   	     
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],   	     
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  	         
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],   	     
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],   	     
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])   	     
                    

	return observation_function_h, jacobian_of_h_wrt_state_H

def odometry_imu_observation_model_with_acceleration_motion_model_no_input_linearized():
	def observation_function_h(mu):
		#return np.eye(8) @ mu
		x, y, theta, vx, vy, w, ax, ay = mu
		return np.array([[x], [y], [theta], [theta], [w], [ax], [ay]])
	
	def jacobian_of_h_wrt_state_H(mu):
		return np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # x = x
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   	 # y = y
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],   	 # theta = theta
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  	     # theta = theta_imu
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],   	 # omega = omega
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],   	 # a_x = a_x
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])   	 # a_y = a_y
                    
	return observation_function_h, jacobian_of_h_wrt_state_H


def odometry_imu_observation_model_with_acceleration_motion_model_3D():
	def observation_function_h(mu):
		x, y, z, roll, pitch, yaw, v_x, v_y, v_z, w_x, w_y, w_z, a_x, a_y, a_z = mu
		return np.array([[x], [y], [z], [roll], [pitch], [yaw], [roll], [pitch], [yaw], [w_x], [w_y], [w_z], [a_x], [a_y], [a_z]])
	
	return observation_function_h

'''
def odometry_imu_observation_model_particles():
	def observation_model(particles, z, Q):
		# Calculate the error between the measured pose and each particle's pose
		error = particles - z
		
		# Handle angle wrapping for the heading error. This is crucial.
		# We subtract the angles and then wrap the result to the [-pi, pi] range.
		error[:, 2] = (error[:, 2] - z[2] + np.pi) % (2 * np.pi) - np.pi
		
		# Calculate the likelihood of the error using a multivariate normal PDF.
		# The mean of the error is [0, 0, 0].
		# This gives a high likelihood to particles with low error.
		likelihood = scipy.stats.multivariate_normal(mean=np.zeros(8), cov=Q).pdf(error)

		return likelihood
	
	return observation_model
'''

def odometry_imu_observation_model_particles_1():
    def observation_model(particles, z, Q):
        z = z.reshape(-1)  # (7,)
        obs_indices = [0, 1, 2, 2, 5, 6, 7]  # x, y, theta(odom), theta(imu), w, ax, ay

        observed_particles = particles[:, obs_indices]
        error = observed_particles - z

        # Wrap every column that maps the angular state 'theta' (index 2 in the state)
        theta_cols = np.where(np.array(obs_indices) == 2)[0]  # could be [2, 3]
        for c in theta_cols:
            error[:, c] = (error[:, c] + np.pi) % (2 * np.pi) - np.pi

        # Now safe to evaluate with a 7x7 PD Q
        likelihood = scipy.stats.multivariate_normal(
            mean=np.zeros(7), cov=Q
        ).pdf(error)

        return likelihood
    return observation_model


def odometry_imu_observation_model_particles_2(beta=1.0, gate_mahal_sq=None):
    """
    Observation z = [x, y, theta_odom, theta_imu, w, a_x, a_y].
    Particles state = [x, y, theta, v_x, v_y, w, a_x, a_y].
    Q is 7x7 diagonal (pass as diag or full); use stds ~ realistic scales.

    beta: temperature in (0,1]; <1 softens likelihood to reduce collapse.
    gate_mahal_sq: if set (e.g., 25.0), downweights outliers via gating.
    """

    TWO_PI = 2.0 * np.pi
    LOG2PI = np.log(2.0 * np.pi)

    def normal_logpdf(residual, var):
        # residual, var: arrays broadcastable to same shape
        return -0.5 * (residual * residual / var + np.log(var) + LOG2PI)

    def observation_model(particles, z, Q):
        z = np.asarray(z, dtype=np.float64).reshape(-1)  # (7,)
        assert z.shape[0] == 7, "z must be 7D"

        # Build diagonal variances from Q (accept diag vector or full matrix)
        if Q.ndim == 1:
            var = np.asarray(Q, dtype=np.float64)
        else:
            var = np.asarray(np.diag(Q), dtype=np.float64)
        assert var.shape[0] == 7, "Q must be 7x7 or diag len 7"
        # Put a floor to avoid near-singular channels
        var = np.maximum(var, 1e-6)

        # Map state to measured components (same as your indices)
        obs_indices = np.array([0, 1, 2, 2, 5, 6, 7], dtype=int)
        Hx = particles[:, obs_indices]  # (N,7)

        # Residuals
        err = Hx - z  # (N,7)

        # Wrap BOTH theta residuals (cols where obs_indices == 2)
        theta_cols = np.where(obs_indices == 2)[0]
        for c in theta_cols:
            # wrap to (-pi, pi]
            err[:, c] = (err[:, c] + np.pi) % (TWO_PI) - np.pi

        # Per-dimension logpdf, then sum → total log-likelihood
        # (Kinda like diag-cov MVN but explicit and stable)
        # Broadcast var to (N,7)
        var_row = var.reshape(1, 7)
        logp_dim = normal_logpdf(err, var_row)  # (N,7)
        logp = np.sum(logp_dim, axis=1)         # (N,)

        # Optional mild tempering to reduce degeneracy (beta in (0,1])
        if beta != 1.0:
            logp = beta * logp

        # Optional gating by Mahalanobis distance
        # m^2 = sum(err^2 / var); gate e.g. at 5^2=25
        if gate_mahal_sq is not None:
            m2 = np.sum((err * err) / var_row, axis=1)
            # Soft gate: subtract a penalty for very large m^2
            # (Hard gate would set to -inf; soft is safer)
            too_far = m2 > gate_mahal_sq
            logp[too_far] += -0.5 * (m2[too_far] - gate_mahal_sq)

        # Stabilize: subtract max before exp
        logp -= np.max(logp)
        w = np.exp(logp)  # unnormalized weights proportional to likelihood
        # Return likelihoods; PF will normalize weights outside
        return w

    return observation_model


