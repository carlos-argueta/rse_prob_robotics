import numpy as np

def velocity_motion_model():
	def state_transition_matrix_A():
		A = np.eye(3)

		return A

	def control_input_matrix_B(mu, delta_t):
		theta = mu[2]
		B = np.array([
			[np.cos(theta) * delta_t, 0],
			[np.sin(theta) * delta_t, 0],
			[0, delta_t]
		])

		return B

	return state_transition_matrix_A, control_input_matrix_B

def velocity_motion_model_linearized_1():
	print("Using VMML1")

	def state_transition_function_g(mu = None, u = None, delta_t = None):
		x = mu[0]
		y = mu[1]
		theta = mu[2]
		
		v = u[0]       
		w = u[1]       
		
		g = np.array([
            x + v * np.cos(theta) * delta_t,
            y + v * np.sin(theta) * delta_t,
            theta + w * delta_t
        ])

		return g

	def jacobian_of_g_wrt_state_G(mu = None, u = None, delta_t = None):
		theta = mu[2]
		v = u[0]       
		w = u[1]       
		
		G = np.array([
			[1, 0, -v * np.sin(theta) * delta_t],
			[0, 1, v * np.cos(theta) * delta_t],
			[0, 0, 1]
		])

		return G

	def jacobian_of_g_wrt_control_V(mu = None, u = None, delta_t = None):
		theta = mu[2]
		v = u[0]       
		w = u[1]       
		
		V = np.array([
			[np.cos(theta) * delta_t, 0],
			[np.sin(theta) * delta_t, 0],
			[0, delta_t]
		])

		return V

	return state_transition_function_g, jacobian_of_g_wrt_state_G, jacobian_of_g_wrt_control_V


def velocity_motion_model_linearized_2():
	print("Using VMML2")

	def state_transition_function_g(mu = None, u = None, delta_t = None):
		x = mu[0]
		y = mu[1]
		theta = mu[2]
		
		v = u[0]       
		w = u[1]       
		if w == 0:
			w = 1e-6   # Avoid division by zero for straight-line motion

		g = np.array([
		   x + -v/w * np.sin(theta) + v/w * np.sin(theta + w * delta_t),
		   y + v/w * np.cos(theta) - v/w * np.cos(theta + w * delta_t),
		   theta + w * delta_t
		])

		return g

	def jacobian_of_g_wrt_state_G(mu = None, u = None, delta_t = None):
		theta = mu[2]
		v = u[0]       
		w = u[1]       
		if w == 0:
			w = 1e-6   # Avoid division by zero for straight-line motion

		G = np.array([
			[1, 0, -v / w * np.cos(theta) + v / w * np.cos(theta + w * delta_t)],
			[0, 1, -v / w * np.sin(theta) + v / w * np.sin(theta + w * delta_t)],
			[0, 0, 1]
		])

		return G

	def jacobian_of_g_wrt_control_V(mu = None, u = None, delta_t = None):
		theta = mu[2]
		v = u[0]       # Linear velocity
		w = u[1]       # Angular velocity
		if w == 0:
			w = 1e-6   # Avoid division by zero for straight-line motion

		V = np.array([
			[(-np.sin(theta) + np.sin(theta + w * delta_t)) / w, v * (np.sin(theta) - np.sin(theta + w * delta_t)) / (w ** 2) + v * np.cos(theta + w * delta_t) * delta_t / w],
			[(np.cos(theta) - np.cos(theta + w * delta_t)) / w, -v * (np.cos(theta) - np.cos(theta + w * delta_t)) / (w ** 2) + v * np.sin(theta + w * delta_t) * delta_t / w],
			[0, delta_t]
		])

		return V

	
	return state_transition_function_g, jacobian_of_g_wrt_state_G, jacobian_of_g_wrt_control_V


def velocity_motion_model_particles():

	def sample_motion_model_velocity(particles, u, dt, alphas):
		N = len(particles)
		v, w = u

		# Extract current particle states
		x = particles[:, 0]
		y = particles[:, 1]
		theta = particles[:, 2]

		# Add noise to the control inputs
		# The noise variance is proportional to the commanded velocities
		# Note: The book uses a complex sampling method. Using np.random.randn
		# to sample from a normal distribution with the specified variance is
		# a more direct and common approach.
		v_var = alphas[0] * v**2 + alphas[1] * w**2
		w_var = alphas[2] * v**2 + alphas[3] * w**2
		gamma_var = alphas[4] * v**2 + alphas[5] * w**2

		v_hat = v + (np.random.randn(N) * np.sqrt(v_var))
		w_hat = w + (np.random.randn(N) * np.sqrt(w_var))
		gamma_hat = np.random.randn(N) * np.sqrt(gamma_var)

		# Pre-calculate common terms
		theta_new = theta + w_hat * dt
		v_div_w = v_hat / w_hat

		# Handle the special case where angular velocity is close to zero
		# In this case, the robot moves in a straight line.
		# We use np.isclose to handle floating point comparisons safely.
		straight_mask = np.isclose(w_hat, 0.0)
		curved_mask = ~straight_mask

		# --- Update particles for the curved path (w_hat is not zero) ---
		x_new = np.zeros(N)
		y_new = np.zeros(N)

		x_new[curved_mask] = x[curved_mask] - v_div_w[curved_mask] * np.sin(theta[curved_mask]) + \
							v_div_w[curved_mask] * np.sin(theta_new[curved_mask])
		y_new[curved_mask] = y[curved_mask] + v_div_w[curved_mask] * np.cos(theta[curved_mask]) - \
							v_div_w[curved_mask] * np.cos(theta_new[curved_mask])

		# --- Update particles for the straight path (w_hat is zero) ---
		x_new[straight_mask] = x[straight_mask] + v_hat[straight_mask] * dt * np.cos(theta[straight_mask])
		y_new[straight_mask] = y[straight_mask] + v_hat[straight_mask] * dt * np.sin(theta[straight_mask])

		# --- Update heading for all particles ---
		# Add the final gamma noise and wrap to [0, 2*pi]
		theta_final = theta_new + gamma_hat * dt
		theta_final %= 2 * np.pi

		# Return the new particle set
		return np.vstack((x_new, y_new, theta_final)).T
	
	return sample_motion_model_velocity
