import numpy as np

def velocity_motion_model():
	def state_transition_matrix_A():
		A = np.eye(3)

		return A

	def control_input_matrix_B(mu, delta_t):
		theta = mu[2]
		print(theta, delta_t)
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
