# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import numpy as np

def acceleration_motion_model_linearized_1():

	def state_transition_function_g(mu = None, u = None, delta_t = None):
		
		x, y, theta, v, w, a_x, a_y = mu

		v = u[0]      
		w = u[1] 
		
		g = np.array([
			x + v * np.cos(theta) * delta_t + 0.5 * a_x * delta_t**2,      
			y + v * np.sin(theta) * delta_t + 0.5 * a_y * delta_t**2,    
			theta + w * delta_t,
			v + a_x * np.cos(theta) * delta_t + a_y * np.sin(theta) * delta_t,
			w,                                                      
			a_x,                                                              
			a_y
		])

		return g

	def jacobian_of_g_wrt_state_G(mu = None, u = None, delta_t = None):
		x, y, theta, v, w, a_x, a_y = mu

		v = u[0]       
		w = u[1]       

		G = np.array([[1.0, 0.0, -delta_t * v * np.sin(theta), delta_t  * np.cos(theta), 0.0, 0.5*delta_t**2, 0.0],   
				   [0.0, 1.0, delta_t * v * np.cos(theta), delta_t * np.sin(theta), 0.0, 0.0, 0.5*delta_t**2],       
				   [0.0, 0.0, 1.0, 0.0, delta_t, 0.0, 0.0],                                      
				   [0.0, 0.0, -delta_t * a_x * np.sin(theta) + delta_t * a_y * np.cos(theta), 
				   1.0, 0.0, delta_t * np.cos(theta), delta_t * np.sin(theta)],                  
				   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],                                          
				   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],                                          
				   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])                                         
		
		return G

	def jacobian_of_g_wrt_control_V(mu = None, u = None, delta_t = None):
		theta = mu[2]

		V = np.array([
			[delta_t * np.cos(theta), 0],
			[delta_t * np.sin(theta), 0],
			[0, delta_t],
			[1, 0],
			[0, 1],
			[0, 0],
			[0, 0],
		])

		return V
	
	return state_transition_function_g, jacobian_of_g_wrt_state_G, jacobian_of_g_wrt_control_V

def acceleration_motion_model_linearized_2():

	def state_transition_function_g(mu = None, u = None, delta_t = None):
		
		x, y, theta, v_x, v_y , w, a_x, a_y = mu

		v = u[0]       
		w = u[1]       
		
		g = np.array([
			x + v * np.cos(theta) * delta_t + 0.5 * a_x * delta_t**2,  
			y + v * np.sin(theta) * delta_t + 0.5 * a_y * delta_t**2,  
			theta + w * delta_t,                                   
			v * np.cos(theta) + a_x * delta_t,                         
			v * np.sin(theta) + a_y * delta_t,                         
			w,                                                         
			a_x,                                                       
			a_y                                                        
		])

		return g

	def jacobian_of_g_wrt_state_G(mu = None, u = None, delta_t = None):
		x, y, theta, v_x, v_y, w, a_x, a_y = mu

		v = u[0]       
		w = u[1]       

		G = np.array([[1.0, 0.0, -delta_t * v * np.sin(theta), 0.0, 0.0, 0.0, 0.5*delta_t**2, 0.0],   
				   [0.0, 1.0, delta_t * v * np.cos(theta), 0.0, 0.0, 0.0, 0.0, 0.5*delta_t**2],        
				   [0.0, 0.0, 1.0, 0.0, 0.0, delta_t, 0.0, 0.0],                                      
				   [0.0, 0.0, -v * np.sin(theta), 0.0, 0.0, 0.0, delta_t, 0.0],                       
				   [0.0, 0.0, v * np.cos(theta), 0.0, 0.0, 0.0, 0.0, delta_t],                        
				   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],                                          
				   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],                                          
				   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])                                         
		
		return G

	def jacobian_of_g_wrt_control_V(mu = None, u = None, delta_t = None):
		theta = mu[2]

		V = np.array([
			[delta_t * np.cos(theta), 0],
			[delta_t * np.sin(theta), 0],
			[0, delta_t],
			[np.cos(theta), 0],
			[np.sin(theta), 0],
			[0, 1],
			[0, 0],
			[0, 0],
		])

		return V
	
	return state_transition_function_g, jacobian_of_g_wrt_state_G, jacobian_of_g_wrt_control_V

def acceleration_motion_model_no_control_linearized():

	def state_transition_function_g(mu = None, u = None, delta_t = None):
		
		x, y, theta, vx, vy, w, ax, ay = mu
		
		g = np.array([
			x + vx * delta_t + 0.5 * ax * delta_t**2,
			y + vy * delta_t + 0.5 * ay * delta_t**2,
			theta + w * delta_t,
			vx + ax * delta_t,
			vy + ay * delta_t,
			w,
			ax,
			ay,
		])

		return g

	def jacobian_of_g_wrt_state_G(mu = None, u = None, delta_t = None):
		G = np.array([[1.0, 0.0, 0.0, delta_t, 0.0, 0.0, 0.5*delta_t**2, 0.0],   # x = x + v_x * dt  + 0.5 * a_x * dt^2
				   [0.0, 1.0, 0.0, 0.0, delta_t, 0.0, 0.0, 0.5*delta_t**2],      # y = y + v_y * dt  + 0.5 * a_y * dt^2
				   [0.0, 0.0, 1.0, 0.0, 0.0, delta_t, 0.0, 0.0],                 # theta = theta + omega * dt 
				   [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, delta_t, 0.0],                 # v_x = v_x + a_x * dt
				   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, delta_t],                 # v_y = v_y + a_y * dt
				   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],                     # omega = omega
				   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],                     # a_x = a_x
				   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])                    # a_y = a_y
		
		return G

	def jacobian_of_g_wrt_control_V(mu = None, u = None, delta_t = None):
		V = np.array([
			[0, 0],
			[0, 0],
			[0, delta_t],
			[0, 0],
			[0, 0],
			[0, 1],
			[0, 0],
			[0, 0]
		])

		return V
	
	return state_transition_function_g, jacobian_of_g_wrt_state_G, jacobian_of_g_wrt_control_V


def acceleration_motion_model_3D():

	def state_transition_function_g(mu = None, u = None, delta_t = None):
		
		x, y, z, roll, pitch, yaw, v_x, v_y, v_z, w_x, w_y, w_z, a_x, a_y, a_z = mu
		a_x, a_y, a_z, w_x, w_y, w_z = u
		
		g = np.array([
			x + v_x * delta_t + 0.5 * a_x * delta_t**2,  # x position update
			y + v_y * delta_t + 0.5 * a_y * delta_t**2,  # y position update
			z + v_z * delta_t + 0.5 * a_z * delta_t**2,  # z position update
			roll + w_x * delta_t,      # roll update
			pitch + w_y * delta_t,      # pitch update
			yaw + w_z * delta_t,  # yaw update
		
			v_x + a_x * delta_t,            # x velocity update
			v_y + a_y * delta_t,            # y velocity update
			v_z + a_z * delta_t,            # z velocity update
			w_x,
			w_y,
			w_z,
			a_x,                            # constant acceleration in x
			a_y,                            # constant acceleration in y
			a_z                             # constant acceleration in z
		])

		return g
	
	return state_transition_function_g


def acceleration_motion_model_particles():

	def sample_acceleration_model_velocity(particles, u, dt, params=None):
		print("Params", params)
		
		v_cmd, w_cmd = u  # scalars or arrays length N
		N = particles.shape[0]
		out = particles.copy()

		# Broadcast commands
		if np.isscalar(v_cmd): v_cmd = np.full(N, float(v_cmd))
		if np.isscalar(w_cmd): w_cmd = np.full(N, float(w_cmd))

		# Parameters for the motion model
		tau_v  = params[0]    # speed response
		tau_w  = params[1]    # yaw response
		tau_vy = params[2]    # lateral velocity damping
		tau_ay = params[3]    # lateral accel damping

		# Process noise (per sqrt(sec))
		sig_ax =  params[4]   # accel noise x  (m/s^2)
		sig_ay =  params[5]   # accel noise y  (m/s^2)
		sig_w  = params[6]   # yaw-rate noise (rad/s)
		sig_vx = params[7]   # direct vel noise (m/s)
		sig_vy = params[8]
		sig_p  = params[9]  # position noise (m)
		sig_th = params[10]  # heading integration noise (rad)

		# Unpack
		x, y, th, vx, vy, w, ax, ay = [out[:, i] for i in range(8)]

		# 1) Command tracking (first-order)
		#    ax drives vx -> v_cmd; w tracks w_cmd directly.
		#    Discretize with Euler–Maruyama, add process noise.
		ax_noise = sig_ax * np.sqrt(dt) * np.random.randn(N)
		ay_noise = sig_ay * np.sqrt(dt) * np.random.randn(N)
		w_noise  = sig_w  * np.sqrt(dt) * np.random.randn(N)

		# a_x tries to reduce (vx - v_cmd) with time constant tau_v
		ax_new = ax + (-(vx - v_cmd)/tau_v - ax/tau_v) * dt + ax_noise
		# a_y damped toward 0 (no lateral command; slip captured by noise)
		ay_new = ay + (-ay / tau_ay) * dt + ay_noise

		# yaw rate tracks w_cmd with time constant tau_w
		w_new  = w + (-(w - w_cmd)/tau_w) * dt + w_noise

		# 2) Integrate body-frame velocities
		vx_noise = sig_vx * np.sqrt(dt) * np.random.randn(N)
		vy_noise = sig_vy * np.sqrt(dt) * np.random.randn(N)

		vx_new = vx + ax_new * dt + vx_noise
		# lateral velocity softly damped to 0 as well
		vy_new = vy + ay_new * dt - (vy / tau_vy) * dt + vy_noise

		# 3) Integrate heading (wrap to (-pi, pi])
		th_new = th + w_new * dt + sig_th * np.sqrt(dt) * np.random.randn(N)
		th_new = (th_new + np.pi) % (2*np.pi) - np.pi

		# 4) World-frame position update using rotated body velocities & accel
		c, s = np.cos(th), np.sin(th)
		vwx = c * vx_new - s * vy_new
		vwy = s * vx_new + c * vy_new

		awx = c * ax_new - s * ay_new
		awy = s * ax_new + c * ay_new

		x_new = x + vwx * dt + 0.5 * awx * dt*dt + sig_p * np.sqrt(dt) * np.random.randn(N)
		y_new = y + vwy * dt + 0.5 * awy * dt*dt + sig_p * np.sqrt(dt) * np.random.randn(N)

		# Pack
		out[:, 0] = x_new
		out[:, 1] = y_new
		out[:, 2] = th_new
		out[:, 3] = vx_new
		out[:, 4] = vy_new
		out[:, 5] = w_new
		out[:, 6] = ax_new
		out[:, 7] = ay_new
		return out

	return sample_acceleration_model_velocity

