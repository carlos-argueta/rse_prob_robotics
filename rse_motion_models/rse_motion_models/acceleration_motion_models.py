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

	def sample_acceleration_model_velocity(particles, u, dt, alphas):
		N = particles.shape[0]
		v_cmd, w_cmd = u
		x, y, theta, v_x, v_y, w, a_x, a_y = particles.T

		# Define noise for accelerations and angular velocity based on commands
		ax_var = alphas[0] * v_cmd**2 + alphas[1] * w_cmd**2
		ay_var = alphas[2] * v_cmd**2 + alphas[3] * w_cmd**2
		w_var = alphas[4] * v_cmd**2 + alphas[5] * w_cmd**2
		
		# Add noise to the highest-order terms
		a_x_new = a_x + np.random.randn(N) * np.sqrt(ax_var)
		a_y_new = a_y + np.random.randn(N) * np.sqrt(ay_var)
		w_new = w + np.random.randn(N) * np.sqrt(w_var)

		# Update velocities based on previous accelerations
		v_x_new = v_x + a_x * dt
		v_y_new = v_y + a_y * dt
		
		# Update pose based on previous velocities and new angular velocity
		cos_theta, sin_theta = np.cos(theta), np.sin(theta)
		dx_world = (v_x * cos_theta - v_y * sin_theta) * dt
		dy_world = (v_x * sin_theta + v_y * cos_theta) * dt
		
		x_new = x + dx_world
		y_new = y + dy_world
		theta_new = (theta + w_new * dt) % (2 * np.pi)

		return np.vstack([x_new, y_new, theta_new, v_x_new, v_y_new, w_new, a_x_new, a_y_new]).T

	return sample_acceleration_model_velocity

