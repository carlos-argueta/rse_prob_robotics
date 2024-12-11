# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import numpy as np
import time

from rse_common_utils.sensor_utils import map_shift, predict_range
from rse_common_utils.helper_utils import normalize_angle

from scipy.ndimage import gaussian_filter

from scipy.stats import norm
from scipy.stats import multivariate_normal

import math

class HistogramFilter:

    def __init__(self, initial_state, motion_model, observation_model, **kwargs):
        # Process arguments
        self.proc_noise_std = kwargs.get('proc_noise_std', 1.0)
        obs_noise_std = kwargs.get('obs_noise_std', [0.01, 0.01])

        # Standard deviations for the noise in x, y, and theta (observation or sensor model noise)
        self.obs_noise_std = np.array(obs_noise_std)
        # Observation noise covariance (Q)rse_gaussian_filters/rse_gaussian_filters/ukf_3d_state_estimation_v1_no_cmd.py
        self.obs_noise_cov = np.diag(self.obs_noise_std ** 2)

        self.belief = initial_state  # Initial state estimate [x, y, theta]
        self.dx = 0.0  # Accumulated motion in x
        self.dy = 0.0  # Accumulated motion in y

        self.A, self.B = motion_model()  # The action model to use. Returns A and B matrices

        # Observation model (C)
        self.C = observation_model()  # The observation model to use

        self.exec_times_pred = []
        self.exec_times_upd = []

        self.first_pass = True


    def predict(self, X, u, dt):
        start_time = time.time()

        # Predict state estimate (mu) 
        X = self.A() @ X + self.B(X, dt) @ u
        X[0] = dt * math.cos(X[2]) * u[0]
        X[1] = dt * math.sin(X[2]) * u[0]
        X[2] = dt * u[1]
        X[2] = normalize_angle(X[2])
        self.dx += X[0]
        self.dy += X[1]
        x_shift = self.dx // self.belief.resolution
        y_shift = self.dy // self.belief.resolution

        if abs(x_shift) >= 1.0 or abs(y_shift) >= 1.0:  # map should be shifted
            print("x_shift: ", x_shift)
            print("y_shift: ", y_shift)
            self.belief = map_shift(self.belief, int(x_shift),
                                      int(y_shift))
            self.dx -= x_shift * self.belief.resolution
            self.dy -= y_shift * self.belief.resolution

        # Add motion noise
        self.belief.data = gaussian_filter(self.belief.data,
                                             sigma=self.proc_noise_std)

        end_time = time.time()
        execution_time = end_time - start_time
        self.exec_times_pred.append(execution_time)
        print(f"Execution time prediction: {execution_time} seconds")
        print("Average exec time pred: ", 
              sum(self.exec_times_pred) / len(self.exec_times_pred))

        return self.belief, X

    def update(self, X, grid_map, lms, z):
        start_time = time.time()

        x, y, theta = X  # the robot pose
        (x_grid_robot, y_grid_robot) = self.belief.discretize(x, y)
        ranges, angles, _, max_range, _ = z  # the sensor data

        threshold = 1e-6  # Only consider cells with a belief higher than this

        total = 0
        if self.first_pass:
            relevant_cells = np.argwhere(self.belief.data > 0.0)
        else:
            rcs = np.argwhere(self.belief.data > threshold)
            total = len(rcs)
            rcs = rcs[np.argsort(self.belief.data[rcs[:, 0], rcs[:, 1]])[-30:]]
            relevant_cells = []
            for rc in rcs:
                x_grid, y_grid = rc
                relevant_cells.append([x_grid, y_grid])

        print("Relevant cells: ", relevant_cells, total)

        # for x_grid, y_grid in relevant_cells:
        # for x_grid in range(x_grid_robot - 10, x_grid_robot + 10):
        #     for y_grid in range(y_grid_robot - 10, y_grid_robot + 10):

        # Loop over all cells in the grid
        count = 0
        for x_grid in range(self.belief.data.shape[0]):
            for y_grid in range(self.belief.data.shape[1]):
                if self.belief.check_pixel(x_grid, y_grid) is False:
                    continue

                # print("Computing likelihood for", x_grid, y_grid, self.belief.data.shape)
                # Get the center of mass of the current cell
                cell_x, cell_y = self.belief.get_cell_center_mass(x_grid, y_grid)
                cell_theta = theta

                total_likelihood = 1.0

            
                # If coords of current grid within 10 cells of robot, compute likelihood
                if abs(x_grid - x_grid_robot) > 5 or abs(y_grid - y_grid_robot) > 5:
                # if self.first_pass and (abs(x_grid - x_grid_robot) > 3 or abs(y_grid - y_grid_robot) > 3):
                    total_likelihood = 0.000000000001
                #elif [x_grid, y_grid] not in relevant_cells:
                #    total_likelihood = 0.00000001   
                else:
                    count += 1
                    # print("processing a relevant cell", count, [x_grid, y_grid])
                    # Compute the expected measurement for each landmark
                    z_j_i = []
                    for lm in lms:
                        m_x_grid, m_y_grid = lm  # unpack the landmark position and signature
                        m_x, m_y = self.belief.get_cell_center_mass(m_x_grid, m_y_grid)
                        delta_x = m_x - cell_x
                        delta_y = m_y - cell_y
                        d = np.sqrt(delta_x**2 + delta_y**2)

                        if d <= 1.5:
                            # Expected measurement for this landmark
                            z_k = np.array([d, normalize_angle(np.arctan2(delta_y, delta_x) - cell_theta)])
                            z_j_i.append(z_k)
                    # print("z_j_i", len(z_j_i))
                    if len(z_j_i) == 0:
                        continue

                    # For every measurement
                    for r, theta in zip(ranges, angles):

                        if r <= max_range:
                            # Obtain the most likely correspondence
                            z_i = np.array([r, theta])
                            j_i, lk = self.correspondence(z_i, z_j_i)

                            z_k = lms[j_i]
                            # print("Most likely landmark: ", j_i, z_k, lk)

                            innovation = z_i - z_k
                            innovation[1] = normalize_angle(innovation[1])

                            # Calculate the likelihood of this observation-landmark pair
                            # likelihood = norm.pdf(innovation, 0.0, self.obs_noise_std)
                            likelihood = multivariate_normal.pdf(innovation,
                                                        mean=[0.0, 0.0],
                                                        cov=self.obs_noise_cov)

                            total_likelihood *= likelihood

                # Update the belief for this cell
                self.belief.data[x_grid, y_grid] = self.belief.data[x_grid, y_grid] * total_likelihood

        # Normalize the belief
        # self.belief.data /= np.sum(self.belief.data)
        self.belief.normalize_probability()
        mlkc = self.belief.get_most_likely_cell()
        # unique = np.unique(self.localization_state.data)
        print(self.belief.data)
        print("Most likely cell:", mlkc)
        X[0], X[1] = self.belief.get_cell_center_mass(mlkc[0], mlkc[1])

        end_time = time.time()
        execution_time = end_time - start_time
        self.exec_times_upd.append(execution_time)
        print(f"Execution time update: {execution_time} seconds")
        print("Average exec time update: ", sum(self.exec_times_upd) / len(self.exec_times_upd))

        self.first_pass = False

        return self.belief, X
    
    def correspondence(self, z_i, z_j_i):
        """
        Compute the most likely correspondence between the measured range and the expected measurements
        """

        likelihoods = []

        # print("z_j_i", z_j_i)

        # Calculate likelihood for each predicted measurement
        for z_k in z_j_i:
            # Compute the innovation (measurement residual)
            # print("comparing")
            # print(z_i)
            # print(z_k)
            innovation = z_i - z_k

            # Normalize the angle in the innovation
            innovation[1] = normalize_angle(innovation[1])

            # Calculate the likelihood of this observation-landmark pair
            # likelihood = norm.pdf(innovation, 0.0, self.obs_noise_std)

            # Example: Innovation is a 2D array,
            # and obs_noise_cov is the covariance matrix
            likelihood = multivariate_normal.pdf(innovation,
                                                 mean=[0.0, 0.0],
                                                 cov=self.obs_noise_cov)

            likelihoods.append(likelihood)

        # Find the index of the maximum likelihood
        j_i = np.argmax(likelihoods)
        
        return j_i, likelihoods[j_i]
        '''
        # likelihood
        pdf = norm.pdf(d - z[iz, 0], 0.0, std)

        min_diff = np.inf
        j_i = None

        for j, z_k in enumerate(z_j_i):
            diff = np.abs(z_k[0] - r)
            if diff < min_diff:
                min_diff = diff
                j_i = j

        return j_i

        '''
