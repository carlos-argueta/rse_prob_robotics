# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

import numpy as np
import math

class Visualizer:
    def __init__(self, title = "Figure 1"):
        self.fig, self.ax = plt.subplots()
        self.gt_line, = self.ax.plot([], [], 'g-', label='Ground Truth')  # Green line for ground truth path
        self.kf_line, = self.ax.plot([], [], 'b-', label='Kalman/Information Filter')  # Red line for Kalman filter path
        self.obs_line, = self.ax.plot([], [], 'r-', label='Observation')
        self.ax.legend()

        self.obs_path = [] 
        self.gt_path = []
        self.kf_path = []

        # Ellipse to represent the covariance matrix
        self.cov_ellipse = Ellipse(xy=(0, 0), width=0, height=0, edgecolor='black', fc='None', lw=2)
        self.ax.add_patch(self.cov_ellipse)

        # Set axis labels
        self.ax.set_xlabel('X coordinate')
        self.ax.set_ylabel('Y coordinate')
        self.ax.set_title('Kalman Filter Visualization')

        # Initialize min and max for x and y
        self.x_min, self.x_max = float('inf'), float('-inf')
        self.y_min, self.y_max = float('inf'), float('-inf')

        plt.title(title)

    def update_bounds(self, x, y):
        # Update the bounds with new x and y values
        self.x_min = min(self.x_min, x)
        self.x_max = max(self.x_max, x)
        self.y_min = min(self.y_min, y)
        self.y_max = max(self.y_max, y)

        # Use a slight margin to ensure all data points are well within the plot
        margin = 5
        range_x = self.x_max - self.x_min
        range_y = self.y_max - self.y_min
        data_range = max(range_x, range_y) / 2.0 + margin

        mid_x = (self.x_max + self.x_min) / 2.0
        mid_y = (self.y_max + self.y_min) / 2.0

        self.ax.set_xlim(mid_x - data_range, mid_x + data_range)
        self.ax.set_ylim(mid_y - data_range, mid_y + data_range)

    def update(self, gt_pose, kf_pose, kf_cov = None, obs_pose = None, step = "update"):
        # Update ground truth path
        self.gt_path.append(gt_pose[:2])
        gt_x, gt_y = zip(*self.gt_path)
        self.gt_line.set_data(gt_x, gt_y)

        # Update Kalman filter path
        self.kf_path.append(kf_pose[:2])
        kf_x, kf_y = zip(*self.kf_path)
        self.kf_line.set_data(kf_x, kf_y)

        # Update observation path if an observation is provided
        if obs_pose is not None:
            self.obs_path.append(obs_pose[:2])
            obs_x, obs_y = zip(*self.obs_path)
            self.obs_line.set_data(obs_x, obs_y)

            # Update bounds with observation pose
            self.update_bounds(obs_pose[0], obs_pose[1])
        
        # Adjust plot limits if necessary
        # self.ax.set_xlim(min(gt_x + kf_x) - 1, max(gt_x + kf_x) + 1)
        # self.ax.set_ylim(min(gt_y + kf_y) - 1, max(gt_y + kf_y) + 1)
        # Call update_bounds for the new points
        self.update_bounds(gt_pose[0], gt_pose[1])
        # print(kf_pose)
        self.update_bounds(kf_pose[0], kf_pose[1])

        # Update the covariance ellipse
        if kf_cov is not None:
            ellipse_color = 'black' if step == "update" else 'gray'
            self._update_covariance_ellipse(kf_pose, kf_cov, ellipse_color)

        # Set the aspect ratio to equal after updating bounds
        # self.ax.set_aspect('equal', adjustable='datalim')

        # Draw the updated plot
        self.fig.canvas.draw_idle()
        plt.pause(0.01)

    def _update_covariance_ellipse(self, pose, cov, color):
        if cov.shape == (3, 3):  # If full covariance matrix, extract x, y part
            cov = cov[:2, :2]

        eigenvals, eigenvecs = np.linalg.eig(cov)
        #angle = np.rad2deg(np.arctan2(*eigenvecs[:, 0][::-1]))
        try:
            angle = np.rad2deg(np.arctan2(*eigenvecs[:, 0][::-1]))
        except TypeError:
            # Handle the case where input is complex
            angle = np.rad2deg(np.arctan2(np.real(eigenvecs[:, 0][1]), np.real(eigenvecs[:, 0][0])))

        # print("eig", eigenvals, 'w,h',2 * np.sqrt(eigenvals))
        width, height = 2 * np.sqrt(eigenvals)[:2]  # Scale factor for visualization
        self.cov_ellipse.set_center((pose[0], pose[1]))
        self.cov_ellipse.width = width
        self.cov_ellipse.height = height
        self.cov_ellipse.angle = angle
        self.cov_ellipse.set_edgecolor(color)

class UKFVisualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.mu_path = []
        self.sigma_points = []

        # Initialize plot elements
        self.trajectory_line, = self.ax.plot([], [], 'g-', label='Trajectory')
        self.sigma_points_plot = self.ax.scatter([], [], c='red', label='Sigma Points', s=1)
        self.cov_ellipse = Ellipse(xy=(0, 0), width=0, height=0, edgecolor='blue', fc='None', lw=2)
        self.ax.add_patch(self.cov_ellipse)

        self.ax.legend()
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('UKF Visualization')

        plt.ion()
        plt.show()

    def update(self, mu, Sigma, sigma_points, Wm):
        # Update trajectory
        self.mu_path.append(mu[:2])
        mu_x, mu_y = zip(*self.mu_path)
        self.trajectory_line.set_data(mu_x, mu_y)

        # Update sigma points
        self.sigma_points = sigma_points[:, :2]
        self.sigma_points_plot.set_offsets(self.sigma_points)
        sizes = Wm * 1000  # Scale weights for visualization
        self.sigma_points_plot.set_sizes(sizes)

        # Update covariance ellipse
        self.update_covariance_ellipse(mu[:2], Sigma[:2, :2])

        # Center the plot around the current mu
        self.ax.set_xlim(mu[0] - 5, mu[0] + 5)
        self.ax.set_ylim(mu[1] - 5, mu[1] + 5)

        # Draw the updated plot
        self.fig.canvas.draw_idle()
        plt.pause(0.01)

    def update_covariance_ellipse(self, mu, cov):
        eigenvals, eigenvecs = np.linalg.eig(cov)
        angle = np.rad2deg(np.arctan2(*eigenvecs[:, 0][::-1]))
        width, height = 2 * np.sqrt(eigenvals)  # 2 standard deviations

        self.cov_ellipse.set_center(mu)
        self.cov_ellipse.width = width
        self.cov_ellipse.height = height
        self.cov_ellipse.angle = angle

    def show(self):
        plt.show()