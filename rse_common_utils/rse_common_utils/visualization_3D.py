import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Visualizer:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.gt_line, = self.ax.plot([], [], [], 'g-', label='Ground Truth')  # Green line for ground truth path
        self.kf_line, = self.ax.plot([], [], [], 'b-', label='Kalman Filter')  # Blue line for Kalman filter path
        self.obs_line, = self.ax.plot([], [], [], 'r-', label='Observation')  # Red line for Observation
        self.ax.legend()

        self.obs_path = []
        self.gt_path = []
        self.kf_path = []

        # Set axis labels
        self.ax.set_xlabel('X coordinate')
        self.ax.set_ylabel('Y coordinate')
        self.ax.set_zlabel('Z coordinate')
        self.ax.set_title('Kalman Filter Visualization')

        # Initialize min and max for x, y, and z
        self.x_min, self.x_max = float('inf'), float('-inf')
        self.y_min, self.y_max = float('inf'), float('-inf')
        self.z_min, self.z_max = float('inf'), float('-inf')

    def update_bounds(self, x, y, z):
        # Update the bounds with new x, y, and z values
        self.x_min = min(self.x_min, x)
        self.x_max = max(self.x_max, x)
        self.y_min = min(self.y_min, y)
        self.y_max = max(self.y_max, y)
        self.z_min = min(self.z_min, z)
        self.z_max = max(self.z_max, z)

        # Use a slight margin to ensure all data points are well within the plot
        margin = 5
        range_x = self.x_max - self.x_min
        range_y = self.y_max - self.y_min
        range_z = self.z_max - self.z_min
        data_range = max(range_x, range_y, range_z) / 2.0 + margin

        mid_x = (self.x_max + self.x_min) / 2.0
        mid_y = (self.y_max + self.y_min) / 2.0
        mid_z = (self.z_max + self.z_min) / 2.0

        self.ax.set_xlim(mid_x - data_range, mid_x + data_range)
        self.ax.set_ylim(mid_y - data_range)
        self.ax.set_zlim(mid_z - data_range, mid_z + data_range)

    def update(self, gt_pose, kf_pose, kf_cov=None, obs_pose=None, step="update"):
        # Update ground truth path
        self.gt_path.append(gt_pose[:3])
        gt_x, gt_y, gt_z = zip(*self.gt_path)
        self.gt_line.set_data(gt_x, gt_y)
        self.gt_line.set_3d_properties(gt_z)

        # Update Kalman filter path
        self.kf_path.append(kf_pose[:3])
        kf_x, kf_y, kf_z = zip(*self.kf_path)
        self.kf_line.set_data(kf_x, kf_y)
        self.kf_line.set_3d_properties(kf_z)

        # Update observation path if an observation is provided
        if obs_pose is not None:
            self.obs_path.append(obs_pose[:3])
            obs_x, obs_y, obs_z = zip(*self.obs_path)
            self.obs_line.set_data(obs_x, obs_y)
            self.obs_line.set_3d_properties(obs_z)

            # Update bounds with observation pose
            self.update_bounds(obs_pose[0], obs_pose[1], obs_pose[2])

        # Adjust plot limits if necessary
        self.update_bounds(gt_pose[0], gt_pose[1], gt_pose[2])
        self.update_bounds(kf_pose[0], kf_pose[1], kf_pose[2])

        # Update the covariance ellipse
        if kf_cov is not None:
            self._update_covariance_ellipse(kf_pose, kf_cov)

        # Draw the updated plot
        self.fig.canvas.draw_idle()
        plt.pause(0.01)

    def _update_covariance_ellipse(self, pose, cov):
        if cov.shape == (7, 7):  # If full covariance matrix, extract x, y, z part
            cov = cov[:3, :3]

        eigenvals, eigenvecs = np.linalg.eig(cov)

        width, height, depth = 2 * np.sqrt(eigenvals)  # Scale factor for visualization
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = width * np.outer(np.cos(u), np.sin(v)) + pose[0]
        y = height * np.outer(np.sin(u), np.sin(v)) + pose[1]
        z = depth * np.outer(np.ones(np.size(u)), np.cos(v)) + pose[2]

        # Clear previous ellipse data
        self.cov_ellipse.remove()
        self.cov_ellipse = Poly3DCollection([list(zip(x.flatten(), y.flatten(), z.flatten()))], edgecolor='black')
        self.ax.add_collection3d(self.cov_ellipse)


def main(args=None):
    # Example usage:
    viz = Visualizer()
    gt_pose = (10, 2, 3, 4, 5, 6)
    kf_pose = (1.1, 2.5, 4.1, 4, 5, 6)
    kf_cov = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    obs_pose = (1.05, 2.05, 2.55, 4)
    viz.update(gt_pose, kf_pose, None, obs_pose)
    plt.show()

if __name__ == '__main__':
    main()