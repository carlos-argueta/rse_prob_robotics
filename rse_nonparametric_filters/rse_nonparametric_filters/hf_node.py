import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import OccupancyGrid

from rse_map_models.grid_map import GridMap
from rse_common_utils.sensor_utils import lidar_scan

from rse_motion_models.velocity_motion_models import velocity_motion_model
from rse_observation_models.odometry_observation_models import odometry_observation_model

from .filters.hf import HistogramFilter

import PyKDL

import matplotlib.pyplot as plt
import cv2
import json

import sys
import signal

import numpy as np


class HistogramFilterNode(Node):
    def __init__(self):
        super().__init__('histogram_filter_node')

        # Load the map from JSON
        self.grid_map = self.load_map_from_json(
            "/home/carlos/pr_ws/src/rse_prob_robotics/maps/grid_map_0.1.json")

        self.landmarks = self.grid_map.get_occupied_cells()
        print(f"Landmarks: {self.landmarks}")

        self.X = [0.0, 0.0, 0.0]  # Initial state [x, y, theta]

        self.localization_state = GridMap(
            X_lim=[self.grid_map.min_x, self.grid_map.max_x],
            Y_lim=[self.grid_map.min_y, self.grid_map.max_y],
            resolution=self.grid_map.resolution,
            initial_p=1.0,
            normalize=True)

        print(f"Initial state: {self.localization_state.data}")

        self.scan_subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.laser_scan_callback,
            10)

        self.odometry_subscription = self.create_subscription(
            Odometry,
            'odom',
            self.odometry_callback,
            10)

        self.map_publisher = self.create_publisher(OccupancyGrid, 'map', 10)

        self.image_publisher = self.create_publisher(Image, 'map_image', 10)

        map_update_timer_period = 0.5  # seconds
        self.map_update_timer = self.create_timer(map_update_timer_period, 
                                                  self.map_update_callback)

        self.scan = None
        self.odom = None
        self.first_odom = True

        self.prev_predict_time = None  # previous prediction time, used to compute the delta_t
        self.prev_update_time = None  # previous update time, used to compute the delta_t

        self.hf = HistogramFilter(self.localization_state,
                                  velocity_motion_model,
                                  odometry_observation_model,
                                  proc_noise_std=1.0,
                                  obs_noise_std=[0.1, 0.1])

        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.handle_shutdown)

    def handle_shutdown(self, signum, frame):
        """
        Handle Ctrl-C for graceful shutdown.
        """
        self.get_logger().info('Shutting down...')
        response = input("Do you want to save the map? (y/n): ").strip().lower()
        if response == 'y':
            filename = input("Enter the filename to save the map to (default: grid_map.json): ").strip()
            if filename == "":
                filename = "grid_map.json"
            self.save_map_to_json(filename)
        self.destroy_node()
        sys.exit(0)

    # Load the map from JSON
    def load_map_from_json(self, filename="grid_map_0.1.json"):
        with open(filename, 'r') as file:
            map_data = json.load(file)
        grid_map = GridMap(
            X_lim=map_data["X_lim"],
            Y_lim=map_data["Y_lim"],
            resolution=map_data["resolution"],
            initial_p=np.array(map_data["log_odds"])  # Convert list back to numpy array
        )
        print(f"Map loaded from {filename}")
        return grid_map

    def laser_scan_callback(self, msg):
        self.scan = msg

        if self.first_odom:
            return

        self.z = lidar_scan(self.scan)

        # Compute dt
        curr_time = self.get_clock().now().nanoseconds
        if self.prev_update_time:
            dt = (curr_time - self.prev_update_time) / 1e9
        else:
            dt = 0.01

        self.localization_state, _ = self.hf.update(self.X,
                                                         self.grid_map,
                                                         self.landmarks,
                                                         self.z)

        # print(f"localization_state: {self.localization_state.data}")
        
        heatmap = self.create_heatmap(self.localization_state.data)

        # Enlarge the image if smaller than 640x640
        if heatmap.shape[0] < 640 or heatmap.shape[1] < 640:
             heatmap = cv2.resize(heatmap, (640, 640), interpolation=cv2.INTER_NEAREST)

        # Show the heatmap using OpenCV
        cv2.imshow("Heatmap", heatmap)
        cv2.waitKey(1)  # Wait for a key press
        
        self.prev_update_time = curr_time

    def odometry_callback(self, msg):
        self.odom = msg

        self.u = np.asarray([self.odom.twist.twist.linear.x,
                             self.odom.twist.twist.angular.z])
        
        
        if self.first_odom:
            self.X = self.odom_to_x()
            
            self.first_odom = False
        
        print(f"u: {self.u}")
        print(f"X: {self.X}")

        # Compute dt
        curr_time = self.get_clock().now().nanoseconds
        if self.prev_predict_time:
            dt = (curr_time - self.prev_predict_time) / 1e9  # Convert nanoseconds to seconds
        else:
            dt = 0.01

        self.localization_state, self.X = self.hf.predict(self.X, self.u, dt)
        # print(f"localization_state: {self.localization_state.data}")

        heatmap = self.create_heatmap(self.localization_state.data)

        # Enlarge the image if smaller than 640x640
        if heatmap.shape[0] < 640 or heatmap.shape[1] < 640:
             heatmap = cv2.resize(heatmap, (640, 640), interpolation=cv2.INTER_NEAREST)

        # Show the heatmap using OpenCV
        cv2.imshow("Heatmap", heatmap)
        cv2.waitKey(1)  # Wait for a key press

        # gs_image = self.localization_state.to_grayscale_image()

        # Enlarge the image if smaller than 640x640
        # if gs_image.shape[0] < 640 or gs_image.shape[1] < 640:
        #     gs_image = cv2.resize(gs_image, (640, 640), interpolation=cv2.INTER_NEAREST)

        # cv2.imshow("Map Image", gs_image)
        # cv2.waitKey(1)  # Refresh the display window

        self.prev_predict_time = curr_time

    def map_update_callback(self):

        if self.scan is None or self.odom is None:
            return

        z_t = lidar_scan(self.scan)
        x_t = self.odom_to_x()

        # self.grid_mapping.update_grid_with_sensor_reading(self.grid_map, x_t, z_t)

        occ_grid = self.grid_map.to_ros_occupancy_grid()
        gs_image = self.grid_map.to_grayscale_image()
        #print(occ_grid)

        # map_msg = OccupancyGrid()
        # Configure map_msg here (header, info, data)
        # Example: Create a 10x10 map with all cells free
        # map_msg.header.frame_id = 'map'
        # map_msg.info.width = self.grid_map.width
        # map_msg.info.height = self.grid_map.height
        # map_msg.info.resolution = self.grid_map.resolution
        # map_msg.data = occ_grid  # -1 indicates unknown, 0 free, 100 occupied
        # self.map_publisher.publish(map_msg)
        # self.get_logger().info('Publishing map')

        # image_msg = self.occupancy_grid_to_image(map_msg)
        # image_msg.header.frame_id = 'map'
        # self.image_publisher.publish(image_msg)

        #cv_image = ros_image_to_opencv(msg)
        # plt.imshow(gs_image, cmap='gray')
        # plt.title("Map Image")
        # plt.show()
        # cv2.imshow("Map Image", gs_image)
        # cv2.waitKey(1)  # Refresh the display window

    def odom_to_x(self):
        return (self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.quat_to_yaw(self.odom.pose.pose.orientation))

    def quat_to_yaw(self, quat):
        rot = PyKDL.Rotation.Quaternion(quat.x, quat.y, quat.z, quat.w)
        return rot.GetRPY()[2]

    def occupancy_grid_to_image(self, occupancy_grid):
        # Occupancy values: -1 (unknown), 0 (free) to 100 (occupied)
        # Convert to 0 (black/free) to 255 (white/occupied), with 128 for unknown areas

        # Normalize the occupancy values
        #img_data = np.array(occupancy_grid.data)
        #img_data = ((img_data + 1) / 101) * 255  # Normalize to [0, 255]
        #img_data[img_data == 255] = 128  # Set unknown areas (-1) to 128 (mid gray)

        # Create an Image message
        image_msg = Image()
        image_msg.height = occupancy_grid.info.height
        image_msg.width = occupancy_grid.info.width
        image_msg.encoding = 'mono8'  # Single-channel grayscale
        image_msg.step = image_msg.width  # Number of bytes in a row
        image_msg.data = occupancy_grid.data.tobytes()

        return image_msg

    def create_heatmap(self, data, colormap=cv2.COLORMAP_BONE):
        """
        Create a heatmap from 2D data using OpenCV.

        Args:
            data: 2D NumPy array representing the grid data.
            colormap: OpenCV colormap to apply (default: COLORMAP_BONE).

        Returns:
            heatmap: Heatmap image as a NumPy array.
        """
        # Normalize data to the range 0–255
        normalized_data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)

        # Convert to 8-bit format for colormap
        normalized_data = normalized_data.astype(np.uint8)

        # Apply a colormap
        heatmap = cv2.applyColorMap(normalized_data, colormap)

        return heatmap

def main(args=None):
    rclpy.init(args=args)

    hf_node = HistogramFilterNode()

    rclpy.spin(hf_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    hf_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

