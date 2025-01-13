import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import OccupancyGrid

from rse_map_models.grid_map import GridMap
from rse_common_utils.sensor_utils import lidar_scan, log_odds
from .occupancy_grid_mapping import OccupancyGridMapping

import PyKDL

import cv2
import signal

import numpy as np
import json
import sys

class OccupancyGridMappingNode(Node):
    def __init__(self):
        super().__init__('occupancy_grid_mapping')

        map_x_lim = [-10, 10]
        map_y_lim = [-10, 10]
        resolution = 0.1 # Grid resolution in [m]
        p_prior = 0.5   # Prior occupancy probability

        self.grid_map = GridMap(map_x_lim, map_y_lim, resolution, log_odds(p_prior))
        self.grid_mapping = OccupancyGridMapping()

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
        self.map_update_timer = self.create_timer(map_update_timer_period, self.map_update_callback)

        self.scan = None
        self.odom = None

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

    # Save the map to JSON
    def save_map_to_json(self, filename="grid_map.json"):
        map_data = {
            "X_lim": self.grid_map.X_lim,
            "Y_lim": self.grid_map.Y_lim,
            "resolution": self.grid_map.resolution,
            "log_odds": self.grid_map.l.tolist()
        }
        with open(filename, 'w') as file:
            json.dump(map_data, file, indent=4)
        print(f"Map saved to {filename}")

    def laser_scan_callback(self, msg):
        self.scan = msg

    def odometry_callback(self, msg):
        self.odom = msg

    def map_update_callback(self):

        if self.scan is None or self.odom is None:
            return

        z_t = lidar_scan(self.scan)
        x_t = self.odom_to_x()

        self.grid_mapping.update_grid_with_sensor_reading(self.grid_map, x_t, z_t)

        occ_grid = self.grid_map.to_ros_occupancy_grid()
        gs_image = self.grid_map.to_grayscale_image()

        map_msg = OccupancyGrid()

        map_msg.info.width = self.grid_map.x_len
        map_msg.info.height = self.grid_map.y_len
        map_msg.info.resolution = self.grid_map.resolution
        map_msg.data = occ_grid 
        self.map_publisher.publish(map_msg)
        self.get_logger().info('Publishing map')

        gs_image = self.plot_robot_pose(gs_image, x_t)
        # Enlarge the image if smaller than 640x640
        if gs_image.shape[0] < 640 or gs_image.shape[1] < 640:
            gs_image = cv2.resize(gs_image, (640, 640), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Map Image", gs_image)
        cv2.waitKey(1)  

    def odom_to_x(self):
        return (self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.quat_to_yaw(self.odom.pose.pose.orientation))

    def quat_to_yaw(self, quat):
        rot = PyKDL.Rotation.Quaternion(quat.x, quat.y, quat.z, quat.w)
        return rot.GetRPY()[2]

    def plot_robot_pose(self, grayscale_image, robot_pose):
        x, y, theta = robot_pose

        # Get the grayscale image of the map
        grayscale_image = (grayscale_image * 255).astype(np.uint8)
        grayscale_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)

        # Convert world coordinates to grid indices
        x_grid, y_grid = self.grid_map.discretize(x, y)

        # Ensure the indices are within bounds
        if not self.grid_map.check_pixel(x_grid, y_grid):
            print("Warning: Robot position is outside the map bounds.")
            return

        # Draw the robot position as a circle
        cv2.circle(grayscale_image, (y_grid, x_grid), radius=1, color=(0, 0, 255), thickness=-1)

        # Draw a line to indicate the robot's orientation
        arrow_length = 5  # Length of the orientation arrow in pixels
        end_x = int(x_grid + arrow_length * np.cos(theta))
        end_y = int(y_grid + arrow_length * np.sin(theta))
        cv2.arrowedLine(grayscale_image, (y_grid, x_grid), (end_y, end_x), color=(255, 0, 0), thickness=1)

        grayscale_image = cv2.flip(grayscale_image, -1)  # Flip the image vertically for display

        return grayscale_image

def main(args=None):
    rclpy.init(args=args)

    occupancy_grid_mapping_node = OccupancyGridMappingNode()

    rclpy.spin(occupancy_grid_mapping_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    occupancy_grid_mapping_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()