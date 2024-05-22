import rclpy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from .conversions import read_points

import numpy as np
import open3d as o3d

from scipy.spatial.transform import Rotation

from rse_common_utils.visualization import Visualizer
from rse_common_utils.sensor_utils import odom_to_pose2D, get_normalized_pose2D, rotate_pose2D

class LaserScanOdometry:
    def __init__(self):
        self.node = rclpy.create_node('laser_scan_odometry_node')

        self.publisher_ = self.node.create_publisher(Odometry, 'pointcloud/odom', 10)

        self.subscription = self.node.create_subscription(
            LaserScan,
            '/scan_filtered',
            self.listener_callback,
            10)

        self.odom_gt_subscription = self.node.create_subscription(
            Odometry,
            'odom', #'wheel_odom',  # Ground Truth
            self.odom_gt_callback,
            10)


        self.ls_odom_msg = Odometry()

        self.prev_cloud = None
        
        # Initialize an identity matrix as the initial odometry
        self.odometry = np.eye(4)

        self.transformation = np.eye(4)

        # For testing purposes
        self.visualizer = Visualizer()
        self.initial_gt_pose = None
        self.normalized_gt_pose = (0.0, 0.0, 0.0)

        
    def listener_callback(self, msg):

        cloud = self.scan_to_pointcloud(msg)
        
        if self.prev_cloud is None:
            self.prev_cloud = cloud
              
            return
        
        self.transformation , inlier_rmse = self.perform_icp_ransac(self.prev_cloud, cloud)
        #self.perform_icp_point_to_plane(self.prev_cloud, cloud)

        # Multiply the current odometry with the new transformation matrix to update the odometry
        self.odometry = np.dot(self.odometry, self.transformation)

        # Extract rotation matrix
        R = self.odometry[0:3, 0:3]
        euler = self.rotation_to_euler(R)

        # Extract translation vector
        T = self.odometry[0:3, 3]

        self.node.get_logger().info('LiDAR Odom: x: %f, y: %f, yaw: %f' % (T[0], T[1], euler[2]))

        self.publish_odometry(T[0], T[1], T[2], R, euler[2])

        self.prev_cloud = cloud

        self.visualizer.update(self.normalized_gt_pose, (-T[0], T[1], euler[2]))

    def odom_gt_callback(self, msg):
        # Set the initial pose
        if not self.initial_gt_pose:

            initial_pose = odom_to_pose2D(msg)  

            self.initial_gt_pose = initial_pose # rotate_pose2D(initial_pose, 70)

        # Get and normalize the pose
        current_pose = odom_to_pose2D(msg)  

        # rotated_pose = rotate_pose2D(current_pose, 70)

        self.normalized_gt_pose = np.array(get_normalized_pose2D(self.initial_gt_pose, current_pose))

    def publish_odometry(self, x, y, z, R, yaw):
            quat_x, quat_y, quat_z, quat_w = self.rotation_to_quaternion(np.transpose(R))

            self.ls_odom_msg = Odometry()
            self.ls_odom_msg.header.stamp = self.node.get_clock().now().to_msg()
            self.ls_odom_msg.header.frame_id = "map"
            self.ls_odom_msg.child_frame_id = ""
            self.ls_odom_msg.pose.pose.position.x = -x
            self.ls_odom_msg.pose.pose.position.y = y
            self.ls_odom_msg.pose.pose.position.z = z
            self.ls_odom_msg.pose.pose.orientation.x = quat_x
            self.ls_odom_msg.pose.pose.orientation.y = quat_y
            self.ls_odom_msg.pose.pose.orientation.z = quat_z
            self.ls_odom_msg.pose.pose.orientation.w = quat_w

            self.publisher_.publish(self.ls_odom_msg)
        
    def scan_to_pointcloud(self, msg):
        """ Converts a ROS LaserScan message to an Open3D PointCloud object. """
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        points = np.array([np.cos(angles)*msg.ranges, np.sin(angles)*msg.ranges, np.zeros(len(msg.ranges))]).T
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        return cloud

    def perform_icp_ransac(self, source, target):
        threshold = 1.0  # Distance threshold for RANSAC
        trans_init = self.transformation  # Initial transformation

        # Perform RANSAC followed by ICP refinement
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

        return reg_p2p.transformation, reg_p2p.inlier_rmse
    
    def perform_icp_point_to_plane(self, source, target):

        # Estimate normals
        o3d.geometry.PointCloud.estimate_normals(
            source,
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))

        o3d.geometry.PointCloud.estimate_normals(
            target,
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))

        threshold = 1.0  # Distance threshold for RANSAC
        trans_init = self.transformation  # Initial transformation

        # Perform point-to-plane ICP
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())

        return reg_p2p.transformation, reg_p2p.inlier_rmse

    def remove_outliers(self, point_cloud):
        cloud, ind = point_cloud.remove_statistical_outlier(nb_neighbors=20,
                                                        std_ratio=2.0)
        return cloud

    
    def rotation_to_euler(self, R):
        """ Converts a rotation matrix to Euler angles. """

        r = Rotation.from_matrix(R)
        euler = r.as_euler('xyz', degrees=True)
        return euler   
    
    def rotation_to_quaternion(self, R):
        """ Converts a rotation matrix to Euler angles. """

        r = Rotation.from_matrix(R)
        # Convert to quaternion
        quaternion = r.as_quat()
        return quaternion
    
def main(args=None):
    rclpy.init(args=args)

    laser_scan_odometry_node = LaserScanOdometry()

    rclpy.spin(laser_scan_odometry_node.node)

    laser_scan_odometry_node.node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
