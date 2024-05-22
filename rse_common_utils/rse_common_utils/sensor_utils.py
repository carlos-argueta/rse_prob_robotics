# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import numpy as np
import PyKDL

from .helper_utils import normalize_angle

# Function to add Gaussian noise to odometry data
def add_noise_to_observation(z, noise_std):
    noise = np.random.normal(0, noise_std, z.shape)
    return z + noise

def odom_to_pose2D(odom):
    x = odom.pose.pose.position.x
    y = odom.pose.pose.position.y
    yaw = get_yaw_from_quaternion(odom.pose.pose.orientation)
    return (x, y, yaw)

def get_normalized_pose2D(initial_pose, current_pose):
    # Check if the initial pose is set
    if initial_pose:
        x, y, yaw = current_pose
        init_x, init_y, init_yaw = initial_pose

        # Adjust position
        x -= init_x
        y -= init_y

        # Adjust orientation
        yaw -= init_yaw
        yaw = normalize_angle(yaw)

        return (x, y, yaw)
    else:
        return (0.0, 0.0, 0.0)  # Default pose if initial pose not set

def rotate_pose2D(pose, degrees):
    # Convert degrees to radians for the rotation matrix
    radians = np.deg2rad(degrees)

    # Rotation matrix for a given degree of rotation
    R = np.array([
        [np.cos(radians), -np.sin(radians)],
        [np.sin(radians),  np.cos(radians)]
    ])

    # Apply the rotation matrix to the position part of the pose
    rotated_position = R.dot(pose[:2])

    # Rotate the orientation by the same amount, ensuring it wraps correctly
    rotated_orientation = (pose[2] + radians) % (2 * np.pi)

    # Return the new pose with the rotated position and orientation
    return (rotated_position[0], rotated_position[1], rotated_orientation)


def get_yaw_from_quaternion(quaternion):
    rot = PyKDL.Rotation.Quaternion(quaternion.x, quaternion.y, quaternion.z, quaternion.w)
    return rot.GetRPY()[2]

class Odom2DDriftSimulator:
    def __init__(self):
        self.error_accumulation = np.array([0.0, 0.0, 0.0])  # Accumulative drift error
        self.last_update = None

    def add_drift(self, odom, current_time):
        if self.last_update is None:
            self.last_update = current_time
            return odom

        time_delta = current_time - self.last_update
        self.last_update = current_time

        # Increase error over time or based on some condition
        drift_rate = np.array([0.001, 0.001, 0.0001])  # Adjust these rates as needed
        self.error_accumulation += drift_rate * time_delta
        print("Error",self.error_accumulation)

        # Optional: Add random walk
        random_walk = np.random.normal(0, [0.001, 0.001, 0.0001], 3)

        # Apply the drift to odometry
        drifted_odom = odom + self.error_accumulation + random_walk

        return drifted_odom