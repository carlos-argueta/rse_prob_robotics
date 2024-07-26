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

def odom_to_pose3D(odom):
    x = odom.pose.pose.position.x
    y = odom.pose.pose.position.y
    z = odom.pose.pose.position.z
    rpy = get_rpy_from_quaternion(odom.pose.pose.orientation)
    return (x, y, z, rpy[0], rpy[1], rpy[2])

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
    
def get_normalized_pose3D(initial_pose, current_pose):
    # Check if the initial pose is set
    if initial_pose:
        x, y, z, roll, pitch, yaw = current_pose
        init_x, init_y, init_z, init_roll, init_pitch, init_yaw = initial_pose

        # Adjust position
        x -= init_x
        y -= init_y
        z -= init_z

        # Adjust orientation
        roll -= init_roll
        pitch -= init_pitch
        yaw -= init_yaw
        
        # Normalize angles
        roll = normalize_angle(roll)
        pitch = normalize_angle(pitch)
        yaw = normalize_angle(yaw)

        return (x, y, z, roll, pitch, yaw)
    else:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # Default pose if initial pose not set

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

def rotate_pose3D(pose, roll_angle, pitch_angle, yaw_angle):
    # Extract position and orientation from the pose
    x, y, z, roll, pitch, yaw = pose

    # Convert angles to radians
    roll_rad = np.deg2rad(roll_angle)
    pitch_rad = np.deg2rad(pitch_angle)
    yaw_rad = np.deg2rad(yaw_angle)

    # Rotation matrices for roll, pitch, and yaw
    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll_rad), -np.sin(roll_rad)],
        [0, np.sin(roll_rad), np.cos(roll_rad)]
    ])
    
    R_pitch = np.array([
        [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
        [0, 1, 0],
        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])
    
    R_yaw = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])

    # Combined rotation matrix
    R_combined = R_yaw @ R_pitch @ R_roll

    # Apply the rotation to the position part of the pose
    rotated_position = R_combined.dot(np.array([x, y, z]))

    # Rotate the orientation by the same amount, ensuring it wraps correctly
    rotated_roll = (roll + roll_rad) % (2 * np.pi)
    rotated_pitch = (pitch + pitch_rad) % (2 * np.pi)
    rotated_yaw = (yaw + yaw_rad) % (2 * np.pi)

    # Return the new pose with the rotated position and orientation
    return (rotated_position[0], rotated_position[1], rotated_position[2], rotated_roll, rotated_pitch, rotated_yaw)


def get_yaw_from_quaternion(quaternion):
    rot = PyKDL.Rotation.Quaternion(quaternion.x, quaternion.y, quaternion.z, quaternion.w)
    return rot.GetRPY()[2]

def get_rpy_from_quaternion(quaternion):
    rot = PyKDL.Rotation.Quaternion(quaternion.x, quaternion.y, quaternion.z, quaternion.w)
    return rot.GetRPY()

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