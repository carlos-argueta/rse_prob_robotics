# Gaussian Filters
![image](https://github.com/user-attachments/assets/7dc167fa-f68d-493b-8b64-ad75d00c681d)

This package contains companion code for my articles covering the Gaussian Filters which include the Kalman family of filters and the Information Filter. 
The articles and code roughly cover the first three chapters of the book (Introduction, Recursive State Estimation, Gaussian Filters) as well as parts of chapter 5 (Robot Motion) and 6 (Robot Perception).


## Installation Instructions

To install the necessary dependencies and clone/build the package, follow these steps:

```bash
# Install some dependency
sudo apt install python3-pykdl

# Clone and build the package
# Change the ROS 2 workspace accordingly
cd ros2_ws/src
git clone https://github.com/carlos-argueta/rse_prob_robotics.git
cd ..
colcon build --symlink-install
```

## The Kalman Filter
### Article:
[Recursive State Estimation with Kalman Filters and ROS 2](https://medium.com/@kidargueta/recursive-state-estimation-with-kalman-filters-and-ros-2-b869d3775357?source=friends_link&sk=7bfc399c3dc05e1e2143933f8c98046d
)

### Run the code:
To run the project, you'll need to open three terminals. Follow the steps below:
#### Terminal 1
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch rse_gaussian_filters rviz_launch.launch.py
```

#### Terminal 2
```bash
source ~/ros2_ws/install/setup.bash
ros2 run rse_gaussian_filters kf_estimation
```

#### Terminal 3
First download the ROS 2 bag with all of the data from [this link](https://www.dropbox.com/scl/fi/tdxin6bzw01siucdv3kgv/linkou-2023-12-27-2-med.zip?rlkey=rcz93bhozjsdymcpn5dqz6rly&dl=0).
Make sure to decompress the file before using it.
```bash
# Navigate to where you extracted the ROS 2 bag and then run it with:
ros2 bag play linkou-2023-12-27-2-med --clock

```
### Demo Video
<div align="center">
  <a href="https://youtu.be/TPbO3kBygb4" target="_blank">
    <img src="https://img.youtube.com/vi/TPbO3kBygb4/0.jpg" alt="Watch the video" style="width:80%;height:auto;">
  </a>
</div>

## The Extended Kalman Filter

### Article
[Sensor Fusion with the Extended Kalman Filter in ROS 2](https://medium.com/@kidargueta/sensor-fusion-with-the-extended-kalman-filter-in-ros-2-d33dbab1829d?source=friends_link&sk=c0298555efc873e7bfecb20960f1791d
)
### Run the code:
To run the project, you'll need to open three terminals. Follow the steps below:
#### Terminal 1
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch rse_gaussian_filters rviz_launch.launch.py
```

#### Terminal 2
Run one of the following commands depending on the version of the Extended Kalman Filter you want to try. There won’t be any output at first, until you play the ROS 2 bag.
```bash
source ~/ros2_ws/install/setup.bash

# Run only one of the lines below

# 3D state, basic velocity model
ros2 run rse_gaussian_filters ekf_estimation_3d_v1 

# 3D state, advanced velocity model
ros2 run rse_gaussian_filters ekf_estimation_3d_v2 

# 7D state, acceleration model, sensor fusion
ros2 run rse_gaussian_filters ekf_estimation_7d 

# 8D state, acceleration model, sensor fusion
ros2 run rse_gaussian_filters ekf_estimation_8d 
```

#### Terminal 3
First download the ROS 2 bag with all of the data from [this link](https://www.dropbox.com/scl/fi/tdxin6bzw01siucdv3kgv/linkou-2023-12-27-2-med.zip?rlkey=rcz93bhozjsdymcpn5dqz6rly&dl=0).
Make sure to decompress the file before using it.
```bash
# Navigate to where you extracted the ROS 2 bag and then run it with:
ros2 bag play linkou-2023-12-27-2-med --clock

```
### Demo Video
<div align="center">
  <a href="https://youtu.be/9p2swpHGr2w" target="_blank">
    <img src="https://img.youtube.com/vi/9p2swpHGr2w/0.jpg" alt="Watch the video" style="width:80%;height:auto;">
  </a>
</div>

## The Unscented Kalman Filter
Coming Soon!

## The Information Filter
Coming Soon!
