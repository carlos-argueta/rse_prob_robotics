# Nonparametric Filters
<img width="1024" height="848" alt="cover" src="https://github.com/user-attachments/assets/780145b9-da3b-4602-bd8a-54f38e8210e1" />

This package contains companion code for my articles covering the Nonparametric Filters, which include the Particle Filter and the Histogram Filter. 
The articles and code build on the first three chapters of the book (Introduction, Recursive State Estimation, Gaussian Filters), and introduce the Nonparametric filters described in chapter 4. It also covers parts of chapter 5 (Robot Motion) and 6 (Robot Perception).


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

## The Particle Filter
### Article:
[Robot State Estimation with the Particle Filter in ROS 2 - Part 1](https://soulhackerslabs.com/robot-state-estimation-with-the-particle-filter-in-ros-2-part-1-e19bd286dfd6?source=friends_link&sk=f86b52432aec6d8864da46a9d7461f9c
)

### Run the code:
To run the project, you'll need to open three terminals. Follow the steps below:
#### Terminal 1
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch rse_gaussian_filters rviz_launch.launch.py
```

#### Terminal 2
Run one of the following commands depending on the version of the Unscented Kalman Filter you want to try. There won’t be any output at first, until you play the ROS 2 bag.
```bash
source ~/ros2_ws/install/setup.bash

# Run only one of the lines below

# 3D state, velocity model
ros2 run rse_parametric_filters pf_estimation_3d

# 8D state, acceleration model, sensor fusion
ros2 run rse_parametric_filters pf_estimation_8d 
```

#### Terminal 3
First, download the ROS 2 bag with all of the data from [this link](https://www.dropbox.com/scl/fi/tdxin6bzw01siucdv3kgv/linkou-2023-12-27-2-med.zip?rlkey=rcz93bhozjsdymcpn5dqz6rly&dl=0).
Make sure to decompress the file before using it.
```bash
# Navigate to where you extracted the ROS 2 bag and then run it with:
ros2 bag play linkou-2023-12-27-2-med --clock

```
### Demo Video
<div align="center">
  <a href="https://youtu.be/46eQsJqjOXk" target="_blank">
    <img src="https://img.youtube.com/vi/46eQsJqjOXk/0.jpg" alt="Watch the video" style="width:80%;height:auto;">
  </a>
</div>

## The Histogram Filter

### Coming soon!
