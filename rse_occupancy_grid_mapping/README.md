# Occupancy Grid Mapping
![image](https://github.com/user-attachments/assets/312f474a-bbd2-4753-8a11-0bb54f3bc350)

This package contains a companion code for the article Occupancy Grid Mapping with The Binary Bayes Filter in ROS 2 which introduces the basic occupancy grid mapping algorithm introduced in section 9.2 of the Probabilistic Robotics book.

## Article:
[Occupancy Grid Mapping with The Binary Bayes Filter in ROS 2](https://medium.com/@kidargueta/recursive-state-estimation-with-kalman-filters-and-ros-2-b869d3775357?source=friends_link&sk=7bfc399c3dc05e1e2143933f8c98046d
)

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
You have two options to run the mapping node. Option one is to use the pre-recorded data, option two is to run a simulation of the Turtlebot 3 yourself. Depending of your choice, follow one of the set of instructions below.

### Option 1: Downloading ROS2 bag

#### Download the ROS2 bag
```bash
wget -O occupancy_grid_mapping_tb3.zip "https://www.dropbox.com/scl/fi/anc0jq4vwhpe78q0ajg4v/occupancy_grid_mapping_tb3.zip?rlkey=bdzpd5uowonfruutk8u4iyzkt&st=9xjq6dr9&dl=1"
```

#### Unzip the bag
```bash
unzip occupancy_grid_mapping_tb3.zip
```

### Option 2: Simulation installation

#### Install Gazebo Classic ROS2 tools 
```bash
sudo apt install ros-humble-gazebo-ros-pkgs 
```

#### Install Turtlebot 3 tools
```bash
sudo apt install ros-humble-turtlebot3-msgs  

sudo apt install ros-humble-turtlebot3 
```

#### Install Turtlebot 3 simulation packages 
```bash
mkdir -p ~/turtlebot3_ws/src

cd ~/turtlebot3_ws/src/ 
 
git clone -b humble-devel https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git 

cd ~/turtlebot3_ws && colcon build --symlink-install
```

## Run the code:
### Option 1: With ROS2 bag
To run the project, you'll need to open two terminals. Follow the steps below:

#### Terminal 1
```bash
# Launch mapping
source ~/ros2_ws/install/setup.bash 
ros2 run rse_occupancy_grid_mapping occupancy_grid_mapping
```

#### Terminal 2
```bash
# Navigate to where you extracted the ROS 2 bag and then run it with:
ros2 bag play rosbag2_2025_01_13-13_30_54 --clock
```

### Option 2: With simulation:
To run the project, you'll need to open three terminals. Follow the steps below:

#### Terminal 1
```bash
# Launch mapping
source ~/ros2_ws/install/setup.bash 
ros2 run rse_occupancy_grid_mapping occupancy_grid_mapping
```
#### Terminal 2
```bash
# Launch simulation
export TURTLEBOT3_MODEL=waffle
source ~/turtlebot3_ws/install/setup.bash 
ros2 launch turtlebot3_gazebo turtlebot3_house.launch.py 
```
It may take a while to load the simulation the first time you run this. Rerun the launch command if it gets stuck for too long or fails.

#### Terminal 3
```bash
# Launch teleop
export TURTLEBOT3_MODEL=waffle
source ~/turtlebot3_ws/install/setup.bash 
ros2 run turtlebot3_teleop teleop_keyboard
```
Use the teleop tool to remote control the Turtlebot 3 and build the map.


## Demo Video
<div align="center">
  <a href="https://youtu.be/-1XAxXY9lo0" target="_blank"> 
    <img src="https://img.youtube.com/vi/-1XAxXY9lo0/0.jpg" alt="Watch the video" style="width:80%;height:auto;">
  </a>
</div>
