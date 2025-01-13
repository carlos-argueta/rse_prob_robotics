# Occupancy Grid Mapping
![image](https://github.com/user-attachments/assets/312f474a-bbd2-4753-8a11-0bb54f3bc350)

This package contains a companion code for the article Occupancy Grid Mapping with The Binary Bayes Filter in ROS 2 which introduces the basic occupancy grid mapping algorithm introduced in section 9.2 of the Probabilistic Robotics book.


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
