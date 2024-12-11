import sys
sys.path.append('/home/carlos/pr_ws/src/rse_prob_robotics/rse_map_models/rse_map_models')
from grid_map import GridMap

import json
import numpy as np

import cv2

# Load the map from JSON
def load_map_from_json(filename="grid_map.json"):
    with open(filename, 'r') as file:
        map_data = json.load(file)
    grid_map = GridMap(
        X_lim=map_data["X_lim"],
        Y_lim=map_data["Y_lim"],
        resolution=map_data["resolution"],
        log_odds_p=np.array(map_data["log_odds"])  # Convert list back to numpy array
    )
    print(f"Map loaded from {filename}")
    return grid_map


grid_map = load_map_from_json("/home/carlos/pr_ws/src/rse_prob_robotics/maps/grid_map.json")

gs_image = grid_map.to_grayscale_image()

cv2.imshow("Grid Map", gs_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
