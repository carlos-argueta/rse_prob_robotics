import numpy as np

P_0 = 0.5
P_OCC = 0.9
P_FREE = 0.3


def log_odds(p):
	return np.log(p / (1 - p))


def retrieve_p(l):
	return 1 - 1 / (1 + np.exp(l))


class GridMap:
	def __init__(self, X_lim, Y_lim, resolution, initial_p, normalize=False):
		self.resolution = resolution
		self.min_x = X_lim[0]
		self.min_y = Y_lim[0]
		self.max_x = X_lim[1]
		self.max_y = Y_lim[1]

		self.x_len = len(np.arange(start=self.min_x, stop=self.max_x + resolution, step=resolution))
		self.y_len = len(np.arange(start=self.min_y, stop=self.max_y + resolution, step=resolution))

		self.data = np.full(shape=(self.x_len, self.y_len), fill_value=initial_p)

		if normalize:
			self.normalize_probability()

	def normalize_probability(self):
		sump = np.sum(self.data)
		self.data /= sump

	def discretize(self, x_cont, y_cont):
		"""
		Convert world coordinates to grid coordinates, aligning with ROS 2's system.
		"""
		x_index = int((x_cont - self.min_x) / self.resolution) 
		y_index = int((y_cont - self.min_y) / self.resolution)
		return x_index, y_index

	def get_cell_center_mass(self, x_grid, y_grid):
		"""
		Get the center of mass for a cell given its grid coordinates.
		"""
		x_cont = (x_grid * self.resolution) + self.min_x + (self.resolution / 2)
		y_cont = (y_grid * self.resolution) + self.min_y + (self.resolution / 2)
		return x_cont, y_cont

	def check_pixel(self, x, y):
		"""
		Check if pixel (x,y) is within the map bounds.
		"""
		return 0 <= x < self.x_len and 0 <= y < self.y_len

	def get_occupied_cells(self):
		"""
		Get the occupied cells in the map.
		"""
		occupied_cells = []
		for x in range(self.x_len):
			for y in range(self.y_len):
				if retrieve_p(self.data[y, x]) > P_0:  # Access in (row, col) format
					occupied_cells.append((x, y))
		return occupied_cells
	
	def to_ros_occupancy_grid(self):
		map_array_2d = self.to_grayscale_image() * 100

		map_array_2d = map_array_2d.astype(int)

		# Flatten the 2D array to a 1D array
		map_array_1d = map_array_2d.flatten()

		# Convert to a list
		map_data_list = map_array_1d.tolist()

		return map_data_list

	def to_BGR_image(self):
		"""
		Transformation to BGR image format
		"""
		# grayscale image
		gray_image = 1 - retrieve_p(self.data)

		# repeat values of grayscale image among 3 axis to get BGR image
		rgb_image = np.repeat(a = gray_image[:,:,np.newaxis], 
							  repeats = 3,
							  axis = 2)

		return rgb_image

	def to_grayscale_image(self):
		"""
		Transformation to GRAYSCALE image format
		"""
		return 1 - retrieve_p(self.data)
