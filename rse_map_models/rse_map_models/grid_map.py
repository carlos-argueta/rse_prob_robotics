import numpy as np 

P_0 = 0.5
P_OCC = 0.9
P_FREE = 0.3

def log_odds(p):
	"""
	Log odds ratio of p(x):

		       p(x)
	 l(x) = log ----------
		     1 - p(x)

	"""
	return np.log(p / (1 - p))


def retrieve_p(l):
	"""
	Retrieve p(x) from log odds ratio:

	 		   1
	 p(x) = 1 - ---------------
		     1 + exp(l(x))

	"""
	return 1 - 1 / (1 + np.exp(l))

class GridMap:
	"""
	Grid map
	"""
	def __init__(self, X_lim, Y_lim, resolution, initial_p, normalize=False):
		
		self.resolution = resolution
		self.min_x = X_lim[0]
		self.min_y = Y_lim[0]
		self.max_x = X_lim[1]
		self.max_y = X_lim[1]
		# self.width = int(round((self.max_x - self.min_x) / self.resolution))
		# self.height = int(round((self.max_y - self.min_y) / self.resolution))
		#self.width = len(np.arange(start = X_lim[0], stop = X_lim[1] + resolution, step = resolution))
		#self.height = len(np.arange(start = Y_lim[0], stop = Y_lim[1] + resolution, step = resolution))

		# probability matrix in log-odds scale:
		#self.data = np.full(shape=(self.width, self.height), fill_value=initial_p)

		self.height = len(np.arange(start=X_lim[0], stop=X_lim[1] + resolution, step=resolution))
		self.width = len(np.arange(start=Y_lim[0], stop=Y_lim[1] + resolution, step=resolution))

		self.data = np.full(shape=(self.height, self.width), fill_value=initial_p)


		if normalize:
			self.normalize_probability()

	def normalize_probability(self):
		sump = sum([sum(i_data) for i_data in self.data])

		for ix in range(self.width):
			for iy in range(self.height):
				self.data[ix][iy] /= sump

	def get_shape(self):
		"""
		Get dimensions
		"""
		return np.shape(self.data)

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


	def discretize(self, x_cont, y_cont):
		"""
		Discretize continious x and y 
		"""
		x = int((x_cont - self.min_x) / self.resolution)
		y = int((y_cont - self.min_y) / self.resolution)
		return (x,y)

	def get_cell_center_mass(self, x_grid, y_grid):
		"""
		Get the center of mass for a cell given its grid coordinates.
		"""
		x_cont = (x_grid * self.resolution) + self.min_x + (self.resolution / 2)
		y_cont = (y_grid * self.resolution) + self.min_y + (self.resolution / 2)
		return (x_cont, y_cont)

	def check_pixel(self, x, y):
		"""
		Check if pixel (x,y) is within the map bounds
		"""
		if x >= 0 and x < self.get_shape()[0] and y >= 0 and y < self.get_shape()[1]:
			
			return True 

		else:

			return False
		
	def get_occupied_cells(self):
		"""
		Get the occupied cells in the map
		"""
		occupied_cells = []
		for x in range(self.width):
			for y in range(self.height):
				if retrieve_p(self.data[x, y]) > P_0:
					occupied_cells.append((x, y))
		return occupied_cells
	
	def get_most_likely_cell(self):
		"""
		Finds the most likely cell in a grid based on the probabilities.

		Args:
			grid: 2D numpy array representing the grid probabilities.

		Returns:
			(row, col): Tuple of the grid coordinates of the most likely cell.
		"""
		# Find the flat index of the maximum probability
		max_index = np.argmax(self.data)
		max_ = np.max(self.data)
		print("max_index", max_index, max_)

		# Convert the flat index to 2D coordinates
		row, col = np.unravel_index(max_index, self.data.shape)

		return row, col

