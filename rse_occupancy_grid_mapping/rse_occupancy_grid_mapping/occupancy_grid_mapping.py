from rse_common_utils.sensor_utils import is_in_perceptual_field
from rse_sensor_models.inverse_sensor_models import InverseSensorModels

class OccupancyGridMapping:

	def __init__(self):
		self.sensor_model = InverseSensorModels()

	def update_grid_with_sensor_reading(self, grid_map, x_t, z_t):
		x, y, theta = x_t # the robot pose
		_, _, _, sensor_range_max, sensor_angle_range = z_t # the sensor reading

		# Loop over all cells in the grid
		for x_grid in range(grid_map.data.shape[0]):
			for y_grid in range(grid_map.data.shape[1]):
				# Get the center of mass of the current cell
				m_i = grid_map.get_cell_center_mass(x_grid, y_grid)
				
				if is_in_perceptual_field(m_i, (x, y), theta, sensor_range_max, sensor_angle_range):
					# Apply the inverse sensor model
					log_odds_update =  self.sensor_model.inverse_range_sensor_model(m_i, x_t, z_t) 

					# Update the cell in the grid with the new log-odds value
					grid_map.data[x_grid, y_grid] += log_odds_update


