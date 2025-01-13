import numpy as np

from rse_common_utils.sensor_utils import log_odds, normalize_angle


class InverseSensorModels:

	P_0 = 0.5
	P_OCC = 0.9
	P_FREE = 0.3

	ALPHA = 0.1

	def inverse_range_sensor_model(self, m_i, x_t, z_t):

		x_i , y_i = m_i # the cell represented by its center of mass
		x, y, theta = x_t # the robot pose
		z_t, theta_sens, _, z_max, beta = z_t # the measurement represented by a list of ranges, angles, max range, width of sensor beam

		r = np.hypot(x_i - x, y_i - y)
		phi = normalize_angle(np.arctan2(y_i - y, x_i - x) - theta)
		k = np.argmin(np.abs(phi - theta_sens))

		if r > min(z_max, z_t[k] + self.ALPHA / 2.0) or np.abs(normalize_angle(phi - theta_sens[k])) > beta / 2.0:
			return log_odds(self.P_0)
		if z_t[k] < z_max and np.abs(r - z_t[k]) <= self.ALPHA / 2.0:
			return log_odds(self.P_OCC)
		if r <= z_t[k]:
			return log_odds(self.P_FREE)
		
	def inverse_range_sensor_model_get_x_y(self, m_i, x_t, z_t):

		x_i , y_i = m_i # the cell represented by its center of mass
		x, y, theta = x_t # the robot pose
		z_t, theta_sens, _, z_max, beta = z_t # the measurement represented by a list of ranges, angles, max range, width of sensor beam

		r = np.hypot(x_i - x, y_i - y)
		phi = normalize_angle(np.arctan2(y_i - y, x_i - x) - theta)
		k = np.argmin(np.abs(phi - theta_sens))

		if z_t[k] < z_max and np.abs(r - z_t[k]) <= self.ALPHA / 2.0:
			return x_i, y_i
		else:
			return None, None

















