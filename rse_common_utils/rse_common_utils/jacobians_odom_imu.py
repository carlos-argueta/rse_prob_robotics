import unittest
import numpy as np
import sympy as sp

import matplotlib.pyplot as plt

from rse_motion_models.motion_models import acceleration_motion_model_linearized_2
from rse_observation_models.observation_models import odometry_imu_observation_model_with_acceleration_motion_model_linearized

# Initialize pretty printing
sp.init_printing(use_latex=True)

# Function to plot a matrix
def plot_matrix(matrix, title="Matrix"):
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.set_title(title)
    sp.preview(matrix, viewer='file', filename='output.png', euler=False, dvioptions=['-D', '1200'])
    img = plt.imread('output.png')
    plt.imshow(img)
    plt.show()

class TestJacobians(unittest.TestCase):
    
    def setUp(self):
        self.G_sp, self.V_sp, self.H_sp = self.create_symbolic_jacobians_acc_model()
        _, self.G_np, self.V_np = acceleration_motion_model_linearized_2() 
        _, self.H_np = odometry_imu_observation_model_with_acceleration_motion_model_linearized()

    # Define the symbolic Jacobian
    def create_symbolic_jacobians(self):
        # Define symbols
        x, y, theta, v, w, delta_t, prev_theta, v_x, v_y, a_x, a_y = sp.symbols('x y theta v omega Delta_t theta_t-1 v_x v_y a_x a_y')

        # Define matrix g
        g = sp.Matrix([
                   x -v/w * sp.sin(theta) + v/w * sp.sin(theta + w * delta_t),
                   y + v/w * sp.cos(theta) - v/w * sp.cos(theta + w * delta_t),
                   theta + w * delta_t
                ])

        # Define matrix h
        h = sp.Matrix([x, y, theta, theta, (theta - prev_theta) / delta_t])

        # Define state vector
        state = sp.Matrix([x, y, theta])

        control = sp.Matrix([v, w])

        # Compute Jacobians
        G = g.jacobian(state)
        V = g.jacobian(control)
        V = sp.simplify(V)
        H = h.jacobian(state)
        print("x = ")
        sp.pprint(state)
        print("\nu= ")
        sp.pprint(control)
        print("\nG = ")
        sp.pprint(G)
        print("\nV = ")
        sp.pprint(V)
        print("\nH = ")
        sp.pprint(H)

        # Plot the state vector and Jacobian
        # plot_matrix(state, "State Vector")
        # plot_matrix(G, "Jacobian Matrix of g wrt state")
        # plot_matrix(V, "Jacobian Matrix of g wrt control")
        
        return sp.lambdify((theta, w, delta_t, v), G, 'numpy'), sp.lambdify((theta, w, delta_t, v), V, 'numpy'), sp.lambdify((x, y, theta, prev_theta, delta_t), H, 'numpy')

    # Define the symbolic Jacobian
    def create_symbolic_jacobians_acc_model(self):
        # Define symbols
        # x, y, theta, v, w, delta_t, prev_theta, a_x, a_y = sp.symbols('x y theta v omega Delta_t theta_t-1 a_x a_y')
        x, y, theta, v_x, v_y, v, w, delta_t, prev_theta, a_x, a_y = sp.symbols('x y theta v_x, v_y, v, omega Delta_t theta_t-1 a_x a_y')

        # Define the motion model matrix 'g' including acceleration
        '''g = sp.Matrix([
            x + v_x * delta_t + 0.5 * a_x * delta_t**2,  # Update for x position
            y + v_y * delta_t + 0.5 * a_y * delta_t**2,  # Update for y position
            theta + w * delta_t,                         # Update for orientation
            v_x + a_x * delta_t,                         # Update for velocity in x
            v_y + a_y * delta_t,                         # Update for velocity in y
            w,                                           # Assuming constant angular velocity
            a_x,                                         # Assuming constant linear acceleration in x
            a_y                                          # Assuming constant linear acceleration in y
        ])'''

        '''g = sp.Matrix([
            x + v * sp.cos(theta) * delta_t + 0.5 * a_x * delta_t**2,           # Update for x position with acceleration
            y + v * sp.sin(theta) * delta_t + 0.5 * a_y * delta_t**2,           # Update for y position with acceleration
            theta + w * delta_t,                                                # Update for orientation
            v + a_x * sp.cos(theta) * delta_t + a_y * sp.sin(theta) * delta_t,  # Update for velocity in direction of theta
            w,                                                                  # Assuming constant angular velocity
            a_x,                                                                # Assuming constant linear acceleration in x
            a_y                                                                 # Assuming constant linear acceleration in y
        ])'''

        # Updated motion model matrix 'g' including linear acceleration and direct velocity component updates
        g = sp.Matrix([
            x + v * sp.cos(theta) * delta_t + 0.5 * a_x * delta_t**2,  # Update for x position with acceleration and forward velocity
            y + v * sp.sin(theta) * delta_t + 0.5 * a_y * delta_t**2,  # Update for y position with acceleration and forward velocity
            theta + w * delta_t,                         # Update for orientation
            v * sp.cos(theta) + a_x * delta_t,                         # Update for velocity in x
            v * sp.sin(theta) + a_y * delta_t,                         # Update for velocity in y
            w,                                           # Assuming constant angular velocity
            a_x,                                             # Assuming constant linear acceleration in x
            a_y                                              # Assuming constant linear acceleration in y
        ])


        # Define matrix h
        h = sp.Matrix([x, y, theta, theta, w, a_x, a_y])

        # Define state vector
        state = sp.Matrix([x, y, theta, v_x, v_y, w, a_x, a_y])

        control = sp.Matrix([v, w])

        # Compute Jacobians
        G = g.jacobian(state)
        V = g.jacobian(control)
        V = sp.simplify(V)
        H = h.jacobian(state)
        print("x = ")
        sp.pprint(state)
        print("\nu= ")
        sp.pprint(control)
        print("\ng = ")
        sp.pprint(g)
        print("\nG = ")
        sp.pprint(G)
        print("\nV = ")
        sp.pprint(V)
        print("\nh = ")
        sp.pprint(h)
        print("\nH = ")
        sp.pprint(H)

        # Plot the state vector and Jacobian
        # plot_matrix(state, "State Vector")
        # plot_matrix(G, "Jacobian Matrix of g wrt state")
        # plot_matrix(V, "Jacobian Matrix of g wrt control")
        
        return sp.lambdify((theta, v, a_x, a_y, delta_t), G, 'numpy'), sp.lambdify((theta, delta_t), V, 'numpy'), sp.lambdify((delta_t), H, 'numpy')


    def test_jacobian_G(self):
        print("Checking jacobian G")
        test_cases = [
            (0, 1.5, 0.5, 0.5, 0.1),
            (np.pi/4, 2.0, 0.1, 0.1, 0.1),
            (np.pi/2, 0.5, 0.05, 0.05, 0.1),
            (0, 0.01, 0.5, 0.5, 0.1)  # Edge case where w is very small
        ]

        # theta, v, a_x, a_y, delta_t
        for theta, v, a_x, a_y, delta_t in test_cases:
            with self.subTest(theta=theta, v=v, a_x=a_x, a_y=a_y, delta_t=delta_t, ):
                result_G_sp = self.G_sp(theta, v, a_x, a_y, delta_t)
                result_G_np = self.G_np([0.0, 0.0, theta, 0.0, 0.0, a_x, a_y], [v, 0.0], delta_t)

                np.testing.assert_allclose(result_G_sp, result_G_np, atol=1e-5)

                
    def test_jacobian_V(self):
        print("Checking jacobian V")
        test_cases = [
            (0, 1, 0.1, 2),
            (np.pi/4, 2, 0.1, 1),
            (np.pi/2, 0.5, 0.05, 3),
            (0, 0.01, 0.5, 4)  # Edge case where w is very small
        ]
        for theta, w, delta_t, v in test_cases:
            with self.subTest(theta=theta, w=w, delta_t=delta_t, v=v):
                
                result_V_sp = self.V_sp(theta, delta_t)
                result_V_np = self.V_np(mu = [0.0, 0.0, theta], delta_t = delta_t)

                np.testing.assert_allclose(result_V_sp, result_V_np, atol=1e-5)

    def test_jacobian_H(self):
        print("Checking jacobian H")
        test_cases = [
            (0.0, 1.5, 0.1, 0.09, 0.01),
            (10.7, 20.3, np.pi/4, np.pi/4 + 0.01, 0.01),
            (50.6, 25.1, np.pi/2, np.pi/2 - 0.02, 0.01),
            (0.0, 0.0, 0.0, 0.0, 0.01)  # Edge case where w is very small
        ]
        for x, y, theta, prev_theta, delta_t in test_cases:
            with self.subTest(x = x, y = y, theta=theta, prev_theta=prev_theta, delta_t=delta_t):
                
                result_H_sp = self.H_sp(delta_t)
                result_H_np = self.H_np() 

                np.testing.assert_allclose(result_H_sp, result_H_np, atol=1e-5)
    

if __name__ == "__main__":
    unittest.main()
