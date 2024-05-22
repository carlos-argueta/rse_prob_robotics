import unittest
import numpy as np
import sympy as sp

import matplotlib.pyplot as plt

from rse_motion_models.velocity_motion_models import velocity_motion_model_linearized_1
from rse_observation_models.odometry_observation_models import odometry_observation_model_linearized

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
        self.G_sp, self.V_sp, self.H_sp = self.create_symbolic_jacobians()
        _, self.G_np, self.V_np = velocity_motion_model_linearized_1() 
        _, self.H_np = odometry_observation_model_linearized()

    # Define the symbolic Jacobian
    def create_symbolic_jacobians(self):
        # Define symbols
        x, y, theta, v, w, delta_t = sp.symbols('x y theta v omega Delta_t')

        g = sp.Matrix([
            x + v * sp.cos(theta) * delta_t,
            y + v * sp.sin(theta) * delta_t,
            theta + w * delta_t
        ])

        # Define matrix g V2
        '''g = sp.Matrix([
                   x -v/w * sp.sin(theta) + v/w * sp.sin(theta + w * delta_t),
                   y + v/w * sp.cos(theta) - v/w * sp.cos(theta + w * delta_t),
                   theta + w * delta_t
                ])
        '''

        # Define matrix h
        h = sp.Matrix([x, y, theta])

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
        print("\ng = ")
        sp.pprint(g)
        
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
        
        return sp.lambdify((theta, w, delta_t, v), G, 'numpy'), sp.lambdify((theta, w, delta_t, v), V, 'numpy'), sp.lambdify((x, y, theta), H, 'numpy')

    def test_jacobian_G(self):
        print("Checking jacobian G")
        test_cases = [
            (0, 1, 0.1, 2),
            (np.pi/4, 2, 0.1, 1),
            (np.pi/2, 0.5, 0.05, 3),
            (0, 0.01, 0.5, 4)  # Edge case where w is very small
        ]
        for theta, w, delta_t, v in test_cases:
            with self.subTest(theta=theta, w=w, delta_t=delta_t, v=v):
                result_G_sp = self.G_sp(theta, w, delta_t, v)
                result_G_np = self.G_np((0.0, 0.0, theta), (v, w), delta_t)

                np.testing.assert_allclose(result_G_sp, result_G_np, atol=1e-5)

                result_V_sp = self.V_sp(theta, w, delta_t, v)
                result_V_np = self.V_np((0.0, 0.0, theta), (v, w), delta_t)

                np.testing.assert_allclose(result_V_sp, result_V_np, atol=1e-5)

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
                
                result_V_sp = self.V_sp(theta, w, delta_t, v)
                result_V_np = self.V_np((0.0, 0.0, theta), (v, w), delta_t)

                np.testing.assert_allclose(result_V_sp, result_V_np, atol=1e-5)

    def test_jacobian_H(self):
        print("Checking jacobian H")
        test_cases = [
            (0.0, 1.5, 0.1),
            (10.7, 20.3, np.pi/4),
            (50.6, 25.1, np.pi/2),
            (0.0, 0.0, 0.0)  # Edge case where w is very small
        ]
        for x, y, theta in test_cases:
            with self.subTest(x = x, y = y, theta=theta):
                
                result_H_sp = self.H_sp(x, y, theta)
                result_H_np = self.H_np(np.array([[x], [y], [theta]])) 

                np.testing.assert_allclose(result_H_sp, result_H_np, atol=1e-5)

if __name__ == "__main__":
    unittest.main()
