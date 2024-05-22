import sympy as sp
import matplotlib.pyplot as plt

def plot_matrix(matrix, title=None, label=""):
    fig, ax = plt.subplots()
    ax.axis('off')
    
    # Adjust title position and pad to give more space
    if title:
        ax.set_title(title, pad=20)  # Increase pad to move the title up

    # Use SymPy to generate a preview (image) of the matrix
    filename = 'output.png'
    sp.preview(matrix, viewer='file', filename=filename, euler=False, dvioptions=['-D', '1200'])

    # Load and display the image
    img = plt.imread(filename)
    ax.imshow(img)

    # If a label is provided, use matplotlib to annotate the image
    if label:
        ax.text(0, 0.5, f"{label} =", va='center', ha='right', transform=ax.transAxes, fontsize=30, color='black')  # Increase fontsize

    plt.show()

# Example of usage
x, y, theta, v, w, delta_t = sp.symbols('x y theta v omega Delta_t')
g = sp.Matrix([
           x - v/w * sp.sin(theta) + v/w * sp.sin(theta + w * delta_t),
           y + v/w * sp.cos(theta) - v/w * sp.cos(theta + w * delta_t),
           theta + w * delta_t
        ])
state = sp.Matrix([x, y, theta])
G = g.jacobian(state)

# Define matrix h
h = sp.Matrix([x, y, theta])
H = h.jacobian(state)

plot_matrix(H, label="H")
