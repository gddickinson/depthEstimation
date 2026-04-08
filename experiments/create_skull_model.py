import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_skull_model(resolution=50):
    # Create a grid of points
    theta = np.linspace(0, 2 * np.pi, resolution)
    phi = np.linspace(0, np.pi, resolution)
    theta, phi = np.meshgrid(theta, phi)

    # Basic sphere for the cranium
    r = 1
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    # Flatten the jaw area
    mask = z < -0.2
    z[mask] = -0.2

    # Create eye sockets
    eye_socket = (x > 0.3) & (x < 0.7) & (y > -0.4) & (y < 0.4) & (z > 0)
    z[eye_socket] -= 0.2

    # Create nasal cavity
    nose = (x > 0.8) & (y > -0.2) & (y < 0.2) & (z > -0.2) & (z < 0.2)
    x[nose] += 0.2

    # Combine x, y, z into a single array of points
    points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T

    return points

# Create the skull model
skull_model = create_skull_model()

# Save the model
np.save('skull_model.npy', skull_model)

# Visualize the model
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(skull_model[:, 0], skull_model[:, 1], skull_model[:, 2], s=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Simplified 3D Skull Model')
plt.show()

print("Skull model created and saved as 'skull_model.npy'")
