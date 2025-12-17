import trimesh
import numpy as np

# Load the .obj file
mesh = trimesh.load_mesh('/Users/george/Downloads/Skull_v3_L2.123c1407fc1e-ea5c-4cb9-9072-d28b8aba4c36/12140_Skull_v3_L2.obj')

# Get the vertices as a NumPy array
skull_model = np.array(mesh.vertices)

# Save the model
np.save('skull_model.npy', skull_model)
