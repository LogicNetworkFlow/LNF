import numpy as np
import matplotlib.pyplot as plt

# Load and visualize terrain data
terrain_x_original, terrain_y_original, roughness_original = np.load('3_terrain_data.npz').values()

plt.figure(figsize=(10, 8))
plt.imshow(roughness_original, 
           extent=[terrain_x_original.min(), terrain_x_original.max(), 
                   terrain_y_original.min(), terrain_y_original.max()],
           origin='lower', cmap='terrain')

import pdb
pdb.set_trace()

plt.colorbar(label='Roughness')
plt.title('Terrain Roughness Map')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.show()

