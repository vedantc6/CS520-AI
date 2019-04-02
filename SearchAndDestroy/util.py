import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from collections import OrderedDict

def generate_map(dim):
    mat = np.zeros([dim, dim])
    for i in range(dim):
        for j in range(dim):
            p = np.random.rand()
            if p <= 0.2:
                mat[i][j] = 0
            elif p > 0.2 and p <= 0.5:
                mat[i][j] = 1
            elif p > 0.5 and p <= 0.7:
                mat[i][j] = 2
            else: 
                mat[i][j] = 3
    
    return mat

grid = gridspec.GridSpec(ncols=2, nrows=2)

fig = plt.figure(figsize=(10,10))
f2_ax1 = fig.add_subplot(grid[0, 0])
f2_ax2 = fig.add_subplot(grid[0, 1])
f2_ax3 = fig.add_subplot(grid[1, 0])
f2_ax4 = fig.add_subplot(grid[1, 1])

a = generate_map(10)
print(a)
# Display matrix
f2_ax1.matshow(a, cmap=cm.get_cmap('Greens', 4))
f2_ax1.set_title("Actual")
f2_ax2.matshow(generate_map(10))
f2_ax3.matshow(generate_map(10))
f2_ax4.matshow(generate_map(10))

f2_ax1.scatter(5, 5, s=100, c='red', marker='x')

plt.show()

class Map