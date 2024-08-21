import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations
import os

class plot_3D:
    def __init__(self, i, V = (480, 96, 114), alpha = 0.4, style = 'default'):
        if style == 'xkcd':
            plt.xkcd()
        else:
            plt.style.use(style)
        self.fig = plt.figure(figsize = (8, 8))
        self.ax = self.fig.add_subplot(projection='3d')
        plt.title(f'Container {i}')
        
        plt.margins(0, 0, 0)
        self.alpha = alpha
        self.V = V
        self.ax.set_xlim(0, V[1])
        self.ax.set_ylim(V[0], 0)
        self.ax.set_zlim(0, V[2])
        self.ax.set_xlabel('Y axis')
        self.ax.set_ylabel('X axis')
        self.ax.set_zlabel('Z axis')
        
        scale_x = V[1] * 2 / (V[0] + V[1] + V[2])
        scale_y = V[0] * 2 / (V[0] + V[1] + V[2])
        scale_z = V[2] * 2 / (V[0] + V[1] + V[2])
        self.ax.get_proj = lambda: np.dot(Axes3D.get_proj(self.ax), np.diag([scale_x, scale_y, scale_z, 1]))
        
        self.boxes_num = 0
        self.boxes_df = pd.DataFrame(columns=['box', 'min_coord', 'max_coord', 'sides', 'color'])
        self.colors = []  # Store colors for legend
    
    def add_box(self, v1, v2, mode = 'EMS'):
        min_coord = np.array(list(v1))
        v2 = np.array(list(v2))
           
        if mode == 'vector':
            sides = v2
            max_coord = [v1[i] + sides[i] for i in range(3)]
        elif mode == 'EMS':
            sides = [v2[i] - v1[i] for i in range(3)]
            max_coord = v2
        
        y_range, x_range, z_range = zip(min_coord, max_coord)
        
        # random color
        color = np.random.rand(3,)
        label = f'Box: {sides}'
        self.colors.append((color, label))  # Store color and dimensions for legend
        
        # Draw 6 Faces
        xx, xy, xz = np.meshgrid(x_range, y_range, z_range)  # X
        yy, yx, yz = np.meshgrid(y_range, x_range, z_range)  # Y
        zx, zz, zy = np.meshgrid(x_range, z_range, y_range)  # Z
        
        zorder_val = min(z_range)**2 + min(x_range)**2 + min(y_range)**2
        
        for i in range(2):
            self.ax.plot_wireframe(xx[i], xy[i], xz[i], color=color, zorder=zorder_val)
            self.ax.plot_surface(xx[i], xy[i], xz[i], color=color, alpha=self.alpha, zorder=zorder_val)
            self.ax.plot_wireframe(yx[i], yy[i], yz[i], color=color, zorder=zorder_val)
            self.ax.plot_surface(yx[i], yy[i], yz[i], color=color, alpha=self.alpha, zorder=zorder_val)
            self.ax.plot_wireframe(zx[i], zy[i], zz[i], color=color, zorder=zorder_val)
            self.ax.plot_surface(zx[i], zy[i], zz[i], color=color, alpha=self.alpha, zorder=zorder_val)
        
        # Record
        self.boxes_df = pd.concat([self.boxes_df, pd.DataFrame({'box': self.boxes_num, 'sides': sides,
                                                                'min_coord': min_coord, 'max_coord': max_coord,
                                                                'color': color})], ignore_index=True)
        self.boxes_num += 1
    
    def show(self, fig, filename, elev = None, azim = None):
        self.ax.view_init(elev=elev, azim=azim)
        # Add legend
        handles = [plt.Line2D([0], [0], color=color, lw=4) for color, label in self.colors]
        labels = [label for color, label in self.colors]
        self.ax.legend(handles, labels)
        save_path = os.path.join('static', filename)
        fig.savefig(save_path)
        plt.close(fig)
