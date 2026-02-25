import numpy as np

def execute(ax): 
    # make 3D plot axes have equal scale so spheres look like spheres
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    max_range = max([abs(x_limits[1] - x_limits[0]), abs(y_limits[1] - y_limits[0]), abs(z_limits[1] - z_limits[0])]) / 2.0
    ax.set_xlim3d([np.mean(x_limits) - max_range, np.mean(x_limits) + max_range])
    ax.set_ylim3d([np.mean(y_limits) - max_range, np.mean(y_limits) + max_range])
    ax.set_zlim3d([np.mean(z_limits) - max_range, np.mean(z_limits) + max_range])
