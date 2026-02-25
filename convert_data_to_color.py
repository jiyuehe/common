import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

def execute(data, data_min, data_max, data_threshold):
    # identify non-active regions (below threshold or NaN) before processing
    non_active_id = (data <= data_threshold) | np.isnan(data) | (data == -1)
    
    # clip values (NaN will remain NaN after clipping)
    data_clipped = np.clip(data, data_min, data_max)
    
    # calculate hue (NaN values will produce NaN hue)
    hue = (data_clipped - data_min) / (data_max - data_min) * (240.0 / 360.0)

    # assign color using HSV colormap
    hsv = np.zeros((hue.size, 3))
    hsv[:, 0] = np.nan_to_num(hue, nan=0.0)  # Replace NaN with 0 for color calculation
    hsv[:, 1] = 1.0
    hsv[:, 2] = 1.0
    map_color = mcolors.hsv_to_rgb(hsv)

    # assign non-active regions (including NaN) to gray
    map_color[non_active_id, :] = 0.5

    return map_color

def purple_yellow(data, data_min, data_max, data_threshold):
    # identify non-active regions (below threshold or NaN) before processing
    non_active_id = (data <= data_threshold) | np.isnan(data)
    
    # clip values (NaN will remain NaN after clipping)
    data_clipped = np.clip(data, data_min, data_max)
    
    # normalize data to range [0, 1]
    normalized = (data_clipped - data_min) / (data_max - data_min)
    
    # use Viridis colormap (purple/dark blue -> cyan -> green -> yellow)
    viridis = plt.cm.viridis
    
    # apply colormap to normalized data
    map_color = viridis(np.nan_to_num(normalized, nan=0.0))[:, :3]  # take only RGB, drop alpha
    
    # assign non-active regions (including NaN) to gray
    map_color[non_active_id, :] = 0.5
    
    return map_color
