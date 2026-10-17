# Copyright 2026 Jiyue He
# Contact: jiyuehe@alumni.upenn.edu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from PIL import Image # pip install pillow
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import subprocess
import sys
import plotly.graph_objects as go

# obj. 3D mesh file
# ==================================================
def load_obj(file_directory, mesh_name):
    file_path = file_directory / f'{mesh_name}.obj'

    vertex = []
    face = []
    with open(file_path, 'r', encoding='utf-8') as fid:
        for line in fid:
            if line.startswith('v '):
                vertex.append(np.fromstring(line[2:], sep=' '))
            elif line.startswith('f '):
                # subtract 1 to convert 1-based OBJ indices to 0-based python indices
                face.append(np.fromstring(line[2:], sep=' ', dtype=int) - 1)

    vertex =  np.asarray(vertex)
    face = np.asarray(face, dtype=int)

    return vertex, face

def save_obj(file_directory, mesh_name, vertex, face):
    file_path = file_directory / f'{mesh_name}.obj'

    with open(file_path, 'w', encoding='utf-8') as fid:
        for v in vertex:
            fid.write(f'v {v[0]} {v[1]} {v[2]}\n')
        for f in face:
            # add 1 to convert 0-based python indices to 1-based OBJ indices
            fid.write(f'f {f[0]+1} {f[1]+1} {f[2]+1}\n')

# load npz file
# ==================================================
def load_npz(file_path):
    data_temp = np.load(file_path, allow_pickle=True)
    data = {}
    for key in data_temp.files:
        value = data_temp[key]
        if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
            value = value.item()
        data[key] = value
        
    return data

# color mapping
# ==================================================
def convert_value_to_red_blue(data, data_min, data_max, data_threshold):
    non_active_id = (data <= data_threshold) | np.isnan(data) | (data == -1) # identify non-active regions (below threshold or NaN) before processing
    data_clipped = np.clip(data, data_min, data_max) # clip values (NaN will remain NaN after clipping)
    hue = (data_clipped - data_min) / (data_max - data_min) * (240.0 / 360.0) # calculate hue (NaN values will produce NaN hue)

    # assign color using HSV colormap
    hsv = np.zeros((hue.size, 3))
    hsv[:, 0] = np.nan_to_num(hue, nan=0.0)  # Replace NaN with 0 for color calculation
    hsv[:, 1] = 1.0
    hsv[:, 2] = 1.0
    map_color = mcolors.hsv_to_rgb(hsv)

    map_color[non_active_id, :] = 0.5 # assign non-active regions (including NaN) to gray

    return map_color

def convert_value_to_red_magenta(data, data_min, data_max, data_threshold):
    non_active_id = (data <= data_threshold) | np.isnan(data) # identify non-active regions (below threshold or NaN) before processing
    data_clipped = np.clip(data, data_min, data_max) # clip values (NaN will remain NaN after clipping)
    normalized = (data_clipped - data_min) / (data_max - data_min)

    # full wrapped hue cycle: 0 and 1 both become red
    hue = np.mod(normalized, 1.0)

    # assign color using HSV colormap
    hsv = np.zeros((hue.size, 3))
    hsv[:, 0] = np.nan_to_num(hue, nan=0.0)  # Replace NaN with 0 for color calculation
    hsv[:, 1] = 1.0
    hsv[:, 2] = 1.0
    map_color = mcolors.hsv_to_rgb(hsv)

    map_color[non_active_id, :] = 0.5 # assign non-active regions (including NaN) to gray
    
    return map_color

def convert_value_to_purple_yellow(data, data_min, data_max, data_threshold):
    non_active_id = (data <= data_threshold) | np.isnan(data) # identify non-active regions (below threshold or NaN) before processing
    data_clipped = np.clip(data, data_min, data_max) # clip values (NaN will remain NaN after clipping)
    normalized = (data_clipped - data_min) / (data_max - data_min) # normalize data to range [0, 1]
    viridis = plt.cm.viridis # use Viridis colormap (purple/dark blue -> cyan -> green -> yellow)
    map_color = viridis(np.nan_to_num(normalized, nan=0.0))[:, :3]  # take only RGB, drop alpha
    map_color[non_active_id, :] = 0.5 # assign non-active regions to gray
    
    return map_color

# figure setting related
# ==================================================
def set_axes_equal(ax): 
    # make 3D plot axes have equal scale so spheres look like spheres
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    max_range = max([abs(x_limits[1] - x_limits[0]), abs(y_limits[1] - y_limits[0]), abs(z_limits[1] - z_limits[0])]) / 2.0
    ax.set_xlim3d([np.mean(x_limits) - max_range, np.mean(x_limits) + max_range])
    ax.set_ylim3d([np.mean(y_limits) - max_range, np.mean(y_limits) + max_range])
    ax.set_zlim3d([np.mean(z_limits) - max_range, np.mean(z_limits) + max_range])

def crop_image(image_file_name): # crop image to remove excess whitespace
    img = Image.open(image_file_name).convert("RGBA")
    data = np.array(img)

    threshold = 245  # define near-white as background
    mask = np.any(data[:, :, :3] < threshold, axis=2)
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    img.crop((x0, y0, x1, y1)).save(image_file_name)

# plots
# ==================================================
def plot_mesh(vertex, face, save_flag, edge_flag, file_path_mesh):
    vertex = np.asarray(vertex)
    face = np.asarray(face, dtype=int)

    # Convert Matplotlib-style elev/azim to Plotly camera eye coordinates.
    # elev=70, azim=-70 corresponds to a camera elevated above the X-Y plane and rotated in the negative azimuth direction.
    elev_deg = 70.0
    azim_deg = -70.0
    radius = 1.5
    azim_rad = np.deg2rad(azim_deg)
    elev_rad = np.deg2rad(elev_deg)
    camera_eye = dict(
        x=radius * np.cos(elev_rad) * np.cos(azim_rad),
        y=radius * np.cos(elev_rad) * np.sin(azim_rad),
        z=radius * np.sin(elev_rad),
    )

    mesh = go.Mesh3d(
        x=vertex[:, 0],
        y=vertex[:, 1],
        z=vertex[:, 2],
        i=face[:, 0],
        j=face[:, 1],
        k=face[:, 2],
        opacity=1.0,
        color='lightgray',
        flatshading=False,
        hoverinfo='skip',
        showscale=False,
    )

    fig = go.Figure(data=[mesh])
    if edge_flag == 1:
        edge_color = 'darkgray'
        edge_width = 1.0

        edge_x = []
        edge_y = []
        edge_z = []
        for tri in face:
            for a, b in [(0, 1), (1, 2), (2, 0)]:
                edge_x.extend([vertex[tri[a], 0], vertex[tri[b], 0], None])
                edge_y.extend([vertex[tri[a], 1], vertex[tri[b], 1], None])
                edge_z.extend([vertex[tri[a], 2], vertex[tri[b], 2], None])

        edge_trace = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(color=edge_color, width=edge_width),
            hoverinfo='skip',
            showlegend=False,
        )
        fig.add_trace(edge_trace)

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',
            camera=dict(
                eye=camera_eye,
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1),
            ),
            dragmode='orbit',
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            camera_eye=camera_eye,
            camera_center=dict(x=0, y=0, z=0),
            camera_up=dict(x=0, y=0, z=1),
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
    )

    if save_flag == 1:
        png_path = f'{file_path_mesh}.png'
        fig.write_image(png_path, scale=2)
        crop_image(png_path)

        html_path = f'{file_path_mesh}.html'
        fig.write_html(html_path, include_plotlyjs='cdn')

    return fig

def plot_all_electrode_positions_in_mesh(vertex, face, electrode_positions, file_path):
    save_flag = 0
    edge_flag = 0
    file_path_mesh = []
    fig = plot_mesh(vertex, face, save_flag, edge_flag, file_path_mesh)






    # grab the electrode locations
    xyz = electrode_positions

    # 3D scatter plot (plotly)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=vertex[:, 0],
        y=vertex[:, 1],
        z=vertex[:, 2],
        mode='markers',
        marker=dict(size=1, color='lightgray', opacity=0.5),
        name='Geometry',
        showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode='markers',
        marker=dict(size=3, color='blue', opacity=1),
        name='Electrodes',
        showlegend=False
    ))
    fig.update_layout(
        scene=dict(
            dragmode='orbit',
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        width=1000,
        height=800
    )
    fig.show()