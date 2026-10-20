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
import matplotlib
matplotlib.use("Agg")
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

def plot_all_electrode_positions_in_mesh(vertex, face, electrode_positions, save_flag, file_path):
    # triangular mesh
    fig = plot_mesh(vertex, face, 0, 0, [])
    fig.data[0].opacity = 0.3

    # electrodes
    fig.add_trace(go.Scatter3d(
        x=electrode_positions[:, 0],
        y=electrode_positions[:, 1],
        z=electrode_positions[:, 2],
        mode='markers',
        marker=dict(size=2, color='blue', opacity=1),
        name='Electrodes',
        showlegend=False
    ))

    if save_flag == 1:
        png_path = f'{file_path}.png'
        fig.write_image(png_path, scale=2)
        crop_image(png_path)

        html_path = f'{file_path}.html'
        fig.write_html(html_path, include_plotlyjs='cdn')

def plot_electrode_electrogram(result_folder, electrode, electrode_point_info, p_id, file_name_prefix):
    # plot some example electrograms
    total_segments = electrode.shape[0]
    if total_segments <= 5:
        e_ids = np.arange(total_segments)
    else:
        e_ids = np.linspace(0, total_segments - 1, 5, dtype=int) # 5 uniformly spaced indices

    egm_unipolar = np.zeros((len(e_ids), 2500))
    unipolar_name = [None] * len(e_ids)
    egm_bipolar = np.zeros((len(e_ids), 2500))
    bipolar_name = [None] * len(e_ids)
    egm_reference = np.zeros((len(e_ids), 2500))
    reference_name = [None] * len(e_ids)
    egm_surface = np.zeros((len(e_ids), 2500))

    woi = np.zeros((len(e_ids), 2))
    k = 0
    for i in e_ids:
        temp = electrode[i]['unipolar']
        if temp is not None:
            egm_unipolar[k, :] = temp
            unipolar_name[k] = electrode[i]['unipolar_name']
            
        temp = electrode[i]['bipolar']
        if temp is not None:
            egm_bipolar[k, :] = temp
            bipolar_name[k] = electrode[i]['bipolar_name']

        temp = electrode[i]['reference']
        if temp is not None:
            egm_reference[k, :] = temp
            reference_name[k] = electrode[i]['reference_name']

        temp = electrode[i]['surface'][:,0] # 0th is the V1 channel
        if temp is not None:
            egm_surface[k, :] = temp

        woi[k, :] = electrode_point_info[i]['window_of_interest']
        k += 1

    # plot the electrograms of the example electrodes
    t = np.arange(egm_unipolar.shape[1])
    for i in range(len(e_ids)):
        fig, axs = plt.subplots(4, 1, figsize=(10, 12))

        woi_start = int(woi[i, 0])
        woi_end = int(woi[i, 1])
        
        egm = egm_unipolar[i,:]
        axs[0].plot(t[:woi_start], egm[:woi_start], color='blue') # plot before woi
        axs[0].plot(t[woi_start:woi_end], egm[woi_start:woi_end], color='red') # plot woi region
        axs[0].plot(t[woi_end:], egm[woi_end:], color='blue') # plot after woi
        title = f'Unipolar Electrograms ({unipolar_name[i]})'
        axs[0].set_title(title)
        axs[0].grid(True, alpha=1)

        egm = egm_bipolar[i,:]
        axs[1].plot(t[:woi_start], egm[:woi_start], color='blue') # plot before woi
        axs[1].plot(t[woi_start:woi_end], egm[woi_start:woi_end], color='red') # plot woi region
        axs[1].plot(t[woi_end:], egm[woi_end:], color='blue') # plot after woi
        title = f'Bipolar Electrograms ({bipolar_name[i]})'
        axs[1].set_title(title)
        axs[1].grid(True, alpha=1)

        egm = egm_reference[i,:]
        axs[2].plot(t[:woi_start], egm[:woi_start], color='blue') # plot before woi
        axs[2].plot(t[woi_start:woi_end], egm[woi_start:woi_end], color='red') # plot woi region
        axs[2].plot(t[woi_end:], egm[woi_end:], color='blue') # plot after woi
        title = f'Reference Electrograms ({reference_name[i]})'
        axs[2].set_title(title)
        axs[2].grid(True, alpha=1)

        egm = egm_surface[i,:]
        axs[3].plot(t[:woi_start], egm[:woi_start], color='blue') # plot before woi
        axs[3].plot(t[woi_start:woi_end], egm[woi_start:woi_end], color='red') # plot woi region
        axs[3].plot(t[woi_end:], egm[woi_end:], color='blue') # plot after woi
        title = 'Surface Electrograms (V1)'
        axs[3].set_title(title)
        axs[3].grid(True, alpha=1)

        plt.tight_layout()
        file_path = str(result_folder / f'{p_id}_{file_name_prefix}_electrode_{e_ids[i]}.png')
        plt.savefig(file_path)
        plt.close()

def catheter_position_in_mesh(vertex, face, catheter, file_path):
    total_segments = len(catheter['mapping_position_unipolar'])
    if total_segments <= 5:
        segment_ids = range(total_segments)
    else:
        segment_ids = np.linspace(0, total_segments - 1, 5, dtype=int) # 5 uniformly spaced indices

    for s_id in segment_ids:
        mapping_position_unipolar = catheter['mapping_position_unipolar'][s_id]
        mapping_position_bipolar = catheter['mapping_position_bipolar'][s_id]
        coronary_sinus_position_unipolar = catheter['coronary_sinus_position_unipolar'][s_id]
        coronary_sinus_position_bipolar = catheter['coronary_sinus_position_bipolar'][s_id]

        # triangular mesh
        fig = plot_mesh(vertex, face, 0, 0, [])
        fig.data[0].opacity = 0.3

        # electrodes
        fig.add_trace(go.Scatter3d(
            x=mapping_position_unipolar[:, 0],
            y=mapping_position_unipolar[:, 1],
            z=mapping_position_unipolar[:, 2],
            mode='markers',
            marker=dict(size=2, color='blue', opacity=1),
            name='Electrodes',
            showlegend=False
        ))

        fig.add_trace(go.Scatter3d(
            x=mapping_position_bipolar[:, 0],
            y=mapping_position_bipolar[:, 1],
            z=mapping_position_bipolar[:, 2],
            mode='markers',
            marker=dict(size=2, color='cyan', opacity=1),
            name='Electrodes',
            showlegend=False
        ))

        if coronary_sinus_position_unipolar is not None:
            fig.add_trace(go.Scatter3d(
                x=coronary_sinus_position_unipolar[:, 0],
                y=coronary_sinus_position_unipolar[:, 1],
                z=coronary_sinus_position_unipolar[:, 2],
                mode='markers',
                marker=dict(size=2, color='yellow', opacity=1),
                name='Electrodes',
                showlegend=False
            ))

        if coronary_sinus_position_bipolar is not None:
            fig.add_trace(go.Scatter3d(
                x=coronary_sinus_position_bipolar[:, 0],
                y=coronary_sinus_position_bipolar[:, 1],
                z=coronary_sinus_position_bipolar[:, 2],
                mode='markers',
                marker=dict(size=2, color='magenta', opacity=1),
                name='Electrodes',
                showlegend=False
            ))

        # save the figure
        png_path = f'{file_path}_{s_id}.png'
        fig.write_image(png_path, scale=2)
        crop_image(png_path)

        html_path = f'{file_path}_{s_id}.html'
        fig.write_html(html_path, include_plotlyjs='cdn')

def simplify_electrode_name(electrode_name):
    # remove the common prefix in the electrode names to make it shorter for display
    prefixes = ['MCC_Dx_UniPolar_', 'MCC_Dx_BiPolar_', 'Deca', '20A_', '20B_', 'CS_']
    
    # Handle both single string and list of strings
    is_single_string = isinstance(electrode_name, str)
    if is_single_string:
        electrode_name = [electrode_name]
    
    simplified_names = []
    for name in electrode_name:
        simplified_name = name
        for prefix in prefixes:
            # replace all occurrences of the prefix in this name
            simplified_name = simplified_name.replace(prefix, '')
        simplified_names.append(simplified_name)
    
    # return a string if input was a string, otherwise return list
    return simplified_names[0] if is_single_string else simplified_names

def catheter_electrogram_draw(egm, electrode_name, egm_ref, reference_channel_name, woi, title, ax):
    electrode_name_display = simplify_electrode_name(electrode_name)
    reference_channel_name_display = simplify_electrode_name(reference_channel_name)
    
    # find out the scale factor
    egm_display = np.zeros_like(egm)
    egm_magnitude = np.zeros(egm.shape[1])
    for e_id in range(egm.shape[1]):
        temp = egm[:, e_id]
        egm_magnitude[e_id] = np.max(temp) - np.min(temp)
    scale_factor = 1 / np.max(egm_magnitude) # scale the largest amplitude to 1

    for e_id in range(egm.shape[1]):
        temp = egm[:, e_id]
        temp = temp * scale_factor # normalize
        egm_display[:, e_id] = temp + e_id + 1

    egm_ref_magnitude = np.max(egm_ref) - np.min(egm_ref)
    egm_ref_display = egm_ref / egm_ref_magnitude # normalize

    t = np.arange(egm.shape[0])
    woi_start = int(woi[0])
    woi_end = int(woi[1])

    # reference electrogram
    ax.plot(t[:woi_start], egm_ref_display[:woi_start], color='c') # plot before woi
    ax.plot(t[woi_start:woi_end], egm_ref_display[woi_start:woi_end], color='red') # plot woi region
    ax.plot(t[woi_end:], egm_ref_display[woi_end:], color='c') # plot after woi

    # electrogram of each electrode of the catheter
    for i in range(egm_display.shape[1]):
        ax.plot(t[:woi_start], egm_display[:woi_start, i], color='b') # plot before woi
        ax.plot(t[woi_start:woi_end], egm_display[woi_start:woi_end, i], color='red') # plot woi region
        ax.plot(t[woi_end:], egm_display[woi_end:, i], color='b') # plot after woi

    ax.set_title(title)
    ax.set_xlabel('Time (ms)')
    ax.set_yticks([0] + list(range(1, egm_display.shape[1] + 1)))
    electrode_name_and_magnitude = [f'{name}, {mag:.2f}' for name, mag in zip(electrode_name_display, egm_magnitude)]
    ax.set_yticklabels([f'{reference_channel_name_display}, {egm_ref_magnitude:.2f}'] + electrode_name_and_magnitude)
    ax.grid(True, alpha=1)
    plt.tight_layout()

def catheter_electrogram(catheter, result_folder, p_id, file_name_prefix):
    # example recording segments to plot
    N = len(catheter['mapping_electrogram_unipolar'])
    if N <= 5:
        segment_ids = range(N)
    else:
        segment_ids = np.linspace(0, N - 1, 5, dtype=int) # 5 uniformly spaced indices
    
    mapping_unipolar_electrode_name = catheter['mapping_name_unipolar']
    mapping_bipolar_electrode_name = catheter['mapping_name_bipolar']
    coronary_sinus_unipolar_electrode_name = catheter['coronary_sinus_name_unipolar']
    coronary_sinus_bipolar_electrode_name = catheter['coronary_sinus_name_bipolar']
    reference_channel_name = catheter['reference_name']
    surface_channel_names = catheter['surface_name']

    for s_id in segment_ids:
        mapping_electrogram_unipolar = catheter['mapping_electrogram_unipolar'][s_id]
        mapping_electrogram_bipolar = catheter['mapping_electrogram_bipolar'][s_id]
        coronary_sinus_electrogram_unipolar = catheter['coronary_sinus_electrogram_unipolar'][s_id]
        coronary_sinus_electrogram_bipolar = catheter['coronary_sinus_electrogram_bipolar'][s_id]
        surface_electrogram = catheter['surface_electrogram'][s_id]
        woi = catheter['window_of_interest'][s_id]
        egm_ref = catheter['reference_electrogram'][s_id]

        fig, ax = plt.subplots(1, 5, figsize=(20, 8))

        # mapping catheter unipolar electrograms
        if mapping_electrogram_unipolar.shape[1] != 0:
            catheter_electrogram_draw(mapping_electrogram_unipolar, mapping_unipolar_electrode_name, egm_ref, reference_channel_name[s_id], woi, 'Unipolar Electrograms', ax[0])

        # mapping catheter bipolar electrograms
        if mapping_electrogram_bipolar.shape[1] != 0:
            catheter_electrogram_draw(mapping_electrogram_bipolar, mapping_bipolar_electrode_name, egm_ref, reference_channel_name[s_id], woi, 'Bipolar Electrograms', ax[1])

        # surface electrograms
        catheter_electrogram_draw(surface_electrogram, surface_channel_names, egm_ref, reference_channel_name[s_id], woi, 'Surface Electrograms', ax[2])

        # coronary sinus catheter bipolar electrograms
        if coronary_sinus_electrogram_bipolar.shape[1] != 0:
            number_of_cs_bipolar = coronary_sinus_electrogram_bipolar.shape[1]
            catheter_electrogram_draw(coronary_sinus_electrogram_bipolar, coronary_sinus_bipolar_electrode_name[0:number_of_cs_bipolar], egm_ref, reference_channel_name[s_id], woi, 'Coronary Sinus Bipolar Electrograms', ax[3])

        # coronary sinus catheter unipolar electrograms
        if coronary_sinus_electrogram_unipolar.shape[1] != 0:
            number_of_cs_unipolar = coronary_sinus_electrogram_unipolar.shape[1]
            catheter_electrogram_draw(coronary_sinus_electrogram_unipolar, coronary_sinus_unipolar_electrode_name[0:number_of_cs_unipolar], egm_ref, reference_channel_name[s_id], woi, 'Coronary Sinus Unipolar Electrograms', ax[4])

        # save the figure
        file_path = str(result_folder / f'{p_id}_{file_name_prefix}_electrode_{s_id}.png')
        plt.savefig(file_path)
        plt.close()
