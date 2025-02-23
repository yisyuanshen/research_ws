import os
import json
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import sys
sys.path.append(os.path.join(os.environ['HOME'], 'research_ws/LegWheel'))

import LegModel
import ViconProcess


def butter_lowpass_filter(raw_data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, raw_data)
    return y


### user config
file_1 = True
# filefolder_1 = 'corgi_ws/corgi_ros_ws/output_data'
filefolder_1 = 'research_ws/data/0221'
filename_1 = 'fsm_test_forward.csv'

file_2 = False
# filefolder_2 = 'corgi_ws/corgi_ros_ws/output_data'
filefolder_2 = 'research_ws/data/0220/vicon'
filename_2 = 'imp_stance_2.csv'

start_idx = 0
end_idx = 50000
vicon_idx = 0


### load data
df_data_1 = pd.DataFrame()
if file_1:
    filepath_1 = os.path.join(os.environ['HOME'], filefolder_1, filename_1)
    try: df_data_1 = pd.read_csv(filepath_1)
    except: df_data_1 = ViconProcess.read_csv(filepath_1, start_idx=vicon_idx)

df_data_2 = pd.DataFrame()
if file_2:
    filepath_2 = os.path.join(os.environ['HOME'], filefolder_2, filename_2)
    try: df_data_2 = pd.read_csv(filepath_2)
    except: df_data_2 = ViconProcess.read_csv(filepath_2, start_idx=vicon_idx)

df_data = pd.concat([df_data_1, df_data_2], axis=1)

# end_idx = min(df_data_1.__len__(), df_data_2.__len__())

### plot config
load_config = True
set_ylim = True
config_name = "eta_state"

if not load_config:
    config = {
        "figure": {
            "rows": 2,
            "cols": 1,
            "size": [8, 6]
        },
        "plots": [
            {
                "title": "Theta",
                "data": ["cmd_theta_a", "cmd_theta_b", "cmd_theta_c", "cmd_theta_d"],
                "labels": ["A", "B", "C", "D"],
                "xy_labels": ["Time (ms)", "Angle (rad)"],
                "line_styles": ["-", "--", "-.", ":"],
                "colors": ["red", "blue", "black", "green"],
                "ylims": [0, 2.5]
            },
            {
                "title": "Beta",
                "data": ["cmd_beta_a", "cmd_beta_b", "cmd_beta_c", "cmd_beta_d"],
                "labels": ["A", "B", "C", "D"],
                "xy_labels": ["Time (ms)", "Angle (rad)"],
                "line_styles": ["-", "--", "-.", ":"],
                "colors": ["red", "blue", "black", "green"],
                "ylims": [-20, 20]
            }
        ]
    }
else:
    with open(os.path.join(os.getcwd(), 'DataProcess', 'PlotConfig.json'), 'r') as file:
        config = json.load(file)[config_name]

### load config
fig_row  = config['figure']['rows']
fig_col  = config['figure']['cols']
fig_size = config['figure']['size']
target_data = [c['data'] for c in config['plots']]
titles = [c['title'] for c in config['plots']]
labels = [c['labels'] for c in config['plots']]
xy_labels = [c['xy_labels'] for c in config['plots']]
styles = [c['line_styles'] for c in config['plots']]
colors = [c['colors'] for c in config['plots']]
ylims  = [c['ylims'] for c in config['plots']]

data = [df_data[col].to_numpy()[start_idx:end_idx, :].T for col in target_data]


### additional process
# legmodel = LegModel.LegModel(sim=False)

# legmodel.contact_map(data[0][0], data[1][0])
# data[0][0] = legmodel.contact_p[:, 0]
# data[1][0] = legmodel.contact_p[:, 1]

# legmodel.contact_map(data[0][1], data[1][1])
# data[0][1] = legmodel.contact_p[:, 0]
# data[1][1] = legmodel.contact_p[:, 1]

# data[3][1] = data[3][1]*47/53


### plot data
fig = plt.figure(figsize=(8, 6))
gs = GridSpec(fig_row, fig_col, figure=fig)

axes = [fig.add_subplot(gs[row, col]) for row in range(fig_row) for col in range(fig_col)]

linewidth = 1.5

for fig_idx in range(len(axes)):
    for data_idx in range(len(data[fig_idx])):
        axes[fig_idx].plot(range(data[fig_idx].shape[1]), data[fig_idx][data_idx], label=labels[fig_idx][data_idx], linewidth=linewidth, linestyle=styles[fig_idx][data_idx],  color=colors[fig_idx][data_idx])

    axes[fig_idx].set_title(titles[fig_idx], fontsize=14)
    axes[fig_idx].set_xlabel(xy_labels[fig_idx][0], fontsize=12)
    axes[fig_idx].set_ylabel(xy_labels[fig_idx][1], fontsize=12)
    axes[fig_idx].legend(fontsize=10, loc='best', frameon=True, shadow=True, facecolor='white', edgecolor='black')
    axes[fig_idx].grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.8)
    axes[fig_idx].tick_params(axis='both', which='major', labelsize=10)
    axes[fig_idx].autoscale(enable=True, axis='both', tight=True)
    # axes[fig_idx].set_facecolor('#F7F7F7')
    if set_ylim: axes[fig_idx].set_ylim(ylims[fig_idx])

plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
plt.show()