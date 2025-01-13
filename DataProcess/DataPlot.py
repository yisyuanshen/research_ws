import os
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def butter_lowpass_filter(raw_data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, raw_data)
    return y

#%%
filefolder = 'corgi_ws/corgi_ros_ws/output_data'
filename = '3.csv'

filepath = os.path.join(os.getenv('HOME'), filefolder, filename)

start_idx = 2000
end_idx = 6000

df_data = pd.read_csv(filepath)

#%%
target_columns = [['imp_cmd_theta_a', 'cmd_theta_a', 'state_theta_a'],
                  ['imp_cmd_Fy_a', 'force_Fy_a']]

line_labels = [['Theta Imp Cmd', 'Theta Motor Cmd', 'Theta State'],
               ['Force Imp Cmd', 'Force State']]

xy_labels = [['Time (ms)', 'Theta (rad)'],
             ['Time (ms)', 'Force (N)']]

titles = ['Theta',
          'Force']

data = [df_data[col].to_numpy()[start_idx:end_idx, :].T for col in target_columns]

# data[1][0] = butter_lowpass_filter(data[1][0], cutoff=100, fs=1000, order=5)

#%%
fig_row = 2
fig_col = 1

fig = plt.figure(figsize=(12, 8))
gs = GridSpec(fig_row, fig_col, figure=fig)

axes = [fig.add_subplot(gs[row, col]) for row in range(fig_row) for col in range(fig_col)]

fig_idx = 0
axes[fig_idx].plot(range(data[fig_idx].shape[1]), data[fig_idx][0], label=line_labels[fig_idx][0], linewidth=1.5, linestyle='-',  color='red')
axes[fig_idx].plot(range(data[fig_idx].shape[1]), data[fig_idx][1], label=line_labels[fig_idx][1], linewidth=1.5, linestyle='--', color='blue')
axes[fig_idx].plot(range(data[fig_idx].shape[1]), data[fig_idx][2], label=line_labels[fig_idx][2], linewidth=1.5, linestyle='-.', color='green')
axes[fig_idx].legend(fontsize=10, loc='lower right', frameon=True, shadow=True, facecolor='white', edgecolor='black')
axes[fig_idx].set_ylim(0.9, 1.1)

fig_idx = 1
axes[fig_idx].plot(range(data[fig_idx].shape[1]), data[fig_idx][0], label=line_labels[fig_idx][0], linewidth=1.5, linestyle='-', color='red')
axes[fig_idx].plot(range(data[fig_idx].shape[1]), data[fig_idx][1], label=line_labels[fig_idx][1], linewidth=1.5, linestyle='--', color='black')
axes[fig_idx].legend(fontsize=10, loc='lower right', frameon=True, shadow=True, facecolor='white', edgecolor='black')
axes[fig_idx].set_ylim(-120, 20)

for fig_idx in range(len(axes)):
    axes[fig_idx].set_title(titles[fig_idx], fontsize=14)
    axes[fig_idx].set_xlabel(xy_labels[fig_idx][0], fontsize=12)
    axes[fig_idx].set_ylabel(xy_labels[fig_idx][1], fontsize=12)
    axes[fig_idx].grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.8)
    axes[fig_idx].tick_params(axis='both', which='major', labelsize=10)
    # axes[fig_idx].set_facecolor('#F7F7F7')

plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
plt.show()
