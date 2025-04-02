import os
import json
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import ViconProcess
import LegModel


def butter_lowpass_filter(raw_data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, raw_data)
    return y


# Config Sets: ['default', 'fx_est_each', 'fz_est_each', 'fx_meas_each', 'fz_meas_each', 'single_leg_est', 'mpc_body_sim', 'mpc_force_sim']
conifg_set = 'mpc_body_real'
# conifg_set = ''
set_ylim = False

ros_data_file = 'data/0402_mpc/0402_mpc_7474_obstacle.csv'
force_data_file = 'data/0402_mpc/vicon/mpc_7474_obstacle.csv'
# force_data_file = ''
# force_data_file = 'data_old/0328_mpc/vicon/0328_mpc_4.csv'

start_idx = 0
end_idx = -1
vicon_offest = 0

# 18673-8000+60 5
# 5659-500 4
# 5200-500 3

def process_data(data):
    data[0][1] -= data[0][1][0]
    data[1][1] -= data[1][1][0]
    # data[2][1] = butter_lowpass_filter(raw_data=data[2][1], cutoff=5, fs=1000)
    for i in range(4):
        # data[i][1] *= -1
        # data[i][1] = 0
        # data[i][0] -= 0.68 * 9.81
        # data[i][2] *= -1
        # data[i][1] = butter_lowpass_filter(raw_data=data[i][1], cutoff=5, fs=1000)
        # data[i][2] = butter_lowpass_filter(raw_data=data[i][2], cutoff=100, fs=1000)
        
        pass
    
    return data


def set_config(config):
    config['row_col'] = [1, 1]
    config['fig_size'] = [8, 6]
    config['titles'] = ["Data Plot"]
    # config['data'] = [["force_Fy_a", "force_Fy_b", "force_Fy_c", "force_Fy_d"]]
    # config['data'] = [["imp_cmd_Fy_a", "imp_cmd_Fy_b", "imp_cmd_Fy_c", "imp_cmd_Fy_d"]]
    config['data'] = [["state_vel_r_a", "state_vel_l_a", "state_trq_r_a", "state_trq_l_a",  "cmd_trq_r_a", "cmd_trq_l_a"],]
    config['labels'] = config['data']
    config['xy_labels'] = [["Time (ms)", "Angle (rad)"]]
    config['ylims'] = [[0, 2]]
    
    config['colors'] = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    config['styles'] = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--"]
    
    return config


def read_data(ros_data_file, force_data_file):
    ros_data = pd.read_csv(ros_data_file) if ros_data_file else None
    force_data = None
    trigger_idx = 0

    if force_data_file:
        try:
            force_data = pd.read_csv(force_data_file)
        except:
            force_data, trigger_idx = ViconProcess.read_csv(force_data_file)
            # trigger_idx-=5023-4255
            force_data = force_data.iloc[trigger_idx:, :].iloc[vicon_offest:, :]

    if ros_data is not None:
        ros_data = ros_data.reset_index(drop=True)
    if force_data is not None:
        force_data = force_data.reset_index(drop=True)

    data_list = [df for df in [ros_data, force_data] if df is not None]
    df_data = pd.concat(data_list, axis=1)
    
    qx = df_data['imu_orien_x'].to_numpy()
    qy = df_data['imu_orien_y'].to_numpy()
    qz = df_data['imu_orien_z'].to_numpy()
    qw = df_data['imu_orien_w'].to_numpy()
    
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    pitch = np.arcsin(2 * (qw * qy - qz * qx))

    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    
    df_data['imu_roll'] = -roll_deg
    df_data['imu_pitch'] = pitch_deg
    
    cmd_pos_x = []
    for i in range(5000):
        cmd_pos_x.append(0)
        
    for i in range(3000):
        cmd_pos_x.append(0.5*0.1/3*i*i*0.001*0.001)
    
    for i in range(df_data.__len__()-8000):
        cmd_pos_x.append(cmd_pos_x[-1]+0.0001)

    df_data['cmd_pos_x'] = np.array(cmd_pos_x)
    
    return df_data
    

if __name__ == '__main__':
    df_data = read_data(ros_data_file=ros_data_file, force_data_file=force_data_file)

    with open('DataProcess/Config.json', 'r') as file:
        config_file = json.load(file)
        print('Config Sets:', list(config_file.keys()))
        
        config = config_file['default']
        if conifg_set != '':
            for key in config_file[conifg_set]:
                config[key] = config_file[conifg_set][key]
        else:
            config = set_config(config)
    
    [fig_row, fig_col]  = config['row_col']
    fig_size = config['fig_size']
    titles = config['titles']
    data_cols = config['data']
    
    labels = config['labels']

    if np.array(labels).shape.__len__() == 1: labels = [labels for i in range(fig_row*fig_col)]
    
    colors = config['colors']
    if np.array(colors).shape.__len__() == 1: colors = [colors for i in range(fig_row*fig_col)]
    
    xy_labels = config['xy_labels']
    if np.array(xy_labels).shape.__len__() == 1: xy_labels = [xy_labels for i in range(fig_row*fig_col)]
    
    styles = config['styles']
    if np.array(styles).shape.__len__() == 1: styles = [styles for i in range(fig_row*fig_col)]
    
    ylims  = config['ylims']
    if np.array(ylims).shape.__len__() == 1: ylims = [ylims for i in range(fig_row*fig_col)]
    
    data = [df_data[col].to_numpy()[start_idx:end_idx].T for col in data_cols]
    
    data = process_data(data)

    # Start Plotting
    fig = plt.figure(figsize=fig_size)
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
    
    
    linewidth = 1.5
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.show()