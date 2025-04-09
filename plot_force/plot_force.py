import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

import ViconProcess
import LegModel

def butter_lowpass_filter(raw_data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, raw_data)
    return y


df = pd.read_csv("data/0409_force/0409_force_test_6.csv")
# df = pd.read_csv("data/0407_mpc/0407_walk_2.csv")

df_vicon, trigger_idx = None, None
# df_vicon, trigger_idx = pd.read_csv("data/test/sim_force_plate.csv"), 0
# df_vicon, trigger_idx = ViconProcess.read_csv("data/0407_mpc/vicon/MPC_1.csv")

start_idx = 0
end_idx = -1
vicon_offset = 0
force = False

time = df['Time'][start_idx:end_idx]

theta_cmd = [df['cmd_theta_a'][start_idx:end_idx], df['cmd_theta_b'][start_idx:end_idx], df['cmd_theta_c'][start_idx:end_idx], df['cmd_theta_d'][start_idx:end_idx]]
theta_state = [df['state_theta_a'][start_idx:end_idx], df['state_theta_b'][start_idx:end_idx], df['state_theta_c'][start_idx:end_idx], df['state_theta_d'][start_idx:end_idx]]
beta_cmd = [df['cmd_beta_a'][start_idx:end_idx], df['cmd_beta_b'][start_idx:end_idx], df['cmd_beta_c'][start_idx:end_idx], df['cmd_beta_d'][start_idx:end_idx]]
beta_state = [df['state_beta_a'][start_idx:end_idx], df['state_beta_b'][start_idx:end_idx], df['state_beta_c'][start_idx:end_idx], df['state_beta_d'][start_idx:end_idx]]

force_state_x = [df['force_Fx_a'][start_idx:end_idx], df['force_Fx_b'][start_idx:end_idx], df['force_Fx_c'][start_idx:end_idx], df['force_Fx_d'][start_idx:end_idx]]
force_state_z = [df['force_Fy_a'][start_idx:end_idx], df['force_Fy_b'][start_idx:end_idx], df['force_Fy_c'][start_idx:end_idx], df['force_Fy_d'][start_idx:end_idx]]
force_cmd_x = [df['imp_cmd_Fx_a'][start_idx:end_idx], df['imp_cmd_Fx_b'][start_idx:end_idx], df['imp_cmd_Fx_c'][start_idx:end_idx], df['imp_cmd_Fx_d'][start_idx:end_idx]]
force_cmd_z = [df['imp_cmd_Fy_a'][start_idx:end_idx], df['imp_cmd_Fy_b'][start_idx:end_idx], df['imp_cmd_Fy_c'][start_idx:end_idx], df['imp_cmd_Fy_d'][start_idx:end_idx]]

force_state_z[0] = butter_lowpass_filter(raw_data=force_state_z[0], cutoff=30, fs=1000)
force_state_z[1] = butter_lowpass_filter(raw_data=force_state_z[1], cutoff=30, fs=1000)
force_state_z[2] = butter_lowpass_filter(raw_data=force_state_z[2], cutoff=30, fs=1000)
force_state_z[3] = butter_lowpass_filter(raw_data=force_state_z[3], cutoff=30, fs=1000)

imu_qx = df['imu_orien_x'][start_idx:end_idx]
imu_qy = df['imu_orien_y'][start_idx:end_idx]
imu_qz = df['imu_orien_z'][start_idx:end_idx]
imu_qw = df['imu_orien_w'][start_idx:end_idx]
imu_roll = np.arctan2(2 * (imu_qw * imu_qx + imu_qy * imu_qz), 1 - 2 * (imu_qx**2 + imu_qy**2))
imu_pitch = np.arcsin(2 * (imu_qw * imu_qy - imu_qz * imu_qx))
imu_roll = np.rad2deg(imu_roll)
imu_pitch = np.rad2deg(imu_pitch)

if df_vicon is not None:
    # vicon_pos_x = df_vicon['vicon_pos_x'][trigger_idx+vicon_offset+start_idx:trigger_idx+vicon_offset+end_idx]
    # vicon_pos_z = df_vicon['vicon_pos_z'][trigger_idx+vicon_offset+start_idx:trigger_idx+vicon_offset+end_idx]
    
    # vicon_pos_x -= vicon_pos_x.iloc[0]
    # vicon_pos_z -= vicon_pos_z.iloc[0]

    # vicon_roll = df_vicon['vicon_roll'][trigger_idx+vicon_offset+start_idx:trigger_idx+vicon_offset+end_idx]
    # vicon_pitch = df_vicon['vicon_pitch'][trigger_idx+vicon_offset+start_idx:trigger_idx+vicon_offset+end_idx]
    
    vicon_force_x = [df_vicon['Fx_1'][trigger_idx+vicon_offset+start_idx:trigger_idx+vicon_offset+end_idx],
                     df_vicon['Fx_4'][trigger_idx+vicon_offset+start_idx:trigger_idx+vicon_offset+end_idx],
                     df_vicon['Fx_3'][trigger_idx+vicon_offset+start_idx:trigger_idx+vicon_offset+end_idx],
                     df_vicon['Fx_2'][trigger_idx+vicon_offset+start_idx:trigger_idx+vicon_offset+end_idx]]
    
leg = LegModel.LegModel(sim=False)

z_cmd = []
leg.contact_map(theta_cmd[0], beta_cmd[0])
z_cmd.append(leg.contact_p[:, 1])
leg.contact_map(theta_cmd[1], beta_cmd[1])
z_cmd.append(leg.contact_p[:, 1])
leg.contact_map(theta_cmd[2], beta_cmd[2])
z_cmd.append(leg.contact_p[:, 1])
leg.contact_map(theta_cmd[3], beta_cmd[3])
z_cmd.append(leg.contact_p[:, 1])

z_state = []
leg.contact_map(theta_state[0], beta_state[0]-np.deg2rad(imu_pitch))
z_state.append(leg.contact_p[:, 1])
leg.contact_map(theta_state[1], beta_state[1]-np.deg2rad(imu_pitch))
z_state.append(leg.contact_p[:, 1])
leg.contact_map(theta_state[2], beta_state[2]-np.deg2rad(imu_pitch))
z_state.append(leg.contact_p[:, 1])
leg.contact_map(theta_state[3], beta_state[3]-np.deg2rad(imu_pitch))
z_state.append(leg.contact_p[:, 1])


    
title_fontsize = 16
label_fontsize = 12
tick_fontsize = 12
legend_fontsize = 12

fig, axs = plt.subplots(4, 2, figsize=(12, 8))

axs[0, 0].plot(range(len(time)), force_cmd_z[0], label='Cmd', color='C0', linestyle='-')
axs[0, 0].plot(range(len(time)), -force_state_z[0], label='State', color='C1', linestyle='--')
axs[0, 0].set_ylabel('Force (N)', fontsize=label_fontsize)
axs[0, 0].set_title('Force - A', fontsize=title_fontsize)
axs[0, 0].set_ylim([-100, 0])

axs[1, 0].plot(range(len(time)), z_cmd[0], label='Cmd', color='C0', linestyle='-')
axs[1, 0].plot(range(len(time)), z_state[0], label='State', color='C1', linestyle='--')
axs[1, 0].set_ylabel('Height (m)', fontsize=label_fontsize)
axs[1, 0].set_title('Height - A', fontsize=title_fontsize)
axs[1, 0].set_ylim([-0.3, -0.1])

axs[0, 1].plot(range(len(time)), force_cmd_z[1], label='Cmd', color='C0', linestyle='-')
axs[0, 1].plot(range(len(time)), -force_state_z[1], label='State', color='C1', linestyle='--')
axs[0, 1].set_ylabel('Force (N)', fontsize=label_fontsize)
axs[0, 1].set_title('Force - B', fontsize=title_fontsize)
axs[0, 1].set_ylim([-100, 0])

axs[1, 1].plot(range(len(time)), z_cmd[1], label='Cmd', color='C0', linestyle='-')
axs[1, 1].plot(range(len(time)), z_state[1], label='State', color='C1', linestyle='--')
axs[1, 1].set_ylabel('Height (m)', fontsize=label_fontsize)
axs[1, 1].set_title('Height - B', fontsize=title_fontsize)
axs[1, 1].set_ylim([-0.3, -0.1])

axs[2, 1].plot(range(len(time)), force_cmd_z[2], label='Cmd', color='C0', linestyle='-')
axs[2, 1].plot(range(len(time)), -force_state_z[2], label='State', color='C1', linestyle='--')
axs[2, 1].set_ylabel('Force (N)', fontsize=label_fontsize)
axs[2, 1].set_title('Force - C', fontsize=title_fontsize)
axs[2, 1].set_ylim([-100, 0])

axs[3, 1].plot(range(len(time)), z_cmd[2], label='Cmd', color='C0', linestyle='-')
axs[3, 1].plot(range(len(time)), z_state[2], label='State', color='C1', linestyle='--')
axs[3, 1].set_ylabel('Height (m)', fontsize=label_fontsize)
axs[3, 1].set_title('Height - C', fontsize=title_fontsize)
axs[3, 1].set_ylim([-0.3, -0.1])

axs[2, 0].plot(range(len(time)), force_cmd_z[3], label='Cmd', color='C0', linestyle='-')
axs[2, 0].plot(range(len(time)), -force_state_z[3], label='State', color='C1', linestyle='--')
axs[2, 0].set_ylabel('Force (N)', fontsize=label_fontsize)
axs[2, 0].set_title('Force - D', fontsize=title_fontsize)
axs[2, 0].set_ylim([-100, 0])

axs[3, 0].plot(range(len(time)), z_cmd[3], label='Cmd', color='C0', linestyle='-')
axs[3, 0].plot(range(len(time)), z_state[3], label='State', color='C1', linestyle='--')
axs[3, 0].set_ylabel('Height (m)', fontsize=label_fontsize)
axs[3, 0].set_title('Height - D', fontsize=title_fontsize)
axs[3, 0].set_ylim([-0.3, -0.1])


for i in range(4):
    for j in range(2):
        axs[i, j].set_xlabel('Time (ms)', fontsize=label_fontsize)
        axs[i, j].tick_params(axis='both', labelsize=tick_fontsize)
        axs[i, j].legend(fontsize=legend_fontsize)
        axs[i, j].grid(True)

plt.tight_layout()
plt.show()