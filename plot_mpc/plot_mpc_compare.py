import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ViconProcess

df_1 = pd.read_csv("data/0407_mpc/0407_walk_2.csv")
df_2 = pd.read_csv("data/0407_mpc/0407_MPC_1.csv")

df_vicon_1, trigger_idx_1 = None, None
df_vicon_1, trigger_idx_1 = ViconProcess.read_csv("data/0407_mpc/vicon/walk_2.csv")
df_vicon_2, trigger_idx_2 = None, None
df_vicon_2, trigger_idx_2 = ViconProcess.read_csv("data/0407_mpc/vicon/MPC_1.csv")

start_idx = 15000
# end_idx = 38000
end_idx = 25000
vicon_offset = 0
time = end_idx-start_idx

odom_pos_x_1 = df_1['odom_pos_x'][start_idx:end_idx]
odom_pos_z_1 = df_1['odom_pos_z'][start_idx:end_idx]
odom_vel_x_1 = df_1['odom_vel_x'][start_idx:end_idx]
odom_vel_z_1 = df_1['odom_vel_z'][start_idx:end_idx]

odom_pos_x_2 = df_2['odom_pos_x'][start_idx:end_idx]
odom_pos_z_2 = df_2['odom_pos_z'][start_idx:end_idx]
odom_vel_x_2 = df_2['odom_vel_x'][start_idx:end_idx]
odom_vel_z_2 = df_2['odom_vel_z'][start_idx:end_idx]

imu_qx = df_1['imu_orien_x'][start_idx:end_idx]
imu_qy = df_1['imu_orien_y'][start_idx:end_idx]
imu_qz = df_1['imu_orien_z'][start_idx:end_idx]
imu_qw = df_1['imu_orien_w'][start_idx:end_idx]
imu_roll = np.arctan2(2 * (imu_qw * imu_qx + imu_qy * imu_qz), 1 - 2 * (imu_qx**2 + imu_qy**2))
imu_pitch = np.arcsin(2 * (imu_qw * imu_qy - imu_qz * imu_qx))
imu_roll_1 = np.rad2deg(imu_roll)
imu_pitch_1 = -np.rad2deg(imu_pitch)

imu_qx = df_2['imu_orien_x'][start_idx:end_idx]
imu_qy = df_2['imu_orien_y'][start_idx:end_idx]
imu_qz = df_2['imu_orien_z'][start_idx:end_idx]
imu_qw = df_2['imu_orien_w'][start_idx:end_idx]
imu_roll = np.arctan2(2 * (imu_qw * imu_qx + imu_qy * imu_qz), 1 - 2 * (imu_qx**2 + imu_qy**2))
imu_pitch = np.arcsin(2 * (imu_qw * imu_qy - imu_qz * imu_qx))
imu_roll_2 = np.rad2deg(imu_roll)
imu_pitch_2 = -np.rad2deg(imu_pitch)

if df_vicon_1 is not None:
    vicon_pos_x_1 = df_vicon_1['vicon_pos_x'][trigger_idx_1+vicon_offset+start_idx:trigger_idx_1+vicon_offset+end_idx]
    vicon_pos_z_1 = df_vicon_1['vicon_pos_z'][trigger_idx_1+vicon_offset+start_idx:trigger_idx_1+vicon_offset+end_idx]
    
    vicon_pos_x_1 -= vicon_pos_x_1.iloc[0]
    # vicon_pos_z_1 -= vicon_pos_z_1.iloc[0]

    vicon_roll_1 = df_vicon_1['vicon_roll'][trigger_idx_1+vicon_offset+start_idx:trigger_idx_1+vicon_offset+end_idx]
    vicon_pitch_1 = df_vicon_1['vicon_pitch'][trigger_idx_1+vicon_offset+start_idx:trigger_idx_1+vicon_offset+end_idx]

if df_vicon_2 is not None:
    vicon_pos_x_2 = df_vicon_2['vicon_pos_x'][trigger_idx_2+vicon_offset+start_idx:trigger_idx_2+vicon_offset+end_idx]
    vicon_pos_z_2 = df_vicon_2['vicon_pos_z'][trigger_idx_2+vicon_offset+start_idx:trigger_idx_2+vicon_offset+end_idx]
    
    vicon_pos_x_2 -= vicon_pos_x_2.iloc[0]
    # vicon_pos_z_2 -= vicon_pos_z_2.iloc[0]

    vicon_roll_2 = df_vicon_2['vicon_roll'][trigger_idx_2+vicon_offset+start_idx:trigger_idx_2+vicon_offset+end_idx]
    vicon_pitch_2 = df_vicon_2['vicon_pitch'][trigger_idx_2+vicon_offset+start_idx:trigger_idx_2+vicon_offset+end_idx]


title_fontsize = 16
label_fontsize = 12
tick_fontsize = 12
legend_fontsize = 12

fig, axs = plt.subplots(3, 2, figsize=(12, 8))

if df_vicon_1 is not None: axs[0, 0].plot(range(time), vicon_pos_x_1, label='Open Loop', color='C0')
if df_vicon_2 is not None: axs[0, 0].plot(range(time), vicon_pos_x_2, label='Closed Loop', color='C1')
# axs[0, 0].plot(range(end_idx-start_idx), odom_pos_x_1, label='Open Loop', color='C0')
# axs[0, 0].plot(range(end_idx-start_idx), odom_pos_x_2, label='Closed Loop', color='C1')
axs[0, 0].set_ylabel('X (m)', fontsize=label_fontsize)
axs[0, 0].set_title('Position X', fontsize=title_fontsize)
axs[0, 0].set_ylim([-0.5, 3])

if df_vicon_1 is not None: axs[0, 1].plot(range(time), vicon_pos_z_1, label='Open Loop', color='C0')
if df_vicon_2 is not None: axs[0, 1].plot(range(time), vicon_pos_z_2, label='Closed Loop', color='C1')
# axs[0, 1].plot(range(end_idx-start_idx), odom_pos_z_1, label='Open Loop', color='C0')
# axs[0, 1].plot(range(end_idx-start_idx), odom_pos_z_2, label='Closed Loop', color='C1')
axs[0, 1].set_ylabel('Z (m)', fontsize=label_fontsize)
axs[0, 1].set_title('Position Z', fontsize=title_fontsize)
# axs[0, 1].set_ylim([-0.05, 0.05])

axs[1, 0].plot(range(time), odom_vel_x_1, label='Open Loop', color='C0')
axs[1, 0].plot(range(time), odom_vel_x_2, label='Closed Loop', color='C1')
axs[1, 0].set_ylabel('Vx (m/s)', fontsize=label_fontsize)
axs[1, 0].set_title('Velocity X', fontsize=title_fontsize)
axs[1, 0].set_ylim([-0.2, 0.4])

axs[1, 1].plot(range(time), odom_vel_z_1, label='Open Loop', color='C0')
axs[1, 1].plot(range(time), odom_vel_z_2, label='Closed Loop', color='C1')
axs[1, 1].set_title('Velocity Z', fontsize=title_fontsize)
axs[1, 1].set_ylim([-0.3, 0.3])

if df_vicon_1 is not None: axs[2, 0].plot(range(time), vicon_pitch_1, label='Open Loop', color='C0')
if df_vicon_2 is not None: axs[2, 0].plot(range(time), vicon_pitch_2, label='Closed Loop', color='C1')
# axs[2, 0].plot(range(time), imu_pitch_1, label='Open Loop', color='C0')
# axs[2, 0].plot(range(time), imu_pitch_2, label='Closed Loop', color='C1')
axs[2, 0].set_ylabel('Angle (deg)', fontsize=label_fontsize)
axs[2, 0].set_title('Pitch', fontsize=title_fontsize)
axs[2, 0].set_ylim([-10, 10])

if df_vicon_1 is not None: axs[2, 1].plot(range(time), vicon_roll_1, label='Open Loop', color='C0')
if df_vicon_2 is not None: axs[2, 1].plot(range(time), vicon_roll_2, label='Closed Loop', color='C1')
# axs[2, 1].plot(range(time), imu_roll_1, label='Open Loop', color='C0')
# axs[2, 1].plot(range(time), imu_roll_2, label='Closed Loop', color='C1')
axs[2, 1].set_ylabel('Angle (deg)', fontsize=label_fontsize)
axs[2, 1].set_title('Roll', fontsize=title_fontsize)
axs[2, 1].set_ylim([-10, 10])

for i in range(3):
    for j in range(2):
        axs[i, j].set_xlabel('Time (ms)', fontsize=label_fontsize)
        axs[i, j].tick_params(axis='both', labelsize=tick_fontsize)
        axs[i, j].legend(fontsize=legend_fontsize)
        axs[i, j].grid(True)

plt.tight_layout()
plt.show()