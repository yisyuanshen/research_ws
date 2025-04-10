import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ViconProcess

df = pd.read_csv("data/0407_mpc/0407_walk_2.csv")
# df = pd.read_csv("data/0407_mpc/0407_MPC_1.csv")
df_odom = pd.read_csv("data/0407_mpc/0407_walk_2_odom.csv")

df_vicon, trigger_idx = None, None
df_vicon, trigger_idx = ViconProcess.read_csv("data/0407_mpc/vicon/walk_2.csv")
# df_vicon, trigger_idx = ViconProcess.read_csv("data/0407_mpc/vicon/MPC_1.csv")

start_idx = 0
end_idx = 38000
# end_idx = 32000
vicon_offset = 0
force = False

time = df['Time'][start_idx:end_idx]
sim_pos_x = df['sim_pos_x'][start_idx:end_idx] - df['sim_pos_x'][start_idx]
sim_pos_z = df['sim_pos_z'][start_idx:end_idx] - df['sim_pos_z'][start_idx]

# odom_pos_x = df['odom_pos_x'][start_idx:end_idx]
# odom_pos_z = df['odom_pos_z'][start_idx:end_idx]
# odom_vel_x = df['odom_vel_x'][start_idx:end_idx]
# odom_vel_z = df['odom_vel_z'][start_idx:end_idx]

odom_pos_x = np.repeat(df_odom['p.x'][start_idx//5:end_idx//5].values, 5)
odom_pos_z = np.repeat(df_odom['p.z'][start_idx//5:end_idx//5].values, 5)
odom_vel_x = np.repeat(df_odom['v_.x'][start_idx//5:end_idx//5].values, 5)
odom_vel_z = np.repeat(df_odom['v_.z'][start_idx//5:end_idx//5].values, 5)


force_state_x = [df['force_Fx_a'][start_idx:end_idx], df['force_Fx_b'][start_idx:end_idx], df['force_Fx_c'][start_idx:end_idx], df['force_Fx_d'][start_idx:end_idx]]
force_state_z = [df['force_Fy_a'][start_idx:end_idx], df['force_Fy_b'][start_idx:end_idx], df['force_Fy_c'][start_idx:end_idx], df['force_Fy_d'][start_idx:end_idx]]
force_cmd_x = [df['imp_cmd_Fx_a'][start_idx:end_idx], df['imp_cmd_Fx_b'][start_idx:end_idx], df['imp_cmd_Fx_c'][start_idx:end_idx], df['imp_cmd_Fx_d'][start_idx:end_idx]]
force_cmd_z = [df['imp_cmd_Fy_a'][start_idx:end_idx], df['imp_cmd_Fy_b'][start_idx:end_idx], df['imp_cmd_Fy_c'][start_idx:end_idx], df['imp_cmd_Fy_d'][start_idx:end_idx]]

imu_qx = df['imu_orien_x'][start_idx:end_idx]
imu_qy = df['imu_orien_y'][start_idx:end_idx]
imu_qz = df['imu_orien_z'][start_idx:end_idx]
imu_qw = df['imu_orien_w'][start_idx:end_idx]
imu_roll = np.arctan2(2 * (imu_qw * imu_qx + imu_qy * imu_qz), 1 - 2 * (imu_qx**2 + imu_qy**2))
imu_pitch = np.arcsin(2 * (imu_qw * imu_qy - imu_qz * imu_qx))
imu_roll = np.rad2deg(imu_roll)
imu_pitch = np.rad2deg(imu_pitch)

if df_vicon is not None:
    vicon_pos_x = df_vicon['vicon_pos_x'][trigger_idx+vicon_offset+start_idx:trigger_idx+vicon_offset+end_idx]
    vicon_pos_z = df_vicon['vicon_pos_z'][trigger_idx+vicon_offset+start_idx:trigger_idx+vicon_offset+end_idx]
    
    vicon_pos_x -= vicon_pos_x.iloc[0]
    vicon_pos_z -= vicon_pos_z.iloc[0]

    vicon_roll = df_vicon['vicon_roll'][trigger_idx+vicon_offset+start_idx:trigger_idx+vicon_offset+end_idx]
    vicon_pitch = df_vicon['vicon_pitch'][trigger_idx+vicon_offset+start_idx:trigger_idx+vicon_offset+end_idx]
    
cmd_pos_x = []
v = 0
for t in range(len(time)+start_idx):
    if t < 5000:
        cmd_pos_x.append(0)
    elif t < 8000:
        v += 0.1/3000
        cmd_pos_x.append(cmd_pos_x[-1]+v*0.001)
    # elif t < 27000:
    elif t < 32000:
        cmd_pos_x.append(cmd_pos_x[-1]+v*0.001)
    # elif t < 30000:
    elif t < 35000:
        v -= 0.1/3000
        cmd_pos_x.append(cmd_pos_x[-1]+v*0.001)
    else:
        cmd_pos_x.append(cmd_pos_x[-1])

cmd_pos_x = cmd_pos_x[start_idx:]


title_fontsize = 16
label_fontsize = 12
tick_fontsize = 12
legend_fontsize = 12

if force: fig, axs = plt.subplots(4, 2, figsize=(12, 8))
else: fig, axs = plt.subplots(3, 2, figsize=(12, 8))

if df_vicon is not None: axs[0, 0].plot(range(len(time)), vicon_pos_x, label='Vicon Pos X', color='C0')
else: axs[0, 0].plot(range(len(time)), sim_pos_x, label='Sim Pos X', color='C0')
axs[0, 0].plot(range(len(time)), odom_pos_x, label='Odom Pos X', color='C1', linestyle='--')
axs[0, 0].plot(range(len(time)), cmd_pos_x, label='Cmd Pos X', color='C2', linestyle='-.')
axs[0, 0].set_ylabel('X (m)', fontsize=label_fontsize)
axs[0, 0].set_title('Position X', fontsize=title_fontsize)
axs[0, 0].set_ylim([-0.5, 3])

if df_vicon is not None: axs[0, 1].plot(range(len(time)), vicon_pos_z, label='Vicon Pos Z', color='C0')
else: axs[0, 1].plot(range(len(time)), sim_pos_z, label='Sim Pos Z', color='C0')
axs[0, 1].plot(range(len(time)), odom_pos_z, label='Odom Pos Z', color='C1', linestyle='--')
axs[0, 1].set_ylabel('Z (m)', fontsize=label_fontsize)
axs[0, 1].set_title('Position Z', fontsize=title_fontsize)
axs[0, 1].set_ylim([-0.05, 0.05])

axs[1, 0].plot(range(len(time)), odom_vel_x, label='Odom Vel X', color='C1')
axs[1, 0].set_ylabel('Vx (m/s)', fontsize=label_fontsize)
axs[1, 0].set_title('Velocity X', fontsize=title_fontsize)
axs[1, 0].set_ylim([-0.2, 0.4])

axs[1, 1].plot(range(len(time)), odom_vel_z, label='Odom Vel Z', color='C1')
axs[1, 1].set_ylabel('Vy (m/s)', fontsize=label_fontsize)
axs[1, 1].set_title('Velocity Z', fontsize=title_fontsize)
axs[1, 1].set_ylim([-0.3, 0.3])

if df_vicon is not None: axs[2, 0].plot(range(len(time)), vicon_pitch, label='Vicon Pitch', color='C0')
axs[2, 0].plot(range(len(time)), -imu_pitch, label='IMU Pitch', color='C1', linestyle='--')
axs[2, 0].set_ylabel('Angle (deg)', fontsize=label_fontsize)
axs[2, 0].set_title('Pitch', fontsize=title_fontsize)
axs[2, 0].set_ylim([-10, 10])

if df_vicon is not None: axs[2, 1].plot(range(len(time)), vicon_roll, label='Vicon Roll', color='C0')
axs[2, 1].plot(range(len(time)), imu_roll, label='IMU Roll', color='C1', linestyle='--')
axs[2, 1].set_ylabel('Angle (deg)', fontsize=label_fontsize)
axs[2, 1].set_title('Roll', fontsize=title_fontsize)
axs[2, 1].set_ylim([-10, 10])

if force:
    axs[3, 0].plot(range(len(time)), force_cmd_x[0], label='Force Cmd X - A', color='C0')
    axs[3, 0].plot(range(len(time)), force_cmd_x[1], label='Force Cmd X - B', color='C1')
    axs[3, 0].plot(range(len(time)), force_cmd_x[2], label='Force Cmd X - C', color='C2')
    axs[3, 0].plot(range(len(time)), force_cmd_x[3], label='Force Cmd X - D', color='C3')
    axs[3, 0].set_ylabel('Force (N)', fontsize=label_fontsize)
    axs[3, 0].set_title('Force X', fontsize=title_fontsize)

    axs[3, 1].plot(range(len(time)), force_cmd_z[0], label='Force Cmd Z - A', color='C0')
    axs[3, 1].plot(range(len(time)), force_cmd_z[1], label='Force Cmd Z - B', color='C1')
    axs[3, 1].plot(range(len(time)), force_cmd_z[2], label='Force Cmd Z - C', color='C2')
    axs[3, 1].plot(range(len(time)), force_cmd_z[3], label='Force Cmd Z - D', color='C3')
    axs[3, 1].set_ylabel('Force (N)', fontsize=label_fontsize)
    axs[3, 1].set_title('Force Z', fontsize=title_fontsize)

    # axs[3, 0].plot(range(len(time)), force_state_x[0], label='Force State X - A', color='C0')
    # axs[3, 0].plot(range(len(time)), force_state_x[1], label='Force State X - B', color='C1')
    # axs[3, 0].plot(range(len(time)), force_state_x[2], label='Force State X - C', color='C2')
    # axs[3, 0].plot(range(len(time)), force_state_x[3], label='Force State X - D', color='C3')
    # axs[3, 0].set_ylabel('Force (N)', fontsize=label_fontsize)
    # axs[3, 0].set_title('Force X', fontsize=title_fontsize)

    # axs[3, 1].plot(range(len(time)), force_state_z[0], label='Force State Z - A', color='C0')
    # axs[3, 1].plot(range(len(time)), force_state_z[1], label='Force State Z - B', color='C1')
    # axs[3, 1].plot(range(len(time)), force_state_z[2], label='Force State Z - C', color='C2')
    # axs[3, 1].plot(range(len(time)), force_state_z[3], label='Force State Z - D', color='C3')
    # axs[3, 1].set_ylabel('Force (N)', fontsize=label_fontsize)
    # axs[3, 1].set_title('Force Z', fontsize=title_fontsize)

for i in range(4 if force else 3):
    for j in range(2):
        axs[i, j].set_xlabel('Time (ms)', fontsize=label_fontsize)
        axs[i, j].tick_params(axis='both', labelsize=tick_fontsize)
        axs[i, j].legend(fontsize=legend_fontsize)
        axs[i, j].grid(True)

plt.tight_layout()
plt.show()