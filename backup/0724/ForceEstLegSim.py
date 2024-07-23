import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

import LegKinematics


def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


enable_link_force = True
robot_data_path = f'{os.getcwd()}/data/test/output_leg.csv'
force_data_path = f'{os.getcwd()}/data/test/Force Plate.csv'

df_robot_data = pd.read_csv(robot_data_path)
df_force_data = pd.read_csv(force_data_path)

df_data =  pd.merge(df_force_data, df_robot_data.loc[:, ['time', 'phi_r', 'phi_l', 'trq_r', 'trq_l', 'dist']], on='time')
df_data.dropna(inplace=True)

t = []
f_est = []
f_meas = []
f_err = []
rim_change_times = []
last_contact_rim = ''

data_test = []

body_pos = np.array([0, 0])
body_pos_last = np.array([0, 0])
body_vel = np.array([0, 0])
body_vel_last = np.array([0, 0])
body_acc = np.array([0, 0])
dt = 0.001
body_mass = 5

for idx in range(df_data.__len__()):
    if idx < 1000: continue
    
    data = df_data.iloc[idx, :]
    
    phi = [data['phi_r'], data['phi_l']]
    trq = [data['trq_r'], data['trq_l']]
    
    theta = (phi[0]-phi[1])/2 + np.deg2rad(17)
    beta =  -(phi[0]+phi[1])/2
    
    alpha, contact_rim = LegKinematics.get_alpha(theta=theta, beta=beta)
    jacobian = LegKinematics.get_jacobian(theta=theta, beta=beta, alpha=alpha)
    
    action_force = np.linalg.inv(jacobian).T @ trq
    
    # body_pos_last = body_pos
    # body_vel_last = body_vel
    # body_pos = LegKinematics.rot_matrix(beta) @ LegKinematics.get_contact_point(alpha) * np.array([-1, -1])
    # body_pos = np.array([body_pos[0](theta), body_pos[1](theta)])
    # body_vel = (body_pos-body_pos_last)/dt
    # body_acc = (body_vel-body_vel_last)/dt
    
    # if idx < 1005: body_force = np.array([0, 0])
    # else: body_force = body_mass * body_acc
    
    body_force = np.array([0, 0])
    
    link_force = LegKinematics.get_link_force(enable=enable_link_force)
    
    total_force = action_force - body_force + link_force
    
    print(f'Theta = {round(np.rad2deg(theta), 4)}; Beta = {round(np.rad2deg(beta), 4)}; Alpha = {round(np.rad2deg(alpha), 4)}')
    
    
    t.append(data['time'])
    f_meas.append([data['force_plate_leg_fx'], data['force_plate_leg_fz']])
    f_est.append(total_force)
    
    if last_contact_rim != contact_rim: rim_change_times.append(data['time'])
    last_contact_rim = contact_rim
    
    P = LegKinematics.get_contact_point(alpha)
    P = np.array([P[0](theta), P[1](theta)])
    # P = LegKinematics.rot_matrix(beta) @ P
    data_test.append([P[0], P[1]])
    
data_test = np.array(data_test)
f_meas = np.array(f_meas)
f_est = np.array(f_est)
f_err = f_est - f_meas

# filter parameters
cutoff_frequency = 3.0
sampling_frequency = 1000.0
filter_order = 4

f_est_filtered = np.zeros_like(f_est)
f_meas_filtered = np.zeros_like(f_meas)
f_err_filtered = np.zeros_like(f_err)

f_est_filtered[:, 0] = butter_lowpass_filter(f_est[:, 0], cutoff_frequency, sampling_frequency, filter_order)
f_est_filtered[:, 1] = butter_lowpass_filter(f_est[:, 1], cutoff_frequency, sampling_frequency, filter_order)
f_meas_filtered[:, 0] = butter_lowpass_filter(f_meas[:, 0], cutoff_frequency, sampling_frequency, filter_order)
f_meas_filtered[:, 1] = butter_lowpass_filter(f_meas[:, 1], cutoff_frequency, sampling_frequency, filter_order)
f_err_filtered[:, 0] = butter_lowpass_filter(f_err[:, 0], cutoff_frequency, sampling_frequency, filter_order)
f_err_filtered[:, 1] = butter_lowpass_filter(f_err[:, 1], cutoff_frequency, sampling_frequency, filter_order)


plt.figure(figsize=(16, 12))

linewidth = 2
# plt.plot(t, f_est[:, 0], label='Estimated Force X', linestyle='-', color='blue', linewidth=linewidth)
# plt.plot(t, f_meas[:, 0], label='Measured Force X', linestyle='-', color='green', linewidth=linewidth)

plt.plot(t, f_est[:, 1], label='Estimated Force Z', linestyle='-', color='blue', linewidth=linewidth)
plt.plot(t, f_meas[:, 1], label='Measured Force Z', linestyle='-', color='green', linewidth=linewidth)

# plt.plot(t, f_est_filtered[:, 0], label='Estimated Force X', linestyle='-', color='blue', linewidth=linewidth)
# plt.plot(t, f_meas_filtered[:, 0], label='Measured Force X', linestyle='-', color='green', linewidth=linewidth)

# plt.plot(t, f_est_filtered[:, 1], label='Estimated Force Z', linestyle='-', color='blue', linewidth=linewidth)
# plt.plot(t, f_meas_filtered[:, 1], label='Measured Force Z', linestyle='-', color='green', linewidth=linewidth)

# plt.plot(t, f_err_filtered[:, 1], label='Estimated Force Z', linestyle='-', color='blue', linewidth=linewidth)
# plt.plot(t, f_err_filtered[:, 1]/f_est_filtered[:, 1]*100, label='Estimated Force Z', linestyle='-', color='blue', linewidth=linewidth)

# plt.plot(df_data.loc[:, 'time'].tolist(), df_data.loc[:, 'trq_r'].tolist(), label='trq_r', linestyle='-', color='blue', linewidth=linewidth)
# plt.plot(df_data.loc[:, 'time'].tolist(), df_data.loc[:, 'trq_l'].tolist(), label='trq_l', linestyle='-', color='green', linewidth=linewidth)

# plt.plot(df_data.loc[:, 'time'].tolist(), df_data.loc[:, 'phi_r'].tolist(), label='phi_r', linestyle='-', color='blue', linewidth=linewidth)
# plt.plot(df_data.loc[:, 'time'].tolist(), df_data.loc[:, 'phi_l'].tolist(), label='phi_l', linestyle='-', color='green', linewidth=linewidth)

# plt.plot(df_data.loc[:, 'time'].tolist(), [d  * 0.62 / 1000 + 0.055 for d in df_data.loc[:, 'dist'].tolist()], label='dist', linestyle='-', color='green', linewidth=linewidth)
# plt.plot(t, data_test[:, 0], label='test', linestyle='-', color='red', linewidth=linewidth)
# plt.plot(t, data_test[:, 1], label='test', linestyle='-', color='black', linewidth=linewidth)
# plt.plot(data_test[:, 0], data_test[:, 1], label='test', linestyle='-', color='black', linewidth=linewidth)

# for time_point in rim_change_times: plt.axvline(x=time_point, color='red', linestyle='--', linewidth=2)

plt.title('Comparison of Estimated and Measured Forces', fontsize=24)
plt.xlabel('Time (s)', fontsize=20)
plt.ylabel('Force (N)', fontsize=20)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)

plt.show()
