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


robot_data_path = f'{os.getcwd()}/data/test_/output_leg.csv'
force_data_path = f'{os.getcwd()}/data/test_/Force Plate.csv'

df_robot_data = pd.read_csv(robot_data_path)
df_force_data = pd.read_csv(force_data_path)

df_data =  pd.merge(df_force_data, df_robot_data.loc[:, ['time', 'phi_r', 'phi_l', 'trq_r', 'trq_l']], on='time')
df_data.dropna(inplace=True)

t = []
f_est = []
f_meas = []
f_err = []
rim_change_times = []
last_contact_rim = ''

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
    
    print(f'Theta = {round(np.rad2deg(theta), 4)}; Beta = {round(np.rad2deg(beta), 4)}; Alpha = {round(np.rad2deg(alpha), 4)}')
    
    t.append(data['time'])
    f_meas.append([data['force_x'], data['force_z']])
    f_est.append(action_force)
    
    if last_contact_rim != contact_rim: rim_change_times.append(data['time'])
    last_contact_rim = contact_rim
    
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

linewidth = 1
# plt.plot(t, f_est[:, 0], label='Estimated Force Z', linestyle='-', color='blue', linewidth=linewidth)
# plt.plot(t, f_meas[:, 0], label='Measured Force Z', linestyle='-', color='green', linewidth=linewidth)

# plt.plot(t, f_est[:, 1], label='Estimated Force Z', linestyle='-', color='blue', linewidth=linewidth)
# plt.plot(t, f_meas[:, 1], label='Measured Force Z', linestyle='-', color='green', linewidth=linewidth)

# plt.plot(t, f_est_filtered[:, 0], label='Estimated Force X', linestyle='-', color='blue', linewidth=linewidth)
# plt.plot(t, f_meas_filtered[:, 0], label='Measured Force X', linestyle='-', color='green', linewidth=linewidth)

plt.plot(t, f_est_filtered[:, 1], label='Estimated Force Z', linestyle='-', color='blue', linewidth=linewidth)
plt.plot(t, f_meas_filtered[:, 1], label='Measured Force Z', linestyle='-', color='green', linewidth=linewidth)

# plt.plot(t, f_err_filtered[:, 1], label='Estimated Force Z', linestyle='-', color='blue', linewidth=linewidth)
# plt.plot(t, f_err_filtered[:, 1]/f_est_filtered[:, 1]*100, label='Estimated Force Z', linestyle='-', color='blue', linewidth=linewidth)

# plt.plot(df_data.loc[:, 'time'].tolist(), df_data.loc[:, 'trq_r'].tolist(), label='trq_r', linestyle='-', color='blue', linewidth=linewidth)
# plt.plot(df_data.loc[:, 'time'].tolist(), df_data.loc[:, 'trq_l'].tolist(), label='trq_l', linestyle='-', color='green', linewidth=linewidth)

# plt.plot(df_data.loc[:, 'time'].tolist(), df_data.loc[:, 'phi_r'].tolist(), label='phi_r', linestyle='-', color='blue', linewidth=linewidth)
# plt.plot(df_data.loc[:, 'time'].tolist(), df_data.loc[:, 'phi_l'].tolist(), label='phi_l', linestyle='-', color='green', linewidth=linewidth)

for time_point in rim_change_times: plt.axvline(x=time_point, color='red', linestyle='--', linewidth=2)

plt.title('Comparison of Estimated and Measured Forces', fontsize=24)
plt.xlabel('Time (s)', fontsize=20)
plt.ylabel('Force (N)', fontsize=20)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)

plt.show()