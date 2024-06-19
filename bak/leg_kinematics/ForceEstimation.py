import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import time

import LegKinematics


def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


robot_data_path = os.path.dirname(os.path.realpath(__file__))+'/data/0611/robot/U_1.csv'
# force_data_path = os.path.dirname(os.path.realpath(__file__))+'/data/0611/vicon/G.csv'

df_robot_data = pd.read_csv(robot_data_path)
# df_force_data = pd.read_csv(force_data_path, dtype=str)
# df_trigger_data = pd.read_csv(force_data_path, dtype=str)

kt = 2.23

phi_list = [[data.AR_rpy_pos, data.AL_rpy_pos] for data in df_robot_data.itertuples()]
trq_list = [[data.AR_rpy_torq * kt, data.AL_rpy_torq * kt] for data in df_robot_data.itertuples()]

for i in range(df_robot_data.__len__()):
    phi = phi_list[i]
    trq = trq_list[i]
    
    theta = (phi[0] - phi[1]) / 2 + np.deg2rad(17)
    beta  = (phi[0] + phi[1]) / 2
    
    alpha, contact_rim = LegKinematics.get_alpha(theta, beta)    
    jacobian = LegKinematics.get_jacobian(theta, beta, alpha)
    
    force_est = np.linalg.inv(jacobian).T @ np.array(trq)
    
    print(force_est[1])

'''
t = []
f_est = []
f_meas = []
rim_change_times = []
last_contact_rim = ''

for data in data_A.itertuples():
    # if data.Time <= 6: continue
    # if data.Time >= 10: continue
    
    print(f'\nTime = {data.Time}')
    
    phi = [data.phi_r, data.phi_l]
    trq = [data.trq_r, data.trq_l]
    
    theta = (phi[0] - phi[1]) / 2 + np.deg2rad(17)
    beta = (phi[0] + phi[1]) / 2
    
    alpha, contact_rim = LegKinematics.get_alpha(theta, beta)    
    jacobian = LegKinematics.get_jacobian(theta, beta, alpha)
    
    force_est = np.linalg.inv(jacobian).T @ np.array(trq)
    force_meas = np.array([data.force_x, -data.force_z])
    
    t.append(data.Time)
    f_est.append(force_est)
    f_meas.append(force_meas)
    
    if last_contact_rim != contact_rim: rim_change_times.append(data.Time)
    last_contact_rim = contact_rim

t = np.array(t)
f_est = np.array(f_est)
f_meas = np.array(f_meas)
f_err = f_meas - f_est

# filter parameters
cutoff_frequency = 30.0
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
# plt.plot(t, f_est[:, 0], label='Estimated Force X', linestyle='-', color='blue', linewidth=linewidth)
# plt.plot(t, f_meas[:, 0], label='Measured Force X', linestyle='-', color='green', linewidth=linewidth)

# plt.plot(t, f_est[:, 1], label='Estimated Force Z', linestyle='-', color='blue', linewidth=linewidth)
# plt.plot(t, f_meas[:, 1], label='Measured Force Z', linestyle='-', color='green', linewidth=linewidth)

# plt.plot(t, f_est_filtered[:, 0], label='Estimated Force X', linestyle='-', color='blue', linewidth=linewidth)
# plt.plot(t, f_meas_filtered[:, 0], label='Measured Force X', linestyle='-', color='green', linewidth=linewidth)

plt.plot(t, f_est_filtered[:, 1], label='Estimated Force Z', linestyle='-', color='blue', linewidth=linewidth)
plt.plot(t, f_meas_filtered[:, 1], label='Measured Force Z', linestyle='-', color='green', linewidth=linewidth)

# plt.plot(t, f_err[:, 0], label='Error Force X', linestyle='-', color='black', linewidth=linewidth)
# plt.plot(t, f_err[:, 1], label='Error Force Z', linestyle='-', color='black', linewidth=linewidth)
# plt.plot(t, f_err_filtered[:, 0], label='Error Force X', linestyle='-', color='black', linewidth=linewidth)
# plt.plot(t, f_err_filtered[:, 1]/f_meas_filtered[:, 1]*100, label='Error Force Z', linestyle='-', color='black', linewidth=linewidth)

for time_point in rim_change_times: plt.axvline(x=time_point, color='red', linestyle='--', linewidth=2)

plt.title('Comparison of Estimated and Measured Forces', fontsize=24)
plt.xlabel('Time (s)', fontsize=20)
plt.ylabel('Force (N)', fontsize=20)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)

plt.show()
'''