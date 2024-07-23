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


robot_data_path = f'{os.getcwd()}/data/real/leg/motor/t120b30.csv'
force_data_path = f'{os.getcwd()}/data/real/leg/force/_t120b30.csv'

df_robot_data = pd.read_csv(robot_data_path)
df_force_data = pd.read_csv(force_data_path)

t = []
f_est = []
f_meas = []
f_err = []
data_test = []


for idx in range(min(df_force_data.__len__(), df_robot_data.__len__()//4)):
    robot_data = df_robot_data.iloc[idx*4, :]
    force_data = df_force_data.iloc[idx, :]
    
    kt = 2.2
    phi = [robot_data['BR_rpy_pos'], robot_data['BL_rpy_pos']]
    trq = [robot_data['BR_rpy_torq']*kt, robot_data['BL_rpy_torq']*kt]
    print(trq)
    theta = (phi[0]-phi[1])/2 + np.deg2rad(17)
    beta =  -(phi[0]+phi[1])/2
    
    alpha, contact_rim = LegKinematics.get_alpha(theta=theta, beta=beta)
    jacobian = LegKinematics.get_jacobian(theta=theta, beta=beta, alpha=alpha)
    
    action_force = np.linalg.inv(jacobian).T @ trq
    link_force = LegKinematics.get_link_force(enable=False)
    total_force = action_force + link_force
    
    # print(total_force)
    
    t.append(idx)
    f_est.append(total_force)
    f_meas.append([0, -force_data['Fz_2']])
    
    data_test.append(trq)
    
t = np.array(t)
f_est = np.array(f_est)
f_meas = np.array(f_meas)
data_test = np.array(data_test)

# filter parameters
cutoff_frequency = 3.0
sampling_frequency = 250.0
filter_order = 4

f_est_filtered = np.zeros_like(f_est)
f_meas_filtered = np.zeros_like(f_meas)
# f_err_filtered = np.zeros_like(f_err)

f_est_filtered[:, 0] = butter_lowpass_filter(f_est[:, 0], cutoff_frequency, sampling_frequency, filter_order)
f_est_filtered[:, 1] = butter_lowpass_filter(f_est[:, 1], cutoff_frequency, sampling_frequency, filter_order)
f_meas_filtered[:, 0] = butter_lowpass_filter(f_meas[:, 0], cutoff_frequency, sampling_frequency, filter_order)
f_meas_filtered[:, 1] = butter_lowpass_filter(f_meas[:, 1], cutoff_frequency, sampling_frequency, filter_order)
# f_err_filtered[:, 0] = butter_lowpass_filter(f_err[:, 0], cutoff_frequency, sampling_frequency, filter_order)
# f_err_filtered[:, 1] = butter_lowpass_filter(f_err[:, 1], cutoff_frequency, sampling_frequency, filter_order)
data_test[:, 0] = butter_lowpass_filter(data_test[:, 0], cutoff_frequency, sampling_frequency, filter_order)
data_test[:, 1] = butter_lowpass_filter(data_test[:, 1], cutoff_frequency, sampling_frequency, filter_order)

plt.figure(figsize=(16, 12))
linewidth = 2

# plt.plot(t, f_est[:, 0], label='Estimated Force X', linestyle='-', color='blue', linewidth=linewidth)
# plt.plot(t, f_meas[:, 0], label='Measured Force X', linestyle='-', color='green', linewidth=linewidth)

# plt.plot(t, f_est[:, 1], label='Estimated Force Z', linestyle='-', color='blue', linewidth=linewidth)
# plt.plot(t, f_meas[:, 1], label='Measured Force Z', linestyle='-', color='green', linewidth=linewidth)

# plt.plot(t, f_est_filtered[:, 0], label='Estimated Force X', linestyle='-', color='blue', linewidth=linewidth)
# plt.plot(t, f_meas_filtered[:, 0], label='Measured Force X', linestyle='-', color='green', linewidth=linewidth)

plt.plot(t, f_est_filtered[:, 1], label='Estimated Force Z', linestyle='-', color='blue', linewidth=linewidth)
plt.plot(t, f_meas_filtered[:, 1], label='Measured Force Z', linestyle='-', color='green', linewidth=linewidth)

# plt.plot(t, data_test[:, 0], label='0', linestyle='-', color='red', linewidth=linewidth)
# plt.plot(t, data_test[:, 1], label='1', linestyle='-', color='black', linewidth=linewidth)

plt.title('Comparison of Estimated and Measured Forces', fontsize=24)
plt.xlabel('Time (s)', fontsize=20)
plt.ylabel('Force (N)', fontsize=20)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)

plt.show()
