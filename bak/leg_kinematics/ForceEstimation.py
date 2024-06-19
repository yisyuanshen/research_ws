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


single_leg_mode = True

curr_path = os.path.dirname(os.path.realpath(__file__))+'/data/G'

if not single_leg_mode:
    robot_data = pd.read_csv(f'{curr_path}/output_robot.csv')
    force_data_1 = pd.read_csv(f'{curr_path}/Force Plate 1.csv')
    force_data_2 = pd.read_csv(f'{curr_path}/Force Plate 2.csv')
    force_data_3 = pd.read_csv(f'{curr_path}/Force Plate 3.csv')
    force_data_4 = pd.read_csv(f'{curr_path}/Force Plate 4.csv')

    data_A = pd.merge(force_data_1, robot_data.loc[:, ['Time', 'A_phi_r', 'A_phi_l', 'A_trq_r', 'A_trq_l']], on='Time')
    data_B = pd.merge(force_data_1, robot_data.loc[:, ['Time', 'B_phi_r', 'B_phi_l', 'B_trq_r', 'B_trq_l']], on='Time')
    data_C = pd.merge(force_data_1, robot_data.loc[:, ['Time', 'C_phi_r', 'C_phi_l', 'C_trq_r', 'C_trq_l']], on='Time')
    data_D = pd.merge(force_data_1, robot_data.loc[:, ['Time', 'D_phi_r', 'D_phi_l', 'D_trq_r', 'D_trq_l']], on='Time')

    data_A.dropna(inplace=True)
    data_B.dropna(inplace=True)
    data_C.dropna(inplace=True)
    data_D.dropna(inplace=True)

    data_A.rename(columns={'A_phi_r': 'phi_r', 'A_phi_l': 'phi_l', 'A_trq_r': 'trq_r', 'A_trq_l': 'trq_l'}, inplace=True)
    data_B.rename(columns={'B_phi_r': 'phi_r', 'B_phi_l': 'phi_l', 'B_trq_r': 'trq_r', 'B_trq_l': 'trq_l'}, inplace=True)
    data_C.rename(columns={'C_phi_r': 'phi_r', 'C_phi_l': 'phi_l', 'C_trq_r': 'trq_r', 'C_trq_l': 'trq_l'}, inplace=True)
    data_D.rename(columns={'D_phi_r': 'phi_r', 'D_phi_l': 'phi_l', 'D_trq_r': 'trq_r', 'D_trq_l': 'trq_l'}, inplace=True)

else:
    robot_data = pd.read_csv(f'{curr_path}/output_leg.csv')
    force_data_1 = pd.read_csv(f'{curr_path}/Force Plate.csv')
    data_A = pd.merge(force_data_1, robot_data.loc[:, ['Time', 'phi_r', 'phi_l', 'trq_r', 'trq_l']], on='Time')
    data_A.dropna(inplace=True)

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
    # print(np.linalg.inv(jacobian).T, np.array(trq))
    force_est = np.linalg.inv(jacobian).T @ np.array(trq) #- [0, 0.751427*9.81]
    force_meas = np.array([data.force_x, -data.force_z])
    
    print(force_est[1])
    
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
