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


robot_data_path = f'{os.getcwd()}/data/real/robot/motor/0722_3.csv'
force_data_path = f'{os.getcwd()}/data/real/robot/force/_0722_3.csv'

df_robot_data = pd.read_csv(robot_data_path)
df_force_data = pd.read_csv(force_data_path).iloc[8732:, :] # 3440 # 8732 # 7476

t = []
f_est = []
f_meas = []
f_err = []
data_test = []

rim_change_times_list = [[], [], [], []]
last_contact_rim_list = ['', '', '', '']

for idx in range(min(df_force_data.__len__(), df_robot_data.__len__())):
# for idx in range(df_robot_data.__len__()//4):
    robot_data = df_robot_data.iloc[idx, :]
    force_data = df_force_data.iloc[idx, :]
    
    kt = 2.2
    phi_list = [[robot_data['AR_rpy_pos'], robot_data['AL_rpy_pos']], [robot_data['BR_rpy_pos'], robot_data['BL_rpy_pos']],
                [robot_data['CR_rpy_pos'], robot_data['CL_rpy_pos']], [robot_data['DR_rpy_pos'], robot_data['DL_rpy_pos']]]
    
    trq_list = [[robot_data['AR_rpy_torq']*kt, robot_data['AL_rpy_torq']*kt], [robot_data['BR_rpy_torq']*kt, robot_data['BL_rpy_torq']*kt],
                [robot_data['CR_rpy_torq']*kt, robot_data['CL_rpy_torq']*kt], [robot_data['DR_rpy_torq']*kt, robot_data['DL_rpy_torq']*kt]]
    
    # theta = (phi[0]-phi[1])/2 + np.deg2rad(17)
    # beta =  -(phi[0]+phi[1])/2
    
    # alpha, contact_rim = LegKinematics.get_alpha(theta=theta, beta=beta)
    # jacobian = LegKinematics.get_jacobian(theta=theta, beta=beta, alpha=alpha)
    
    # action_force = np.linalg.inv(jacobian).T @ trq
    # link_force = LegKinematics.get_link_force(enable=False)
    # total_force = action_force + link_force
    
    # t.append(idx)
    # f_est.append(total_force)
    # f_meas.append([0, -force_data['Fz_2']])
    
    # data_test.append(trq)
    
    force_list = []
    for i in range(4):
        theta = (phi_list[i][0]-phi_list[i][1])/2 + np.deg2rad(17)
        beta =  -(phi_list[i][0]+phi_list[i][1])/2
        alpha, contact_rim = LegKinematics.get_alpha(theta=theta, beta=beta)
        
        print(f'{i}. Theta = {round(np.rad2deg(theta), 4)}; Beta = {round(np.rad2deg(beta), 4)}; Alpha = {round(np.rad2deg(alpha), 4)}')
        
        jacobian = LegKinematics.get_jacobian(theta=theta, beta=beta, alpha=alpha)
    
        action_force = np.linalg.inv(jacobian).T @ trq_list[i]
        link_force = LegKinematics.get_link_force(1)

        force_list.append(action_force+link_force)
        
        if last_contact_rim_list[i] != contact_rim: rim_change_times_list[i].append(idx)
        last_contact_rim_list[i] = contact_rim
        
        P = LegKinematics.get_contact_point(alpha)
        P_deriv = np.array([P[0].deriv(), P[1].deriv()])
        
        P = [P[0](theta), P[1](theta)]
        P_deriv = [P_deriv[0](theta), P_deriv[1](theta)]
    
    data_test.append(trq_list)
    
    t.append(idx)
    f_est.append(force_list)
    f_meas.append([[force_data['Fx_1'], -force_data['Fz_1']],
                   [force_data['Fx_4'], -force_data['Fz_4']],
                   [force_data['Fx_3'], -force_data['Fz_3']],
                   [force_data['Fx_2'], -force_data['Fz_2']]])
    
t = np.array(t)
f_est = np.array(f_est)
f_meas = np.array(f_meas)
data_test = np.array(data_test)

# filter parameters
cutoff_frequency = 5.0
sampling_frequency = 1000.0
filter_order = 4

f_est_filtered = np.zeros_like(f_est)
f_meas_filtered = np.zeros_like(f_meas)
# f_err_filtered = np.zeros_like(f_err)

for i in range(4):
    f_est_filtered[:, i, 0] = butter_lowpass_filter(f_est[:, i, 0], cutoff_frequency, sampling_frequency, filter_order)
    f_est_filtered[:, i, 1] = butter_lowpass_filter(f_est[:, i, 1], cutoff_frequency, sampling_frequency, filter_order)
    f_meas_filtered[:, i, 0] = butter_lowpass_filter(f_meas[:, i, 0], cutoff_frequency, sampling_frequency, filter_order)
    f_meas_filtered[:, i, 1] = butter_lowpass_filter(f_meas[:, i, 1], cutoff_frequency, sampling_frequency, filter_order)
    # f_err_filtered[:, i, 0] = butter_lowpass_filter(f_err[:, i, 0], cutoff_frequency, sampling_frequency, filter_order)
    # f_err_filtered[:, i, 1] = butter_lowpass_filter(f_err[:, i, 1], cutoff_frequency, sampling_frequency, filter_order)

plt.figure(figsize=(16, 12))
linewidth = 1

# '''
# plt.plot(t, f_est[:, 0, 0], label='Estimated Force X', linestyle='-', color='blue', linewidth=linewidth)
# plt.plot(t, f_meas[:, 0, 0], label='Measured Force X', linestyle='-', color='green', linewidth=linewidth)

# plt.plot(t, f_est[:, 0, 1], label='Estimated Force Z', linestyle='-', color='blue', linewidth=linewidth)
# plt.plot(t, f_meas[:, 0, 1], label='Measured Force Z', linestyle='-', color='green', linewidth=linewidth)

# plt.plot(t, f_est_filtered[:, 0, 0], label='Estimated Force X', linestyle='-', color='blue', linewidth=linewidth)
# plt.plot(t, f_meas_filtered[:, 0, 0], label='Measured Force X', linestyle='-', color='green', linewidth=linewidth)

plt.plot(t, f_est_filtered[:, 0, 1], label='Estimated Force Z', linestyle='-', color='blue', linewidth=linewidth)
plt.plot(t, f_meas_filtered[:, 0, 1], label='Measured Force Z', linestyle='-', color='green', linewidth=linewidth)

# plt.plot(t, f_err_filtered[:, 0, 1], label='Estimated Force Z', linestyle='-', color='blue', linewidth=linewidth)
# plt.plot(t, f_err_filtered[:, 0, 1]/f_est_filtered[:, 0, 1]*100, label='Estimated Force Z', linestyle='-', color='blue', linewidth=linewidth)

# plt.plot(df_data.loc[:, 'time'].tolist(), df_data.loc[:, 'A_trq_r'].tolist(), label='trq_r', linestyle='-', color='blue', linewidth=linewidth)
# plt.plot(df_data.loc[:, 'time'].tolist(), df_data.loc[:, 'A_trq_l'].tolist(), label='trq_l', linestyle='-', color='green', linewidth=linewidth)

# plt.plot(df_data.loc[:, 'time'].tolist(), df_data.loc[:, 0, 'phi_r'].tolist(), label='phi_r', linestyle='-', color='blue', linewidth=linewidth)
# plt.plot(df_data.loc[:, 'time'].tolist(), df_data.loc[:, 0, 'phi_l'].tolist(), label='phi_l', linestyle='-', color='green', linewidth=linewidth)

# plt.plot(t, data_test[:, 0, 0], label=f'Data x', linestyle='-', color='blue', linewidth=linewidth)
# plt.plot(t, data_test[:, 0, 1], label=f'Data y', linestyle='-', color='green', linewidth=linewidth)

# plt.plot(t, data_test, label='test', linestyle='-', color='black', linewidth=linewidth)

for time_point in rim_change_times_list[0]: plt.axvline(x=time_point, color='red', linestyle='--', linewidth=2)

plt.title('Comparison of Estimated and Measured Forces', fontsize=24)
plt.xlabel('Time (s)', fontsize=20)
plt.ylabel('Force (N)', fontsize=20)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(range(0, 101, 5), fontsize=20)
plt.grid(True)

plt.show()


for i in range(4):
    plt.subplot(2, 2, i + 1)
    # plt.plot(t, f_est[:, i, 0], label=f'Estimated Force X {i}', linestyle='-', color='blue', linewidth=linewidth)
    # plt.plot(t, f_meas[:, i, 0], label=f'Measured Force X {i}', linestyle='-', color='green', linewidth=linewidth)

    # plt.plot(t, f_est[:, i, 1], label=f'Estimated Force Z {i}', linestyle='-', color='blue', linewidth=linewidth)
    # plt.plot(t, f_meas[:, i, 1], label=f'Measured Force Z {i}', linestyle='-', color='green', linewidth=linewidth)

    # plt.plot(t, f_est_filtered[:, i, 0], label=f'Estimated Force X {i}', linestyle='-', color='blue', linewidth=linewidth)
    # plt.plot(t, f_meas_filtered[:, i, 0], label=f'Measured Force X {i}', linestyle='-', color='green', linewidth=linewidth)

    plt.plot(t, f_est_filtered[:, i, 1], label=f'Estimated Force Z {i}', linestyle='-', color='blue', linewidth=linewidth)
    plt.plot(t, f_meas_filtered[:, i, 1], label=f'Measured Force Z {i}', linestyle='-', color='green', linewidth=linewidth)

    # plt.plot(t, f_err_filtered[:, i, 1], label=f'Estimated Force Z {i}', linestyle='-', color='blue', linewidth=linewidth)
    # plt.plot(t, f_err_filtered[:, i, 1]/f_est_filtered[:, i, 1]*100, label=f'Estimated Force Z {i}', linestyle='-', color='blue', linewidth=linewidth)

    # plt.plot(t, data_test[:, i, 0], label=f'Data x', linestyle='-', color='blue', linewidth=linewidth)
    # plt.plot(t, data_test[:, i, 1], label=f'Data y', linestyle='-', color='green', linewidth=linewidth)

    for time_point in rim_change_times_list[i]: plt.axvline(x=time_point, color='red', linestyle='--', linewidth=2)
    
    plt.title(f'Comparison of Estimated and Measured Forces {i}', fontsize=24)
    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('Force (N)', fontsize=20)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(range(0, 101, 5), fontsize=20)
    plt.grid(True)

plt.tight_layout()
plt.show()
