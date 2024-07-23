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

plot_all = False
enable_link_force = True
robot_data_path = f'{os.getcwd()}/data/test/robot.csv'
force_data_path = f'{os.getcwd()}/data/test/Force Plate.csv'


df_robot_data = pd.read_csv(robot_data_path)
df_force_data = pd.read_csv(force_data_path)

df_data =  pd.merge(df_force_data, df_robot_data.loc[:, ['time',
                                                         'A_phi_r', 'A_phi_l', 'A_trq_r', 'A_trq_l',
                                                         'B_phi_r', 'B_phi_l', 'B_trq_r', 'B_trq_l',
                                                         'C_phi_r', 'C_phi_l', 'C_trq_r', 'C_trq_l',
                                                         'D_phi_r', 'D_phi_l', 'D_trq_r', 'D_trq_l',
                                                         'acc_x', 'acc_y', 'acc_z']], on='time')
df_data.dropna(inplace=True)

t = []
f_est = []
f_meas = []
f_err = []

rim_change_times_list = [[], [], [], []]
last_contact_rim_list = ['', '', '', '']
test_data = [[],[],[],[]]

body_mass = 5

for idx in range(df_data.__len__()):
    # if idx < 3000: continue
    
    data = df_data.iloc[idx, :]
    
    print(f'Time = {data["time"]}')
    
    phi_list = [[data['A_phi_r'], data['A_phi_l']], [data['B_phi_r'], data['B_phi_l']],
                [data['C_phi_r'], data['C_phi_l']], [data['D_phi_r'], data['D_phi_l']]]
    
    trq_list = [[data['A_trq_r'], data['A_trq_l']], [data['B_trq_r'], data['B_trq_l']],
                [data['C_trq_r'], data['C_trq_l']], [data['D_trq_r'], data['D_trq_l']]]
    
    force_list = []
    for i in range(4):
        theta = (phi_list[i][0]-phi_list[i][1])/2 + np.deg2rad(17)
        beta = -(phi_list[i][0]+phi_list[i][1])/2
        alpha, contact_rim = LegKinematics.get_alpha(theta=theta, beta=beta)
        
        print(f'Theta = {round(np.rad2deg(theta), 4)}; Beta = {round(np.rad2deg(beta), 4)}; Alpha = {round(np.rad2deg(alpha), 4)}')
        
        jacobian = LegKinematics.get_jacobian(theta=theta, beta=beta, alpha=alpha)
    
        action_force = np.linalg.inv(jacobian).T @ trq_list[i]
        
        body_force = body_mass * np.array([data['acc_x'], data['acc_z']])
        body_force = np.array([0, 0])
        link_force = LegKinematics.get_link_force(enable_link_force)

        force_list.append(action_force+body_force+link_force)
        
        if last_contact_rim_list[i] != contact_rim: rim_change_times_list[i].append(data['time'])
        last_contact_rim_list[i] = contact_rim
        
        P = LegKinematics.get_contact_point(alpha)
        P_deriv = np.array([P[0].deriv(), P[1].deriv()])
        
        P = [P[0](theta), P[1](theta)]
        P_deriv = [P_deriv[0](theta), P_deriv[1](theta)]
    
    t.append(data['time'])
    f_est.append(force_list)
    f_meas.append([[data['force_plate_1_fx'], data['force_plate_1_fz']],
                   [data['force_plate_2_fx'], data['force_plate_2_fz']],
                   [data['force_plate_3_fx'], data['force_plate_3_fz']],
                   [data['force_plate_4_fx'], data['force_plate_4_fz']]])

test_data = np.array(test_data)
f_meas = np.array(f_meas)
f_est = np.array(f_est)
f_err = f_est - f_meas


f_est_filtered = np.zeros_like(f_est)
f_meas_filtered = np.zeros_like(f_meas)
f_err_filtered = np.zeros_like(f_err)

# filter parameters
cutoff_frequency = 10.0
sampling_frequency = 1000.0
filter_order = 4

f_est_filtered = np.zeros_like(f_est)
f_meas_filtered = np.zeros_like(f_meas)
f_err_filtered = np.zeros_like(f_err)

for i in range(4):
    f_est_filtered[:, i, 0] = butter_lowpass_filter(f_est[:, i, 0], cutoff_frequency, sampling_frequency, filter_order)
    f_est_filtered[:, i, 1] = butter_lowpass_filter(f_est[:, i, 1], cutoff_frequency, sampling_frequency, filter_order)
    f_meas_filtered[:, i, 0] = butter_lowpass_filter(f_meas[:, i, 0], cutoff_frequency, sampling_frequency, filter_order)
    f_meas_filtered[:, i, 1] = butter_lowpass_filter(f_meas[:, i, 1], cutoff_frequency, sampling_frequency, filter_order)
    f_err_filtered[:, i, 0] = butter_lowpass_filter(f_err[:, i, 0], cutoff_frequency, sampling_frequency, filter_order)
    f_err_filtered[:, i, 1] = butter_lowpass_filter(f_err[:, i, 1], cutoff_frequency, sampling_frequency, filter_order)


plt.figure(figsize=(16, 12))
linewidth = 1

if not plot_all:
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

    # plt.plot(t, test_data[0, :, 0], label=f'Data x', linestyle='-', color='blue', linewidth=linewidth)
    # plt.plot(t, test_data[0, :, 1], label=f'Data y', linestyle='-', color='green', linewidth=linewidth)
    
    for time_point in rim_change_times_list[0]: plt.axvline(x=time_point, color='red', linestyle='--', linewidth=2)

    plt.title('Comparison of Estimated and Measured Forces', fontsize=24)
    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('Force (N)', fontsize=20)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(range(0, 76, 5), fontsize=20)
    plt.grid(True)

    plt.show()
    # '''
# else:
    # '''
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
    # '''