import LegKinematics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import butter,filtfilt


def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / fs * 2
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


###
mode = 'real'  # 'sim' or 'real'
motor_filename = '0722'
force_filename = '0722'

motor_data_path = f'{os.getcwd()}/data/{mode}/motor/{motor_filename}.csv'
force_data_path = f'{os.getcwd()}/data/{mode}/force/{force_filename}.csv'

df_motor_data = pd.read_csv(motor_data_path)
df_force_data = pd.read_csv(force_data_path)


###
trq_list = []
phi_list = []
force_list = []

if mode == 'sim':
    df_data = pd.merge(df_force_data, df_motor_data, on='time')
    df_data.dropna(inplace=True)
    
    for i in range(df_data.__len__()):
        data = df_data.iloc[i, :]
        
        trq_list.append([[data['A_trq_r'], data['A_trq_l']],
                         [data['B_trq_r'], data['B_trq_l']],
                         [data['C_trq_r'], data['C_trq_l']],
                         [data['D_trq_r'], data['D_trq_l']]])
    
        phi_list.append([[data['A_phi_r'], data['A_phi_l']],
                         [data['B_phi_r'], data['B_phi_l']],
                         [data['C_phi_r'], data['C_phi_l']],
                         [data['D_phi_r'], data['D_phi_l']]])

        force_list.append([[data['force_plate_1_fx'], data['force_plate_1_fz']],
                           [data['force_plate_2_fx'], data['force_plate_2_fz']],
                           [data['force_plate_3_fx'], data['force_plate_3_fz']],
                           [data['force_plate_4_fx'], data['force_plate_4_fz']]])
elif mode == 'real':
    for i in range(df_motor_data.__len__()):
        data = df_motor_data.iloc[i, :]
        
        trq_list.append([[data['AR_rpy_torq']*2.2, data['AR_rpy_torq']*2.2],
                         [data['BR_rpy_torq']*2.2, data['BR_rpy_torq']*2.2],
                         [data['CR_rpy_torq']*2.2, data['CR_rpy_torq']*2.2],
                         [data['DR_rpy_torq']*2.2, data['DR_rpy_torq']*2.2]])

        phi_list.append([[data['AR_rpy_pos'], data['AL_rpy_pos']],
                         [data['BR_rpy_pos'], data['BL_rpy_pos']],
                         [data['CR_rpy_pos'], data['CL_rpy_pos']],
                         [data['DR_rpy_pos'], data['DL_rpy_pos']]])

    trigger = False
    for i in range(df_force_data.__len__()):
        data = df_force_data.iloc[i, :]
        
        if str(data['Trigger1_x']) != 'nan': trigger = True
        
        if not trigger: continue
        
        force_list.append([[data['Fx_1'], -data['Fz_1']],
                           [data['Fx_2'], -data['Fz_2']],
                           [data['Fx_3'], -data['Fz_3']],
                           [data['Fx_4'], -data['Fz_4']]])
        
    if df_motor_data.__len__() > len(force_list):
        trq_list = trq_list[:len(force_list)]
        phi_list = trq_list[:len(force_list)]
        
    elif df_motor_data.__len__() < len(force_list):
        force_list = force_list[:df_motor_data.__len__()]

# print(len(trq_list))
# print(len(phi_list))
# print(len(force_list))

trq_list = np.array(trq_list)
force_list = np.array(force_list)


cutoff_frequency = 5.0
sampling_frequency = 1000.0
filter_order = 5

for i in range(4):
    trq_list[:, i, 0] = butter_lowpass_filter(trq_list[:, i, 0], cutoff_frequency, sampling_frequency, filter_order)
    trq_list[:, i, 1] = butter_lowpass_filter(trq_list[:, i, 1], cutoff_frequency, sampling_frequency, filter_order)
    force_list[:, i, 0] = butter_lowpass_filter(force_list[:, i, 0], cutoff_frequency, sampling_frequency, filter_order)
    force_list[:, i, 1] = butter_lowpass_filter(force_list[:, i, 1], cutoff_frequency, sampling_frequency, filter_order)


###
t = []
f_est = []
f_meas = []
data_test = []
contact_rim_hist = [[2, 2, 2, 2]]
contact_rim_changed_time = []

for i in range(len(trq_list)):
# for i in range(70000):
    t.append(i)
    f_est.append([])
    f_meas.append([])
    data_test.append([])
    contact_rim_hist.append([])
    contact_rim_changed_time.append([])
    
    for j in range(4):
        trq = trq_list[i][j]
        phi = phi_list[i][j]
        force = force_list[i][j]
        
        theta = (phi[0]-phi[1])/2 + np.deg2rad(17)
        beta = -(phi[0]+phi[1])/2
        
        print(f't = {t[i]}:  Theta = {round(np.rad2deg(theta), 2)}, Beta = {round(np.rad2deg(beta), 2)}, Trq = {trq}')
        contact_rim, alpha, P = LegKinematics.get_contact_point(theta, beta)
        jacobian = LegKinematics.get_jacobian(theta, beta)
        print(jacobian)
        action_force = np.linalg.inv(jacobian).T @ trq
        link_force = LegKinematics.get_link_force(enable=True)
        total_force = action_force + link_force
        
        f_est[i].append(total_force)
        f_meas[i].append(force)
        if i != 0 and contact_rim != contact_rim_hist[i-1][j]: contact_rim_changed_time[i].append(i)
        
        contact_rim_hist[i].append(contact_rim)
        
        data_test[i].append(np.linalg.det(jacobian))
        
        
        
t = np.array(t)
f_est = np.array(f_est)
f_meas = np.array(f_meas)
data_test = np.array(data_test)


###
linewidth = 1

fig = plt.figure(figsize=(16, 12))
ax1 = fig.add_subplot(1, 1, 1)

ax1.plot(t/1000, f_est[:, 0, 1], label=f'Estimated Force Z {i}', linestyle='-', color='blue', linewidth=1)
ax1.plot(t/1000, f_meas[:, 0, 1], label=f'Measured Force Z {i}', linestyle='-', color='green', linewidth=1)
# ax1.plot(t, data_test[:, 0, 0], label=f'', linestyle='-', color='red', linewidth=1)
# ax1.plot(t, data_test[:, 0, 1], label=f'', linestyle='-', color='blue', linewidth=1)

ax1.set_xlabel('Time (s)', fontsize=20)
ax1.set_ylabel('Force (N)', fontsize=20)
ax1.tick_params(axis='both', which='major', labelsize=20)
ax1.set_title(f'Comparison of Estimated and Measured Forces {i}', fontsize=24)
ax1.legend(loc='upper left', fontsize=14)
ax1.grid(True)

# ax2 = ax1.twinx()
# ax2.plot(t/1000, data_test[:, i], label=f'det J {i}', linestyle='-', color='red', linewidth=linewidth)
# ax2.set_ylabel('det J', fontsize=20)
# ax2.tick_params(axis='both', which='major', labelsize=20)
# ax2.legend(loc='upper right', fontsize=14)

plt.tight_layout()
plt.show()

###
fig = plt.figure(figsize=(16, 12))
for i in range(4):
    ax1 = fig.add_subplot(2, 2, i + 1)

    ax1.plot(t/1000, f_est[:, i, 1], label=f'Estimated Force Z {i}', linestyle='-', color='blue', linewidth=1)
    ax1.plot(t/1000, f_meas[:, i, 1], label=f'Measured Force Z {i}', linestyle='-', color='green', linewidth=1)
    # ax1.plot(t, data_test[:, i, 1], label=f'', linestyle='-', color='red', linewidth=1)
    # ax1.plot(t, data_test[:, i, 1], label=f'', linestyle='-', color='blue', linewidth=1)
    
    ax1.set_xlabel('Time (s)', fontsize=20)
    ax1.set_ylabel('Force (N)', fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax1.set_title(f'Comparison of Estimated and Measured Forces {i}', fontsize=24)
    ax1.legend(loc='upper left', fontsize=14)
    ax1.grid(True)
    
    ax2 = ax1.twinx()
    ax2.plot(t/1000, data_test[:, i], label=f'det J {i}', linestyle='-', color='red', linewidth=linewidth)
    ax2.set_ylabel('det J', fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax2.legend(loc='upper right', fontsize=14)
    
    plt.tight_layout()
    
plt.show()