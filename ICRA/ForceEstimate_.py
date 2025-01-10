from scipy.signal import butter,filtfilt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np
import os

import LegKinematics


def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / fs * 2
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def read_data(motor_data_path, force_data_path, start_idx, end_idx):
    df_motor_data = pd.read_csv(motor_data_path)
    df_force_data = pd.read_csv(force_data_path)
    
    try:
        df_data = pd.merge(df_force_data, df_motor_data, on='time').dropna()

        trq_list = -df_data[['A_trq_r', 'A_trq_l', 'B_trq_r', 'B_trq_l',
                            'C_trq_r', 'C_trq_l', 'D_trq_r', 'D_trq_l']].values.reshape(-1, 4, 2)
        phi_list = df_data[['A_phi_r', 'A_phi_l', 'B_phi_r', 'B_phi_l',
                            'C_phi_r', 'C_phi_l', 'D_phi_r', 'D_phi_l']].values.reshape(-1, 4, 2)
        force_ref = -df_data[['force_plate_1_fx', 'force_plate_1_fz', 'force_plate_2_fx', 'force_plate_2_fz',
                            'force_plate_3_fx', 'force_plate_3_fz', 'force_plate_4_fx', 'force_plate_4_fz']].values.reshape(-1, 4, 2)
        force_ref[:, 0, 0] *= -1
        force_ref[:, 3, 0] *= -1
        # force_ref[:, 3, 0] *= -1
        
    except:
        kt = [[2.2, 2.2], [2.2, 2.2], [2.2, 2.2], [2.2, 2.2]]
        trq_list = -df_motor_data[['AR_rpy_torq', 'AL_rpy_torq', 'BR_rpy_torq', 'BL_rpy_torq',
                                'CR_rpy_torq', 'CL_rpy_torq', 'DR_rpy_torq', 'DL_rpy_torq']].values.reshape(-1, 4, 2)*kt
        phi_list = df_motor_data[['AR_rpy_pos', 'AL_rpy_pos', 'BR_rpy_pos', 'BL_rpy_pos',
                                'CR_rpy_pos', 'CL_rpy_pos', 'DR_rpy_pos', 'DL_rpy_pos']].values.reshape(-1, 4, 2)
        
        trigger_indices = np.where(~df_force_data['Trigger_x'].isna())[0]
        if trigger_indices.size > 0:
            start_index = trigger_indices[0]
            force_ref = df_force_data.loc[start_index:, ['Fx_1', 'Fz_1', 'Fx_4', 'Fz_4',
                                                        'Fx_3', 'Fz_3', 'Fx_2', 'Fz_2']].values.reshape(-1, 4, 2)
                        
            force_ref[:, 0, 1] += 3.75
            force_ref[:, 3, 1] += 3.66
            force_ref[:, 0, 0] -= 2
            force_ref[:, 2, 0] += 40
            # force_ref[:, 2, 0] += 35
            force_ref[:, 3, 0] -= 6
    
    trq_list = trq_list[start_idx:end_idx]
    phi_list = phi_list[start_idx:end_idx]
    force_ref = force_ref[start_idx:end_idx]
    
    cutoff_frequency = 35.0
    sampling_frequency = 1000.0
    filter_order = 5
    
    for i in range(4):
        trq_list[:, i, 0] = butter_lowpass_filter(trq_list[:, i, 0], cutoff_frequency, sampling_frequency, filter_order)
        trq_list[:, i, 1] = butter_lowpass_filter(trq_list[:, i, 1], cutoff_frequency, sampling_frequency, filter_order)
        force_ref[:, i, 0] = butter_lowpass_filter(force_ref[:, i, 0], cutoff_frequency, sampling_frequency, filter_order)
        force_ref[:, i, 1] = butter_lowpass_filter(force_ref[:, i, 1], cutoff_frequency, sampling_frequency, filter_order)
    
    print('= = = Read Data = = =')
    print(f'Trq list length = {trq_list.shape}')
    print(f'Phi list length = {phi_list.shape}')
    print(f'Force list length = {force_ref.shape}')
    
    return trq_list, phi_list, force_ref


def force_estimate(trq_list, phi_list):
    theta_list = (phi_list[:, :, 0]-phi_list[:, :, 1])/2 + np.deg2rad(17)
    beta_list = (-phi_list[:, :, 0]-phi_list[:, :, 1])/2
    
    reaction_force = [[], [], [], []]
    contact_rims = []
    jacobians = []
    
    for i in range(4):
        jacobian, contact_rim = LegKinematics.get_jacobian(theta_list[:,i], beta_list[:,i])
        jacobian = np.array([np.linalg.inv(j).T for j in jacobian])
        
        reaction_force[i] = np.einsum('ijk,ik->ij', jacobian, trq_list[:,i])#+0.5
        if i == 0 or i == 3: reaction_force[i] *= [-1, 1]

        contact_rims.append(contact_rim)
        jacobians.append(jacobian)
        
    reaction_force += LegKinematics.get_link_force().reshape(1,1,2)
    reaction_force = np.array(reaction_force)
    reaction_force[abs(np.linalg.det(jacobians))>800] = 0
    reaction_force[:,:,1][reaction_force[:,:,1]>0] = 0
    
    return reaction_force, contact_rims, jacobians


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

# Define a function to process data for a given mode
def process_data(mode):
    force_ref_list = []
    force_est_list = []
    det_jacobian_list = []
    
    if mode == 'Walk': start_idx, end_idx = 2000, 5000
    if mode == 'Hybrid': start_idx, end_idx = 2000, 9500
    if mode == 'Wave': start_idx, end_idx = 0, 29000
        
    for i in range(10):
        print(f'Processing {mode} Real Data {i+1} ...')
        
        motor_data_path = f'{os.getcwd()}/Data/{mode}/motor_real_{i+1}.csv'
        force_data_path = f'{os.getcwd()}/Data/{mode}/force_real_{i+1}.csv'
        trq_list, phi_list, force_ref = read_data(motor_data_path, force_data_path, start_idx, end_idx)
        
        if mode == 'Wave':
            trq_list  = np.concatenate((trq_list [5000:6000], trq_list [16000:18000], trq_list [28000:29000]), axis=0)
            phi_list  = np.concatenate((phi_list [5000:6000], phi_list [16000:18000], phi_list [28000:29000]), axis=0)
            force_ref = np.concatenate((force_ref[5000:6000], force_ref[16000:18000], force_ref[28000:29000]), axis=0)
    
        force_est, contact_rims, jacobians = force_estimate(trq_list, phi_list)
        
        force_ref_list.append(force_ref)
        force_est_list.append(force_est)
        
        det_jacobian_list.append(np.array([np.linalg.det(np.linalg.inv(jacobians)[i, :]) for i in range(4)]))
        
    force_ref_list = np.array(force_ref_list)
    force_est_list = np.array(force_est_list)
    force_est_list[:,:,:,1][force_est_list[:,:,:,1]>0] = 0
    det_jacobian_list = np.array(det_jacobian_list)
    
    force_ref_mean_list = np.mean(force_ref_list, axis=0)
    force_ref_std_list = np.std(force_ref_list, axis=0)
    force_est_mean_list = np.mean(force_est_list, axis=0)
    
    det_jacobian_list = np.mean(det_jacobian_list, axis=0)
        
    # Process simulation data
    if mode == 'Walk': start_idx, end_idx = 2776, 3976
    if mode == 'Hybrid': start_idx, end_idx = 2800, 5800
    if mode == 'Wave': start_idx, end_idx = 0, 13600
    
    motor_data_path = f'{os.getcwd()}/Data/{mode}/motor_sim.csv'
    force_data_path = f'{os.getcwd()}/Data/{mode}/force_sim.csv'
    trq_list, phi_list, force_ref = read_data(motor_data_path, force_data_path, start_idx, end_idx)
    
    if mode == 'Wave':
        trq_list  = np.concatenate((trq_list [4000:4400], trq_list [8400:9200], trq_list [13200:13600]), axis=0)
        phi_list  = np.concatenate((phi_list [4000:4400], phi_list [8400:9200], phi_list [13200:13600]), axis=0)
        force_ref = np.concatenate((force_ref[4000:4400], force_ref[8400:9200], force_ref[13200:13600]), axis=0)

    force_est, contact_rims, jacobians = force_estimate(trq_list, phi_list)
    
    force_ref_sim = np.array(force_ref)
    force_est_sim = np.array(force_est)
    
    return force_ref_mean_list, force_ref_std_list, force_est_mean_list, force_ref_sim, force_est_sim

# Start of main plotting code
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{mathptmx}')
plt.rcParams.update({'font.size': 26})
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
linewidth = 1.5

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(36, 18), constrained_layout=True)
outer_gs = gridspec.GridSpec(nrows=4, ncols=9, wspace=2, hspace=0.6)

modes = ['Wave', 'Walk', 'Hybrid']
phase_offsets = {'Walk':11, 'Hybrid':9, 'Wave':19}

for mode_idx, mode in enumerate(modes):
    start_col = mode_idx * 3
    # Process data
    force_ref_mean_list, force_ref_std_list, force_est_mean_list, force_ref_sim, force_est_sim = process_data(mode)
    if mode == 'Walk': force_ref_mean_list[:, 2, 0] -= 5
    phase_offset = phase_offsets[mode]
    
    for fig_idx in range(4):
        ax = fig.add_subplot(outer_gs[fig_idx, start_col:start_col+3])
        
        mod_idx = 1 if fig_idx in [0, 2] else 2
        direction = 1 if fig_idx in [0, 1] else 0
        
        len_force_ref = force_ref_mean_list[phase_offset:, mod_idx, direction].__len__()
        len_force_est = force_est_mean_list[mod_idx, :, direction].__len__()
        
        # Adjust time scaling per mode
        if mode == 'Wave':
            time_scale_real = 1000
            time_scale_sim = 400
        else:
            time_scale_real = 1000
            time_scale_sim = 400
        
        real_ref, = ax.plot(np.array(range(len_force_ref))/time_scale_real, force_ref_mean_list[phase_offset:, mod_idx, direction], label='Measured Force (Real)', linestyle='-', color='red', linewidth=linewidth)
        real_est, = ax.plot(np.array(range(len_force_est))/time_scale_real, force_est_mean_list[mod_idx, :, direction], label='Estimated Force (Real)', linestyle='--', color='blue', linewidth=linewidth)
        
        ax.fill_between(np.array(range(len_force_ref))/time_scale_real, 
                        (force_ref_mean_list - force_ref_std_list)[phase_offset:, mod_idx, direction], 
                        (force_ref_mean_list + force_ref_std_list)[phase_offset:, mod_idx, direction], 
                        color='gray', alpha=0.5, label='±1 Std Dev (Real)')
        
        len_force_ref_sim = force_ref_sim[:, mod_idx, direction].__len__()
        len_force_est_sim = force_est_sim[mod_idx, :, direction].__len__()
        
        sim_ref, = ax.plot(np.array(range(len_force_ref_sim))/time_scale_sim, force_ref_sim[:, mod_idx, direction], label='Measured Force (Sim)', linestyle='-', color='green', linewidth=linewidth)
        sim_est, = ax.plot(np.array(range(len_force_est_sim))/time_scale_sim, force_est_sim[mod_idx, :, direction], label='Estimated Force (Sim)', linestyle='--', color='orange', linewidth=linewidth)
        
        title = f'{["Right Front", "Left Hind"][fig_idx % 2]} - {["Z", "X"][fig_idx // 2]} Direction'
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Force (N)')
        ax.set_title(title, fontsize=26)
        ax.grid(True)
        
        if mode == 'Wave': x_ticks = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        if mode == 'Walk': x_ticks = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        if mode == 'Hybrid': x_ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{tick:.1f}' for tick in x_ticks])
        
        y_min, y_max = ax.get_ylim()
        y_min = (np.floor(y_min / 50) * 50)
        y_max = (np.ceil(y_max / 50) * 50)
        if y_min > -100 and y_max < 100: y_ticks = np.arange(y_min+25, y_max, 25)
        else: y_ticks = np.arange(y_min+50, y_max, 50)
        ax.set_yticks(y_ticks)
        
        # Optionally format y-tick labels
        ax.set_yticklabels([f'{int(tick)}' for tick in y_ticks])
        
        
        if mode_idx == 0 and fig_idx == 0:
            handles = [real_ref, sim_ref, real_est, sim_est, ax.collections[0]]
            labels = [h.get_label() for h in handles]
    
    # Add big title for the mode
    x_position = (start_col+[2,1.5,1][mode_idx]) / 9  # Center of the mode's columns
    y_position = 0.92  # Adjust as needed
    fig.text(x_position, y_position, ['(a)', '(b)', '(c)'][mode_idx], ha='center', va='center', fontsize=30)

# Add the legend
fig.legend(handles, labels, loc='center', ncol=5, bbox_to_anchor=(0.42, 0.05), fontsize=26)

plt.tight_layout()
fig.subplots_adjust(bottom=0.15)

plt.savefig('Force.pdf', format='pdf', bbox_inches='tight', dpi=150)
plt.show()

if __name__ == '__main_':
    
    force_ref_list = []
    force_est_list = []
    det_jacobian_list = []
    mode = 'Walk'
    # mode = 'Hybrid'
    # mode = 'Wave'
    
    if mode == 'Walk': start_idx, end_idx = 2000, 5000
    if mode == 'Hybrid': start_idx, end_idx = 2000, 9500
    if mode == 'Wave': start_idx, end_idx = 0, 29000
    
    for i in range(10):
        print(f'Processing {i+1} ...')
        
        motor_data_path = f'{os.getcwd()}/Data/{mode}/motor_real_{i+1}.csv'
        force_data_path = f'{os.getcwd()}/Data/{mode}/force_real_{i+1}.csv'
        trq_list, phi_list, force_ref = read_data(motor_data_path, force_data_path, start_idx, end_idx)
        
        if mode == 'Wave':
            trq_list  = np.concatenate((trq_list [5000:6000], trq_list [16000:18000], trq_list [28000:29000]), axis=0)
            phi_list  = np.concatenate((phi_list [5000:6000], phi_list [16000:18000], phi_list [28000:29000]), axis=0)
            force_ref = np.concatenate((force_ref[5000:6000], force_ref[16000:18000], force_ref[28000:29000]), axis=0)

        force_est, contact_rims, jacobians = force_estimate(trq_list, phi_list)
        
        force_ref_list.append(force_ref)
        force_est_list.append(force_est)
        
        det_jacobian_list.append(np.array([np.linalg.det(np.linalg.inv(jacobians)[i, :]) for i in range(4)]))
        
    force_ref_list = np.array(force_ref_list)
    force_est_list = np.array(force_est_list)
    force_est_list[:,:,:,1][force_est_list[:,:,:,1]>0] = 0
    det_jacobian_list = np.array(det_jacobian_list)
    
    force_ref_mean_list = np.mean(force_ref_list, axis=0)
    force_ref_std_list = np.std(force_ref_list, axis=0)
    force_est_mean_list = np.mean(force_est_list, axis=0)
    
    det_jacobian_list = np.mean(det_jacobian_list, axis=0)
        
    if mode == 'Walk': start_idx, end_idx = 2776, 3976
    if mode == 'Hybrid': start_idx, end_idx = 2800, 5800
    if mode == 'Wave': start_idx, end_idx = 0, 13600
    
    motor_data_path = f'{os.getcwd()}/Data/{mode}/motor_sim.csv'
    force_data_path = f'{os.getcwd()}/Data/{mode}/force_sim.csv'
    trq_list, phi_list, force_ref = read_data(motor_data_path, force_data_path, start_idx, end_idx)
    
    if mode == 'Wave':
        trq_list  = np.concatenate((trq_list [4000:4400], trq_list [8400:9200], trq_list [13200:13600]), axis=0)
        phi_list  = np.concatenate((phi_list [4000:4400], phi_list [8400:9200], phi_list [13200:13600]), axis=0)
        force_ref = np.concatenate((force_ref[4000:4400], force_ref[8400:9200], force_ref[13200:13600]), axis=0)

    force_est, contact_rims, jacobians = force_estimate(trq_list, phi_list)
    
    force_ref_sim = np.array(force_ref)
    force_est_sim = np.array(force_est)
        
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{mathptmx}')
    plt.rcParams.update({'font.size': 26})
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    linewidth = 1.5
    
    # '''
    fig = plt.figure(figsize=(36, 18))
    
    for fig_idx in range(4):
        ax = fig.add_subplot(4, 3, fig_idx*3+1)
        
    
        mod_idx = 1 if fig_idx in [0, 2] else 2
        direction = 1 if fig_idx in [0, 1] else 0
        
        phase_offset = 11
        # phase_offset = 9
        # phase_offset = 19
        len_force_ref = force_ref_mean_list[phase_offset:, mod_idx, direction].__len__()
        len_force_est = force_est_mean_list[mod_idx, :, direction].__len__()
        
        real_ref, = ax.plot(np.array(range(len_force_ref))/1000, force_ref_mean_list[phase_offset:, mod_idx, direction], label='Measured Force (Real)', linestyle='-', color='red', linewidth=linewidth)
        real_est, = ax.plot(np.array(range(len_force_est))/1000, force_est_mean_list[mod_idx, :, direction], label='Estimated Force (Real)', linestyle='--', color='blue', linewidth=linewidth)
        
        ax.fill_between(np.array(range(len_force_ref))/1000, 
                        (force_ref_mean_list - force_ref_std_list)[phase_offset:, mod_idx, direction], 
                        (force_ref_mean_list + force_ref_std_list)[phase_offset:, mod_idx, direction], 
                        color='gray', alpha=0.5, label='±1 Std Dev (Real)')
        
        
        
        len_force_ref_sim = force_ref_sim[:, mod_idx, direction].__len__()
        len_force_est_sim = force_est_sim[mod_idx, :, direction].__len__()
        
        sim_ref, = ax.plot(np.array(range(len_force_ref_sim))/400, force_ref_sim[:, mod_idx, direction], label='Measured Force (Sim)', linestyle='-', color='green', linewidth=linewidth)
        sim_est, = ax.plot(np.array(range(len_force_est_sim))/400, force_est_sim[mod_idx, :, direction], label='Estimated Force (Sim)', linestyle='--', color='orange', linewidth=linewidth)
        
        
        title = f'{["Right Front", "Left Hind"][fig_idx % 2]} - {["Z", "X"][fig_idx // 2]} Direction'
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Force (N)')
        ax.set_title(title, fontsize=26)
        ax.grid(True)
        
        
        if fig_idx == 0:
            handles = [real_ref, sim_ref, real_est, sim_est, ax.collections[0]]
            labels = [h.get_label() for h in handles]
        
        fig_idx += 1
        
        
    force_ref_list = []
    force_est_list = []
    det_jacobian_list = []
    mode = 'Walk'
    mode = 'Hybrid'
    # mode = 'Wave'
    
    if mode == 'Walk': start_idx, end_idx = 2000, 5000
    if mode == 'Hybrid': start_idx, end_idx = 2000, 9500
    if mode == 'Wave': start_idx, end_idx = 0, 29000
    
    for i in range(10):
        print(f'Processing {i+1} ...')
        
        motor_data_path = f'{os.getcwd()}/Data/{mode}/motor_real_{i+1}.csv'
        force_data_path = f'{os.getcwd()}/Data/{mode}/force_real_{i+1}.csv'
        trq_list, phi_list, force_ref = read_data(motor_data_path, force_data_path, start_idx, end_idx)
        
        if mode == 'Wave':
            trq_list  = np.concatenate((trq_list [5000:6000], trq_list [16000:18000], trq_list [28000:29000]), axis=0)
            phi_list  = np.concatenate((phi_list [5000:6000], phi_list [16000:18000], phi_list [28000:29000]), axis=0)
            force_ref = np.concatenate((force_ref[5000:6000], force_ref[16000:18000], force_ref[28000:29000]), axis=0)

        force_est, contact_rims, jacobians = force_estimate(trq_list, phi_list)
        
        force_ref_list.append(force_ref)
        force_est_list.append(force_est)
        
        det_jacobian_list.append(np.array([np.linalg.det(np.linalg.inv(jacobians)[i, :]) for i in range(4)]))
        
    force_ref_list = np.array(force_ref_list)
    force_est_list = np.array(force_est_list)
    force_est_list[:,:,:,1][force_est_list[:,:,:,1]>0] = 0
    det_jacobian_list = np.array(det_jacobian_list)
    
    force_ref_mean_list = np.mean(force_ref_list, axis=0)
    force_ref_std_list = np.std(force_ref_list, axis=0)
    force_est_mean_list = np.mean(force_est_list, axis=0)
    
    det_jacobian_list = np.mean(det_jacobian_list, axis=0)
        
    if mode == 'Walk': start_idx, end_idx = 2776, 3976
    if mode == 'Hybrid': start_idx, end_idx = 2800, 5800
    if mode == 'Wave': start_idx, end_idx = 0, 13600
    
    motor_data_path = f'{os.getcwd()}/Data/{mode}/motor_sim.csv'
    force_data_path = f'{os.getcwd()}/Data/{mode}/force_sim.csv'
    trq_list, phi_list, force_ref = read_data(motor_data_path, force_data_path, start_idx, end_idx)
    
    if mode == 'Wave':
        trq_list  = np.concatenate((trq_list [4000:4400], trq_list [8400:9200], trq_list [13200:13600]), axis=0)
        phi_list  = np.concatenate((phi_list [4000:4400], phi_list [8400:9200], phi_list [13200:13600]), axis=0)
        force_ref = np.concatenate((force_ref[4000:4400], force_ref[8400:9200], force_ref[13200:13600]), axis=0)

    force_est, contact_rims, jacobians = force_estimate(trq_list, phi_list)
    
    force_ref_sim = np.array(force_ref)
    force_est_sim = np.array(force_est)
    
    
    for fig_idx in range(4):
        ax = fig.add_subplot(4, 3, fig_idx*3+2)
        
    
        mod_idx = 1 if fig_idx in [0, 2] else 2
        direction = 1 if fig_idx in [0, 1] else 0
        
        phase_offset = 11
        phase_offset = 9
        # phase_offset = 19
        len_force_ref = force_ref_mean_list[phase_offset:, mod_idx, direction].__len__()
        len_force_est = force_est_mean_list[mod_idx, :, direction].__len__()
        
        real_ref, = ax.plot(np.array(range(len_force_ref))/1000, force_ref_mean_list[phase_offset:, mod_idx, direction], label='Measured Force (Real)', linestyle='-', color='red', linewidth=linewidth)
        real_est, = ax.plot(np.array(range(len_force_est))/1000, force_est_mean_list[mod_idx, :, direction], label='Estimated Force (Real)', linestyle='--', color='blue', linewidth=linewidth)
        
        ax.fill_between(np.array(range(len_force_ref))/1000, 
                        (force_ref_mean_list - force_ref_std_list)[phase_offset:, mod_idx, direction], 
                        (force_ref_mean_list + force_ref_std_list)[phase_offset:, mod_idx, direction], 
                        color='gray', alpha=0.5, label='±1 Std Dev (Real)')
        
        len_force_ref_sim = force_ref_sim[:, mod_idx, direction].__len__()
        len_force_est_sim = force_est_sim[mod_idx, :, direction].__len__()
        
        sim_ref, = ax.plot(np.array(range(len_force_ref_sim))/400, force_ref_sim[:, mod_idx, direction], label='Measured Force (Sim)', linestyle='-', color='green', linewidth=linewidth)
        sim_est, = ax.plot(np.array(range(len_force_est_sim))/400, force_est_sim[mod_idx, :, direction], label='Estimated Force (Sim)', linestyle='--', color='orange', linewidth=linewidth)
        
        title = f'{["Right Front", "Left Hind"][fig_idx % 2]} - {["Z", "X"][fig_idx // 2]} Direction'
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Force (N)')
        ax.set_title(title, fontsize=26)
        ax.grid(True)
        
        
        if fig_idx == 0:
            handles = [real_ref, sim_ref, real_est, sim_est, ax.collections[0]]
            labels = [h.get_label() for h in handles]
        
        fig_idx += 1
    
    force_ref_list = []
    force_est_list = []
    det_jacobian_list = []
    mode = 'Walk'
    mode = 'Hybrid'
    mode = 'Wave'
    
    if mode == 'Walk': start_idx, end_idx = 2000, 5000
    if mode == 'Hybrid': start_idx, end_idx = 2000, 9500
    if mode == 'Wave': start_idx, end_idx = 0, 29000
    
    for i in range(10):
        print(f'Processing {i+1} ...')
        
        motor_data_path = f'{os.getcwd()}/Data/{mode}/motor_real_{i+1}.csv'
        force_data_path = f'{os.getcwd()}/Data/{mode}/force_real_{i+1}.csv'
        trq_list, phi_list, force_ref = read_data(motor_data_path, force_data_path, start_idx, end_idx)
        
        if mode == 'Wave':
            trq_list  = np.concatenate((trq_list [5000:6000], trq_list [16000:18000], trq_list [28000:29000]), axis=0)
            phi_list  = np.concatenate((phi_list [5000:6000], phi_list [16000:18000], phi_list [28000:29000]), axis=0)
            force_ref = np.concatenate((force_ref[5000:6000], force_ref[16000:18000], force_ref[28000:29000]), axis=0)

        force_est, contact_rims, jacobians = force_estimate(trq_list, phi_list)
        
        force_ref_list.append(force_ref)
        force_est_list.append(force_est)
        
        det_jacobian_list.append(np.array([np.linalg.det(np.linalg.inv(jacobians)[i, :]) for i in range(4)]))
        
    force_ref_list = np.array(force_ref_list)
    force_est_list = np.array(force_est_list)
    force_est_list[:,:,:,1][force_est_list[:,:,:,1]>0] = 0
    det_jacobian_list = np.array(det_jacobian_list)
    
    force_ref_mean_list = np.mean(force_ref_list, axis=0)
    force_ref_std_list = np.std(force_ref_list, axis=0)
    force_est_mean_list = np.mean(force_est_list, axis=0)
    
    det_jacobian_list = np.mean(det_jacobian_list, axis=0)
        
    if mode == 'Walk': start_idx, end_idx = 2776, 3976
    if mode == 'Hybrid': start_idx, end_idx = 2800, 5800
    if mode == 'Wave': start_idx, end_idx = 0, 13600
    
    motor_data_path = f'{os.getcwd()}/Data/{mode}/motor_sim.csv'
    force_data_path = f'{os.getcwd()}/Data/{mode}/force_sim.csv'
    trq_list, phi_list, force_ref = read_data(motor_data_path, force_data_path, start_idx, end_idx)
    
    if mode == 'Wave':
        trq_list  = np.concatenate((trq_list [4000:4400], trq_list [8400:9200], trq_list [13200:13600]), axis=0)
        phi_list  = np.concatenate((phi_list [4000:4400], phi_list [8400:9200], phi_list [13200:13600]), axis=0)
        force_ref = np.concatenate((force_ref[4000:4400], force_ref[8400:9200], force_ref[13200:13600]), axis=0)

    force_est, contact_rims, jacobians = force_estimate(trq_list, phi_list)
    
    force_ref_sim = np.array(force_ref)
    force_est_sim = np.array(force_est)
    
    for fig_idx in range(4):
        ax = fig.add_subplot(4, 3, fig_idx*3+3)
        
    
        mod_idx = 1 if fig_idx in [0, 2] else 2
        direction = 1 if fig_idx in [0, 1] else 0
        
        phase_offset = 11
        phase_offset = 9
        phase_offset = 19
        len_force_ref = force_ref_mean_list[phase_offset:, mod_idx, direction].__len__()
        len_force_est = force_est_mean_list[mod_idx, :, direction].__len__()
        
        real_ref, = ax.plot(np.array(range(len_force_ref))/1000, force_ref_mean_list[phase_offset:, mod_idx, direction], label='Measured Force (Real)', linestyle='-', color='red', linewidth=linewidth)
        real_est, = ax.plot(np.array(range(len_force_est))/1000, force_est_mean_list[mod_idx, :, direction], label='Estimated Force (Real)', linestyle='--', color='blue', linewidth=linewidth)
        
        ax.fill_between(np.array(range(len_force_ref))/1000, 
                        (force_ref_mean_list - force_ref_std_list)[phase_offset:, mod_idx, direction], 
                        (force_ref_mean_list + force_ref_std_list)[phase_offset:, mod_idx, direction], 
                        color='gray', alpha=0.5, label='±1 Std Dev (Real)')
        
        len_force_ref_sim = force_ref_sim[:, mod_idx, direction].__len__()
        len_force_est_sim = force_est_sim[mod_idx, :, direction].__len__()
        
        sim_ref, = ax.plot(np.array(range(len_force_ref_sim))/400, force_ref_sim[:, mod_idx, direction], label='Measured Force (Sim)', linestyle='-', color='green', linewidth=linewidth)
        sim_est, = ax.plot(np.array(range(len_force_est_sim))/400, force_est_sim[mod_idx, :, direction], label='Estimated Force (Sim)', linestyle='--', color='orange', linewidth=linewidth)
        
        title = f'{["Right Front", "Left Hind"][fig_idx % 2]} - {["Z", "X"][fig_idx // 2]} Direction'
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Force (N)')
        ax.set_title(title, fontsize=26)
        ax.grid(True)
        
        
        if fig_idx == 0:
            handles = [real_ref, sim_ref, real_est, sim_est, ax.collections[0]]
            labels = [h.get_label() for h in handles]
        
        fig_idx += 1
    
    
        
    
    fig.legend(handles, labels, loc='center', ncol=5, bbox_to_anchor=(0.5, 0.05), fontsize=26)
    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    
    plt.savefig('Force.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    # '''
    
    
    '''
    fig, ax = plt.subplots(figsize=(12, 9))
    
    mod_idx = 0
    len_force_ref = force_ref_mean_list[phase_offset:, mod_idx, 1].__len__()
    len_force_est = force_est_mean_list[mod_idx, :-phase_offset, 1].__len__()
    ax.plot(np.array(range(len_force_ref))/1000, abs(force_ref_mean_list[phase_offset:, mod_idx, 1]-force_est_mean_list[mod_idx, :-phase_offset, 1]), label=f'Force Error', linestyle='-', color='black', linewidth=linewidth)
    
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightcoral', 'lightblue']
    labels = ['Upper Rim', 'Lower Rim', 'G (Toe Tip)', 'Lower Rim', 'Upper Rim']
    patches = []
    used_labels = set()
    contact_rims[mod_idx] = np.where(contact_rims[mod_idx] is None, np.nan, contact_rims[mod_idx])
    contact_rims = np.array(contact_rims, dtype=float)
    valid_indices = ~np.isnan(contact_rims[mod_idx])
    change_indices = np.nonzero(valid_indices)[0]
    if change_indices.size > 0:
        start_index = change_indices[0]
        for k in range(1, len(change_indices)):
            if contact_rims[mod_idx, change_indices[k]] != contact_rims[mod_idx, change_indices[k - 1]]:
                ax.axvspan(start_index/400, change_indices[k]/400, color=colors[int(contact_rims[mod_idx, change_indices[k-1]])], alpha=0.3)
                
                phase_color = colors[int(contact_rims[mod_idx, change_indices[k-1]])]
                phase_label = labels[int(contact_rims[mod_idx, change_indices[k-1]])]
                if phase_label not in used_labels and len(used_labels) < 3:
                    patches.append(mpatches.Patch(color=phase_color, label=phase_label))
                    used_labels.add(phase_label)
                start_index = change_indices[k]
                
        ax.axvspan(start_index/400, (change_indices[-1] + (1 if len(force_ref) > change_indices[-1] else 0))/400, color=colors[int(contact_rims[mod_idx, start_index])], alpha=0.3)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (N)')
    ax.set_ylim(-3, 80)
    ax.set_title('Relation Between Error and Determinants of Jacobian Matrix')
    # ax.add_artist(ax.legend())
    ax.legend(loc='upper left', fontsize=26)
    ax.grid(True)
    
    ax_ = ax.twinx()
    ax_.plot(np.array(range(det_jacobian_list[mod_idx].__len__()))/1000, det_jacobian_list[mod_idx], label=f'Determinants', linestyle='--', color='red', linewidth=linewidth*2)
    ax_.set_ylabel('Determinants of Jacobian Matrix')
    ax_.set_ylim(-0.01, 0)
    ax_.tick_params(axis='both', which='major')
    ax_.legend(loc='upper right', fontsize=26)
    
    fig.legend(handles=patches, loc='lower center', ncol=3, bbox_to_anchor=(0.455, 0.02), fontsize=26)
    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.26)
    
    # plt.savefig('Error.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    # '''
    
    # fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 16))
    # t = []
    # force_ref_real_data = [[], []]
    # force_est_real_data = [[], []]
    # force_ref_sim_data  = [[], []]
    # force_est_sim_data  = [[], []]
    
    # force_ref_real_ani_z = axes[0].plot([], [], label='Measured Force (Real)', linestyle='-', color='red', linewidth=linewidth, animated=True)
    # force_ref_real_ani_x = []
    # force_est_real_ani_z = []
    # force_est_real_ani_x = []
    # force_ref_sim_ani_z  = []
    # force_ref_sim_ani_x  = []
    # force_est_sim_ani_z  = []
    # force_est_sim_ani_x  = []
    
    # mod_idx = 0
    
    # def init():
        
    #     return force_ref_real_ani_z, force_ref_real_ani_x, force_est_real_ani_z, force_est_real_ani_x,\
    #         force_ref_sim_ani_z, force_ref_sim_ani_x, force_est_sim_ani_z, force_est_sim_ani_x
    
    # def update(frame):
    #     t.append(frame)
    #     force_ref_real_ani_z.append(force_ref_mean_list[:frame, mod_idx, 1])
        
    #     return force_ref_real_ani_z, force_ref_real_ani_x, force_est_real_ani_z, force_est_real_ani_x,\
    #         force_ref_sim_ani_z, force_ref_sim_ani_x, force_est_sim_ani_z, force_est_sim_ani_x
    
    # ani = FuncAnimation(fig, update, frames=np.linspace(0, 3, 3000), init_func=init, blit=True)
    # plt.tight_layout()
    # plt.show()