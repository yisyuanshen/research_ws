from scipy.signal import butter,filtfilt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
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




if __name__ == '__main__':
    
    force_ref_list = []
    force_est_list = []
    det_jacobian_list = []
    # mode = 'Walk'
    # mode = 'Hybrid'
    mode = 'Wave'
    
    if mode == 'Walk': start_idx, end_idx = 2000, 5000+100
    if mode == 'Hybrid': start_idx, end_idx = 2000, 9500+100
    if mode == 'Wave': start_idx, end_idx = 0, 29000+100
    
    for i in range(10):
        print(f'Processing {i+1} ...')
        
        motor_data_path = f'{os.getcwd()}/Data/{mode}/motor_real_{i+1}.csv'
        force_data_path = f'{os.getcwd()}/Data/{mode}/force_real_{i+1}.csv'
        trq_list, phi_list, force_ref = read_data(motor_data_path, force_data_path, start_idx, end_idx)
        
        if mode == 'Wave':
            trq_list  = np.concatenate((trq_list [5000:6000], trq_list [16000:18000], trq_list [28000:29100]), axis=0)
            phi_list  = np.concatenate((phi_list [5000:6000], phi_list [16000:18000], phi_list [28000:29100]), axis=0)
            force_ref = np.concatenate((force_ref[5000:6000], force_ref[16000:18000], force_ref[28000:29100]), axis=0)

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
    
    from scipy.signal import correlate, correlation_lags
    from sklearn.metrics import mean_squared_error
    
    force_ref_mean_list_ = np.mean(force_ref_list, axis=0)[:, 2, 1]
    force_ref_std_list_ = np.std(force_ref_list, axis=0)
    force_est_mean_list_ = np.mean(force_est_list, axis=0)[2, :, 1]
    
    cross_corr = correlate(force_est_mean_list_, force_ref_mean_list_, mode='full')
    lags = correlation_lags(len(force_est_mean_list_), len(force_ref_mean_list_), mode='full')
    lag = lags[np.argmax(cross_corr)]
    print(lag)
    
    if lag > 0:
        aligned_est = np.roll(force_est_mean_list_, -lag)
    elif lag < 0:
        aligned_est = np.roll(force_est_mean_list_, -lag)
    else:
        aligned_est = force_est_mean_list_
        
    trim_size = abs(lag)
    if trim_size > 0:
        aligned_est = aligned_est[trim_size:-trim_size]
        force_ref_mean_list_ = force_ref_mean_list_[trim_size:-trim_size]

    rmse = np.sqrt(mean_squared_error(force_ref_mean_list_, aligned_est))
    print(f"RMSE: {rmse}")
        
        
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
    
    from scipy.signal import correlate, correlation_lags
    from sklearn.metrics import mean_squared_error

    force_ref_mean_list_ = force_ref_sim[:, 2, 1]
    force_est_mean_list_ = force_est_sim[2, :, 1]
    
    rmse = np.sqrt(mean_squared_error(force_ref_mean_list_, force_est_mean_list_))
    print(f"RMSE: {rmse}")
    
        
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{mathptmx}')
    plt.rcParams.update({'font.size': 22})
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    linewidth = 1.5
    
    '''
    fig = plt.figure(figsize=(12, 14))
    
    for fig_idx in range(4):
        ax = fig.add_subplot(4, 1, fig_idx+1)
        
    
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
        ax.set_title(title, fontsize=22)
        ax.grid(True)
        
        
        if fig_idx == 0:
            handles = [real_ref, sim_ref, real_est, sim_est, ax.collections[0]]
            labels = [h.get_label() for h in handles]
        
        fig_idx += 1
        
    
    fig.legend(handles, labels, loc='center', ncol=3, bbox_to_anchor=(0.5, 0.05), fontsize=22)
    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    
    # plt.savefig('Force.pdf', format='pdf', bbox_inches='tight')
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
    ax.legend(loc='upper left', fontsize=22)
    ax.grid(True)
    
    ax_ = ax.twinx()
    ax_.plot(np.array(range(det_jacobian_list[mod_idx].__len__()))/1000, det_jacobian_list[mod_idx], label=f'Determinants', linestyle='--', color='red', linewidth=linewidth*2)
    ax_.set_ylabel('Determinants of Jacobian Matrix')
    ax_.set_ylim(-0.01, 0)
    ax_.tick_params(axis='both', which='major')
    ax_.legend(loc='upper right', fontsize=22)
    
    fig.legend(handles=patches, loc='lower center', ncol=3, bbox_to_anchor=(0.455, 0.02), fontsize=22)
    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    
    # plt.savefig('Error.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    # '''
    
    mod_idx = 1
    linewidth = 2
    
    if mode == 'Walk':
        phase_offset = 11
        t = 3000
    if mode == 'Hybrid':
        phase_offset = 9
        t = 7500
    if mode == 'Wave':
        phase_offset = 19
        t = 4000
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 9))
    
    # freq = 1000
    # line_ref_z, = axes[0].plot(np.linspace(0, 3, t), force_ref_mean_list[phase_offset:t+phase_offset,mod_idx,1], label='Measured Force (Real)', linestyle='-', color='red', linewidth=linewidth)
    # line_est_z, = axes[0].plot(np.linspace(0, 3, t), force_est_mean_list[mod_idx,:t,1], label='Estimated Force (Real)', linestyle='--', color='blue', linewidth=linewidth)
    # line_ref_x, = axes[1].plot(np.linspace(0, 3, t), force_ref_mean_list[phase_offset:t+phase_offset,mod_idx,0], lw=2, linestyle='-', color='red')
    # line_est_x, = axes[1].plot(np.linspace(0, 3, t), force_est_mean_list[mod_idx,:t,0], lw=2, linestyle='--', color='blue')
    
    freq = 400
    phase_offset = 0
    line_ref_z, = axes[0].plot(np.linspace(0, 3, int(t*0.4)), force_ref_sim[:,mod_idx,1], label='Measured Force (Sim)', linestyle='-', color='green', linewidth=linewidth)
    line_est_z, = axes[0].plot(np.linspace(0, 3, int(t*0.4)), force_est_sim[mod_idx,:,1], label='Estimated Force (Sim)', linestyle='--', color='orange', linewidth=linewidth)
    line_ref_x, = axes[1].plot(np.linspace(0, 3, int(t*0.4)), force_ref_sim[:,mod_idx,0], lw=2, linestyle='-', color='green')
    line_est_x, = axes[1].plot(np.linspace(0, 3, int(t*0.4)), force_est_sim[mod_idx,:,0], lw=2, linestyle='--', color='orange')
    
    if mode == 'Wave': x_ticks = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    if mode == 'Walk': x_ticks = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    if mode == 'Hybrid': x_ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    axes[0].set_xticks(x_ticks)
    axes[0].set_xticklabels([f'{tick:.1f}' for tick in x_ticks])
    y_min, y_max = axes[0].get_ylim()
    y_min = (np.floor(y_min / 50) * 50)
    y_max = (np.ceil(y_max / 50) * 50)
    if y_min > -100 and y_max < 100: y_ticks = np.arange(y_min+25, y_max, 25)
    else: y_ticks = np.arange(y_min+50, y_max, 50)
    axes[0].set_yticks(y_ticks)
    
    axes[1].set_xticks(x_ticks)
    axes[1].set_xticklabels([f'{tick:.1f}' for tick in x_ticks])
    y_min, y_max = axes[1].get_ylim()
    y_min = (np.floor(y_min / 50) * 50)
    y_max = (np.ceil(y_max / 50) * 50)
    if y_min > -100 and y_max < 100: y_ticks = np.arange(y_min+25, y_max, 25)
    else: y_ticks = np.arange(y_min+50, y_max, 50)
    axes[1].set_yticks(y_ticks)
    
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Force (N)')
    axes[0].set_title('Right Front - Z Direction', fontsize=26)
    axes[0].grid(True)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Force (N)')
    axes[1].set_title('Right Front - X Direction', fontsize=26)
    axes[1].grid(True)
    
    def animate(frame):
        print(frame)
        # data_ref = force_ref_mean_list[phase_offset:phase_offset+frame, mod_idx, 1]
        # data_est = force_est_mean_list[mod_idx, :frame, 1]
        # line_ref_z.set_data(np.linspace(0, frame/freq, frame), data_ref)
        # line_est_z.set_data(np.linspace(0, frame/freq, frame), data_est)
        
        # data_ref = force_ref_mean_list[phase_offset:phase_offset+frame, mod_idx, 0]
        # data_est = force_est_mean_list[mod_idx, :frame,0]
        # line_ref_x.set_data(np.linspace(0, frame/freq, frame), data_ref)
        # line_est_x.set_data(np.linspace(0, frame/freq, frame), data_est)
        
        data_ref = force_ref_sim[:frame, mod_idx, 1]
        data_est = force_est_sim[mod_idx, :frame, 1]
        line_ref_z.set_data(np.linspace(0, frame/freq, frame), data_ref)
        line_est_z.set_data(np.linspace(0, frame/freq, frame), data_est)
        
        data_ref = force_ref_sim[:frame, mod_idx, 0]
        data_est = force_est_sim[mod_idx, :frame,0]
        line_ref_x.set_data(np.linspace(0, frame/freq, frame), data_ref)
        line_est_x.set_data(np.linspace(0, frame/freq, frame), data_est)
        
        return line_ref_z, line_est_z, line_ref_x, line_est_x
    
    handles = [line_ref_z, line_est_z]
    labels = [h.get_label() for h in handles]
    
    fig.legend(handles, labels, loc='center', ncol=2, bbox_to_anchor=(0.5, 0.05))
    fig.align_ylabels(axes)
    
    # ani = FuncAnimation(fig, animate, frames=t, interval=1, blit=False)
    ani = FuncAnimation(fig, animate, frames=int(t*0.4), interval=2.5, blit=False)
    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    
    # plt.show()
    
    fps = freq  # Frames per second
    writer = FFMpegWriter(fps=fps)

    # Save the animation before showing it
    ani.save('force.mp4', writer=writer)