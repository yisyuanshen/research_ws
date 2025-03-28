import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read all files
r_cw1  = pd.read_csv('data/0321_friction/0321_friction_R_cw.csv').iloc[:80000, :]
r_cw2  = pd.read_csv('data/0321_friction/0321_friction_R_cw_1.csv').iloc[:80000, :]
r_ccw1 = pd.read_csv('data/0321_friction/0321_friction_R_ccw.csv').iloc[:80000, :]
r_ccw2 = pd.read_csv('data/0321_friction/0321_friction_R_ccw_1.csv').iloc[:80000, :]
r_cw3  = pd.read_csv('data/0321_friction/0321_friction_R_cw_2.csv').iloc[:80000, :]
r_ccw3 = pd.read_csv('data/0321_friction/0321_friction_R_ccw_2.csv').iloc[:80000, :]
l_cw1  = pd.read_csv('data/0321_friction/0321_friction_L_cw.csv').iloc[:80000, :]
l_cw2  = pd.read_csv('data/0321_friction/0321_friction_L_cw_1.csv').iloc[:80000, :]
l_ccw1 = pd.read_csv('data/0321_friction/0321_friction_L_ccw.csv').iloc[:80000, :]
l_ccw2 = pd.read_csv('data/0321_friction/0321_friction_L_ccw_1.csv').iloc[:80000, :]
l_cw3  = pd.read_csv('data/0321_friction/0321_friction_L_cw_2.csv').iloc[:80000, :]
l_ccw3 = pd.read_csv('data/0321_friction/0321_friction_L_ccw_2.csv').iloc[:80000, :]
all_cw = pd.read_csv('data/0321_friction/0321_friction_cw.csv').iloc[:80000, :]
all_ccw = pd.read_csv('data/0321_friction/0321_friction_ccw.csv').iloc[:80000, :]
all_cw_ccw = pd.read_csv('data/0321_friction/0321_friction_cw_ccw.csv').iloc[:80000, :]
all_ccw_cw = pd.read_csv('data/0321_friction/0321_friction_ccw_cw.csv').iloc[:80000, :]

df_data = pd.concat([r_cw1, r_cw2, r_ccw1, r_ccw2, r_cw3, r_ccw3,
                     l_cw1, l_cw2, l_ccw1, l_ccw2, l_cw3, l_ccw3,
                     all_cw, all_ccw, all_cw_ccw, all_ccw_cw])

print(f'Data is loaded. Size = {df_data.__len__()}')

# extract timing
def process_data(module, motor):

    vel = df_data[f'state_vel_{motor}_{module}'].to_numpy()

    phi_l = df_data[f'state_theta_{module}']+df_data[f'state_beta_{module}']-np.deg2rad(17)
    phi_r = df_data[f'state_beta_{module}']-df_data[f'state_theta_{module}']+np.deg2rad(17)
    

    trq_cmd = df_data[f'cmd_trq_{motor}_{module}'].to_numpy()

    timing = [0]

    for idx in range(df_data.__len__()):
        if abs(vel[idx]) > 5 and idx - 1000 > timing[-1]:
            timing.append(idx)

    trq_thres = []
    data_1 = []
    data_2 = []
    
    for idx in timing[1:]:
        trq_thres.append(max(abs(trq_cmd[idx-1000:idx+1000])))
        # data_1.extend(trq_cmd[idx-3000:idx+500])
        # data_1.extend(vel[idx-1000:idx+500])
        data_1.extend(phi_l[idx-1000:idx+1000])
        data_2.extend(phi_r[idx-1000:idx+1000])
        
    # print(f'Motor {module.upper()}{motor.upper()}: {np.average(trq_thres)}')
    
    avg_trq = np.mean(trq_thres)
    median_trq = np.median(trq_thres)
    max_trq = np.max(trq_thres)
    std_trq = np.std(trq_thres)
    perc_90 = np.percentile(trq_thres, 90)
    perc_95 = np.percentile(trq_thres, 95)
    
    print(f"\nModule {module.upper()} Motor {motor.upper()} (Velocity-based):")
    print(f"  Average Torque: {avg_trq:.3f}")
    # print(f"  Median Torque:  {median_trq:.3f}")
    # print(f"  Max Torque:     {max_trq:.3f}")
    # print(f"  Std Deviation:  {std_trq:.3f}")
    # print(f"  90th Percentile:{perc_90:.3f}")
    print(f"  95th Percentile:{perc_95:.3f}")
    
    
    # plt.plot(range(len(data_1)), data_1)
    # plt.plot(range(len(data_2)), data_2)
    # plt.show()
    
    return avg_trq
    
    
if __name__ == '__main__':
    modules = ['a', 'b', 'c', 'd']  # a, b, c, d
    motors = ['r', 'l']  # r, l
    
    # modules = ['a']  # a, b, c, d
    # motors = ['r']  # r, l
    
    friction = []
    for module in modules:
        for motor in motors:
            friction.append(round(process_data(module, motor), 3))
            
    print('\nFriction Compensation:', friction)