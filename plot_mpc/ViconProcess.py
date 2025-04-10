import pandas as pd
import numpy as np

def read_csv(filepath):
    data = dict()
    trigger_idx = 0
    
    ### read force data
    force_cols = ['Force Frame', 'Force Sub Frame']
    
    for i in [3, 1, 0, 2]:
        force_cols.extend([f'Fx_{i+1}', f'Fy_{i+1}', f'Fz_{i+1}'])
        force_cols.extend([f'Mx_{i+1}', f'My_{i+1}', f'Mz_{i+1}'])
        force_cols.extend([f'Cx_{i+1}', f'Cy_{i+1}', f'Cz_{i+1}'])
    
    for col in force_cols:
        data[col] = []
    
    # open vicon data
    vicon_csv = open(filepath, 'r', encoding='utf-8')

    # read "Device"
    vicon_csv.readline()

    # read "Hz"
    vicon_csv.readline()

    # read "class"
    vicon_csv.readline()

    # read "sub class"
    vicon_csv.readline()

    # read "unit"
    vicon_csv.readline()
    
    while True:
        line = vicon_csv.readline().strip().split(',')
        if line == ['']: break
        
        for i in range(len(force_cols)):
            data[force_cols[i]].append(float(line[i]))
            
    ### read point data
    if vicon_csv.readline() == '':
        vicon_csv.close()
        return pd.DataFrame(data), trigger_idx
    
    # read "Hz"
    point_freq = int(vicon_csv.readline())
    
    points = vicon_csv.readline().strip().split(',')[2:]
    points = [points[3*i] for i in range((len(points))//3)]
    points = [col.split(':')[-1] for col in points]
    
    point_cols = ['Point Frame', 'Point Sub Frame']
    
    for col in points:
        point_cols.extend([f'{col}_x', f'{col}_y', f'{col}_z'])
    
    for col in point_cols:
        data[col] = []
    
    # read "sub class"
    vicon_csv.readline()

    # read "unit"
    vicon_csv.readline()
    
    print('Points:', points)
    
    row_idx = 0
    while True:
        line = vicon_csv.readline().strip().split(',')
        if line == ['']: break
        
        for i in range(len(point_cols)):
            if point_cols[i] == 'Trigger_x' or point_cols[i] == 'O5_x':
                if line[i] != '' and trigger_idx == 0:
                    trigger_idx = row_idx*int(1000/point_freq)
            for _ in range(1000//int(point_freq)):
                data[point_cols[i]].append(float(line[i]) if line[i] != '' else np.nan)

        row_idx += 1
    
    
    for key in data.keys():
        data[key] = np.array(data[key])
    
    data['vicon_pos_x'] = (data['O1_x'] + data['O2_x'] + data['O3_x'] + data['O4_x'])/4/1000
    data['vicon_pos_z'] = (data['O1_z'] + data['O2_z'] + data['O3_z'] + data['O4_z'])/4/1000
    
    data['vicon_vel_x'] = []
    data['vicon_vel_z'] = []
    
    for i in range(data['vicon_pos_x'].__len__()-point_freq):
        data['vicon_vel_x'].append((data['vicon_pos_x'][i+point_freq]-data['vicon_pos_x'][i])/0.1)
        data['vicon_vel_z'].append((data['vicon_pos_z'][i+point_freq]-data['vicon_pos_z'][i])/0.1)
        
    for i in range(point_freq):
        data['vicon_vel_x'].append(0)
        data['vicon_vel_z'].append(0)
    
    x_front = (data['O1_x']+data['O2_x'])/2
    x_hind  = (data['O3_x']+data['O4_x'])/2
    x_left  = (data['O1_x']+data['O4_x'])/2
    x_right = (data['O2_x']+data['O3_x'])/2
    
    y_front = (data['O1_y']+data['O2_y'])/2
    y_hind  = (data['O3_y']+data['O4_y'])/2
    y_left  = (data['O1_y']+data['O4_y'])/2
    y_right = (data['O2_y']+data['O3_y'])/2
    
    z_front = (data['O1_z']+data['O2_z'])/2
    z_hind  = (data['O3_z']+data['O4_z'])/2
    z_left  = (data['O1_z']+data['O4_z'])/2
    z_right = (data['O2_z']+data['O3_z'])/2
    
    data['vicon_roll'] = np.rad2deg(np.arctan2((z_left-z_right),np.sqrt((x_left-x_right)**2+(y_left-y_right)**2)))
    data['vicon_pitch'] = -np.rad2deg(np.arctan2((z_front-z_hind),np.sqrt((x_front-x_hind)**2+(y_front-y_hind)**2)))
        
    return pd.DataFrame(data), trigger_idx
    
    
if __name__ == '__main__':
    filepath = 'data/test/stair_3_vicon.csv'
    data_vicon, trigger_idx = read_csv(filepath)

    # print(data_vicon['Trigger_x'][trigger_idx])
    # print(trigger_idx)