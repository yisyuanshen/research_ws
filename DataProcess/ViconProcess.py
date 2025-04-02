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
    point_freq = vicon_csv.readline()
    
    if int(point_freq) == 1000:
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
                if point_cols[i] == 'O5_x' or point_cols[i] == 'O5_x':
                    if line[i] != '' and trigger_idx == 0:
                        trigger_idx = row_idx
                data[point_cols[i]].append(float(line[i]) if line[i] != '' else np.nan)

            row_idx += 1
            
        for key in data.keys():
            data[key] = np.array(data[key])
        
        data['vicon_pos_x'] = (data['O1_x'] + data['O2_x'] + data['O3_x'] + data['O4_x'])/4/1000
        data['vicon_pos_z'] = (data['O1_z'] + data['O2_z'] + data['O3_z'] + data['O4_z'])/4/1000
        
        data['vicon_vel_x'] = []
        data['vicon_vel_z'] = []
        
        for i in range(data['vicon_pos_x'].__len__()-100):
            data['vicon_vel_x'].append((data['vicon_pos_x'][i+100]-data['vicon_pos_x'][i])/0.1)
            data['vicon_vel_z'].append((data['vicon_pos_z'][i+100]-data['vicon_pos_z'][i])/0.1)
            
        for i in range(100):
            data['vicon_vel_x'].append(0)
            data['vicon_vel_z'].append(0)
        
        
        p1 = np.stack([data['O1_x'], data['O1_y'], data['O1_z']], axis=1)
        p2 = np.stack([data['O2_x'], data['O2_y'], data['O2_z']], axis=1)
        p3 = np.stack([data['O3_x'], data['O3_y'], data['O3_z']], axis=1)
        p4 = np.stack([data['O4_x'], data['O4_y'], data['O4_z']], axis=1)
        
        vec_x = p4 - p1  # body x 軸方向（右）
        vec_y = p2 - p1  # body y 軸方向（前）
        vec_z = np.cross(vec_x, vec_y)  # body z 軸方向（上）
        vec_z = vec_z / np.linalg.norm(vec_z, axis=1, keepdims=True)
        
        vec_x = vec_x / np.linalg.norm(vec_x, axis=1, keepdims=True)
        vec_y = np.cross(vec_z, vec_x)  # 確保 Y 軸與 X,Z 正交
        vec_y = vec_y / np.linalg.norm(vec_y, axis=1, keepdims=True)
        
        R = np.stack([vec_x, vec_y, vec_z], axis=2)  # 注意這裡 axis=2，(N,3,3)

        pitch = np.arcsin(-R[:, 2, 0])
        roll = np.arctan2(R[:, 2, 1], R[:, 2, 2])
        yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
        
        data['vicon_roll'] = np.degrees(roll)
        data['vicon_pitch'] = np.degrees(pitch)
        data['vicon_yaw'] = np.degrees(yaw)
        
    return pd.DataFrame(data), trigger_idx
    
    
if __name__ == '__main__':
    filepath = 'data/0402_mpc/vicon/mpc_7474_walk.csv'
    data_vicon, trigger_idx = read_csv(filepath)
    
    print(data_vicon['vicon_pos_x'])
    print(trigger_idx)