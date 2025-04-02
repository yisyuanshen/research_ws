import pandas as pd

def read_csv(filepath, start_idx = 0):
    data = dict()
    force_cols = []

    force_cols.extend(['Frame', 'Sub Frame'])
    
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

    # start reading data from force plate
    for i in range(start_idx):
        vicon_csv.readline()
        
    while True:
        line = vicon_csv.readline().strip().split(',')
        if line == ['']: break
        
        for i in range(len(force_cols)):
            data[force_cols[i]].append(float(line[i]))
        
    vicon_csv.close()
    
    return pd.DataFrame(data)


if __name__ == '__main__':
    
    vicon_filepath = 'data/0206/vicon/imp_test.csv'
    
    print(read_csv(filepath=vicon_filepath))
    