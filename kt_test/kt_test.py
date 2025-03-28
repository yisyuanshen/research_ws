import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.optimize import curve_fit

def cosine_model(theta, a, b, c):
    return a * np.cos(b * theta + c)

def butter_lowpass_filter(raw_data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, raw_data)
    return y

def process_data(filename, trq_col, beta_col, friction):
    
    kt_data = pd.read_csv(filename)
    beta_data = kt_data[beta_col].to_numpy()
    trq_data = kt_data[trq_col].to_numpy()
    trq_data = butter_lowpass_filter(raw_data=trq_data, cutoff=1, fs=1000)
        
    segment = trq_data[15000:35000]
    theta = np.linspace(0, np.pi, len(segment))
    X = np.cos(theta).reshape(-1, 1)
    a_solution, residuals, rank, s = np.linalg.lstsq(X, segment, rcond=None)
    a = a_solution[0]
    y_fit = a * np.cos(theta)

    # plt.plot(theta, y_fit, '-', label=f'Fit: {a:.2f} cos(theta)')
    # print(a)
    
    data = []
    
    for i in range(12000, 14000, 1):
        # if (abs(beta_data[15000+i]) > np.pi/9):
        #     trq_ref = 3*9.81*0.1*np.sin(beta_data[15000+i])
        #     data.append(trq_ref/(y_fit[i]-friction*np.sign(y_fit[i])))
        # else:
        #     trq_ref = 0
        
        trq_ref = 3*9.81*0.1*np.sin(beta_data[i])
        data.append(trq_ref/(trq_data[i]-friction*np.sign(trq_data[i])))
        
        # trq_ref = 3*9.81*0.1*np.sin(beta_data[15000+i])
        
        
    for i in range(37000, 39000, 1):
        # if (abs(beta_data[15000+i]) > np.pi/9):
        #     trq_ref = 3*9.81*0.1*np.sin(beta_data[15000+i])
        #     data.append(trq_ref/(y_fit[i]-friction*np.sign(y_fit[i])))
        # else:
        #     trq_ref = 0
        
        trq_ref = 3*9.81*0.1*np.sin(beta_data[i])
        data.append(trq_ref/(trq_data[i]-friction*np.sign(trq_data[i])))
        
        
    # data = butter_lowpass_filter(raw_data=data, cutoff=100, fs=1000)

    plt.plot(data)
    
    print(round(np.average(data), 3), '\n')
    
    # plt.show()

if __name__ == '__main__':
    friction = [0.429, 0.272, 0.326, 0.267, 0.353, 0.288, 0.306, 0.236]
    # friction = [0.568, 0.334, 0.370, 0.330, 0.440, 0.356, 0.330, 0.359]
    
    
    
    filename = 'data/0320_kt/kt_test_AR.csv'
    trq_col = 'state_trq_r_a'
    beta_col = 'state_beta_a'
    process_data(f'{os.getcwd()}/kt_test/{filename}', trq_col, beta_col, friction[0])

    filename = 'data/0320_kt/kt_test_AL.csv'
    trq_col = 'state_trq_l_a'
    beta_col = 'state_beta_a'
    process_data(f'{os.getcwd()}/kt_test/{filename}', trq_col, beta_col, friction[1])

    filename = 'data/0320_kt/kt_test_BR.csv'
    trq_col = 'state_trq_r_b'
    beta_col = 'state_beta_b'
    process_data(f'{os.getcwd()}/kt_test/{filename}', trq_col, beta_col, friction[2])

    filename = 'data/0320_kt/kt_test_BL.csv'
    trq_col = 'state_trq_l_b'
    beta_col = 'state_beta_b'
    process_data(f'{os.getcwd()}/kt_test/{filename}', trq_col, beta_col, friction[3])

    filename = 'data/0320_kt/kt_test_CR.csv'
    trq_col = 'state_trq_r_c'
    beta_col = 'state_beta_c'
    process_data(f'{os.getcwd()}/kt_test/{filename}', trq_col, beta_col, friction[4])

    filename = 'data/0320_kt/kt_test_CL.csv'
    trq_col = 'state_trq_l_c'
    beta_col = 'state_beta_c'
    process_data(f'{os.getcwd()}/kt_test/{filename}', trq_col, beta_col, friction[5])

    filename = 'data/0320_kt/kt_test_DR.csv'
    trq_col = 'state_trq_r_d'
    beta_col = 'state_beta_d'
    process_data(f'{os.getcwd()}/kt_test/{filename}', trq_col, beta_col, friction[6])

    filename = 'data/0320_kt/kt_test_DL.csv'
    trq_col = 'state_trq_l_d'
    beta_col = 'state_beta_d'
    process_data(f'{os.getcwd()}/kt_test/{filename}', trq_col, beta_col, friction[7])
    
    plt.show()