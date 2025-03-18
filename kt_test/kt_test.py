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

def process_data(filename, trq_col, beta_col):
    
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
    
    for i in range(20000):
        if (abs(beta_data[15000+i]) > np.pi/9):
            trq_ref = 3*9.81*0.1*np.sin(beta_data[15000+i])
            data.append(trq_ref/y_fit[i])
        else:
            trq_ref = 0
        
        # trq_ref = 3*9.81*0.1*np.sin(beta_data[15000+i])
        
    # data = butter_lowpass_filter(raw_data=data, cutoff=100, fs=1000)

    plt.plot(data)
    
    print(round(np.average(data), 3), '\n')
    
    # plt.show()

if __name__ == '__main__':
    filename = 'kt_test_AR.csv'
    trq_col = 'state_trq_r_a'
    beta_col = 'state_beta_a'
    process_data(f'{os.getcwd()}/kt_test/{filename}', trq_col, beta_col)

    filename = 'kt_test_AL.csv'
    trq_col = 'state_trq_l_a'
    beta_col = 'state_beta_a'
    process_data(f'{os.getcwd()}/kt_test/{filename}', trq_col, beta_col)

    filename = 'kt_test_BR.csv'
    trq_col = 'state_trq_r_b'
    beta_col = 'state_beta_b'
    process_data(f'{os.getcwd()}/kt_test/{filename}', trq_col, beta_col)

    filename = 'kt_test_BL.csv'
    trq_col = 'state_trq_l_b'
    beta_col = 'state_beta_b'
    process_data(f'{os.getcwd()}/kt_test/{filename}', trq_col, beta_col)

    filename = 'kt_test_CR.csv'
    trq_col = 'state_trq_r_c'
    beta_col = 'state_beta_c'
    process_data(f'{os.getcwd()}/kt_test/{filename}', trq_col, beta_col)

    filename = 'kt_test_CL.csv'
    trq_col = 'state_trq_l_c'
    beta_col = 'state_beta_c'
    process_data(f'{os.getcwd()}/kt_test/{filename}', trq_col, beta_col)

    filename = 'kt_test_DR.csv'
    trq_col = 'state_trq_r_d'
    beta_col = 'state_beta_d'
    process_data(f'{os.getcwd()}/kt_test/{filename}', trq_col, beta_col)

    filename = 'kt_test_DL.csv'
    trq_col = 'state_trq_l_d'
    beta_col = 'state_beta_d'
    process_data(f'{os.getcwd()}/kt_test/{filename}', trq_col, beta_col)
    
    plt.show()