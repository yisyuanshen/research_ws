from LegModel import *
import pandas as pd
import numpy as np
from numpy.polynomial import Polynomial
import os
import matplotlib.pyplot as plt

legmodel = LegModel(sim=True)

# filefolder = 'corgi_ws/corgi_ros_ws/output_data'
filefolder = 'research_ws/data'
filename = 'transform_i90_v1_sh6_trq.csv'

df_data = pd.read_csv(os.path.join(os.getenv('HOME'), filefolder, filename))

eta_cmd = df_data[['cmd_theta_a', 'cmd_beta_a',
                   'cmd_theta_b', 'cmd_beta_b',
                   'cmd_theta_c', 'cmd_beta_c',
                   'cmd_theta_d', 'cmd_beta_d']].to_numpy()

eta_state = df_data[['state_theta_a', 'state_beta_a',
                     'state_theta_b', 'state_beta_b',
                     'state_theta_c', 'state_beta_c',
                     'state_theta_d', 'state_beta_d']].to_numpy()

trq_cmd = df_data[['cmd_trq_r_a', 'cmd_trq_l_a',
                   'cmd_trq_r_b', 'cmd_trq_l_b',
                   'cmd_trq_r_c', 'cmd_trq_l_c',
                   'cmd_trq_r_d', 'cmd_trq_l_d']].to_numpy()

trq_state = df_data[['state_trq_r_a', 'state_trq_l_a',
                     'state_trq_r_b', 'state_trq_l_b',
                     'state_trq_r_c', 'state_trq_l_c',
                     'state_trq_r_d', 'state_trq_l_d']].to_numpy()

force_cmd = df_data[['imp_cmd_Fx_a', 'imp_cmd_Fy_a',
                     'imp_cmd_Fx_b', 'imp_cmd_Fy_b',
                     'imp_cmd_Fx_c', 'imp_cmd_Fy_c',
                     'imp_cmd_Fx_d', 'imp_cmd_Fy_d']].to_numpy()

force_state = df_data[['force_Fx_a', 'force_Fy_a',
                       'force_Fx_b', 'force_Fy_b',
                       'force_Fx_c', 'force_Fy_c',
                       'force_Fx_d', 'force_Fy_d']].to_numpy()


for mod_idx in range(4):
    theta = eta_state[:, 2*mod_idx]
    beta = eta_state[:, 2*mod_idx+1]
    trq = trq_state[:, [2*mod_idx, 2*mod_idx+1]]
    
    legmodel.contact_map(theta, beta)
    
    ### Calculate poly at contact point
    cos_alpha = np.cos(legmodel.alpha)
    sin_alpha = np.sin(legmodel.alpha)
    rot_alpha = np.array([[cos_alpha, -sin_alpha], [sin_alpha, cos_alpha]]).T
    
    data_len = len(legmodel.rim)
    P_poly = np.zeros((data_len, 2, 8))
    scaled_radius = legmodel.radius / legmodel.R
    
    mask_1 = legmodel.rim == 1
    mask_2 = legmodel.rim == 2
    mask_3 = legmodel.rim == 3
    mask_4 = legmodel.rim == 4
    mask_5 = legmodel.rim == 5

    if np.any(mask_1):
        H_l_coef = np.array([H_l_poly[0].coef, H_l_poly[1].coef])
        U_l_coef = np.array([U_l_poly[0].coef, U_l_poly[1].coef])
        P_coef = np.dot(rot_alpha, H_l_coef - U_l_coef) * scaled_radius + U_l_coef
        P_poly[mask_1] = P_coef[mask_1]
        
    if np.any(mask_2):
        F_l_coef = np.array([F_l_poly[0].coef, F_l_poly[1].coef])
        L_l_coef = np.array([L_l_poly[0].coef, L_l_poly[1].coef])
        P_coef = np.dot(rot_alpha, F_l_coef - L_l_coef) * scaled_radius + L_l_coef
        P_poly[mask_2] = P_coef[mask_2]
        
    if np.any(mask_3):
        G_coef = np.array([np.zeros((8,)), G_poly[1].coef])
        r_coef = np.zeros((2, 8))
        r_coef[1, 0] = -legmodel.r
        P_coef = np.dot(rot_alpha, r_coef) + G_coef
        P_poly[mask_3] = P_coef[mask_3]

    if np.any(mask_4):
        G_coef = np.array([np.zeros((8,)), G_poly[1].coef])
        L_r_coef = np.array([L_r_poly[0].coef, L_r_poly[1].coef])
        P_coef = np.dot(rot_alpha, G_coef - L_r_coef) * scaled_radius + L_r_coef
        P_poly[mask_4] = P_coef[mask_4]
        
    if np.any(mask_5):
        UF_r_poly = F_r_poly - U_r_poly
        F_r_coef = np.array([F_r_poly[0].coef, F_r_poly[1].coef])
        U_r_coef = np.array([U_r_poly[0].coef, U_r_poly[1].coef])
        P_coef = np.dot(rot_alpha, F_r_coef - U_r_coef) * scaled_radius + U_r_coef
        P_poly[mask_5] = P_coef[mask_5]
        
    
    P_poly = np.array([[Polynomial(poly[0]), Polynomial(poly[1])] for poly in P_poly])    
    P_poly_deriv = np.array([[poly[0].deriv(), poly[1].deriv()] for poly in P_poly])

    ### Calculate Jacobian
    P_theta = np.array([[P_poly[i][0](theta[i]), P_poly[i][1](theta[i])] for i in range(data_len)])
    P_theta_deriv = np.array([[P_poly_deriv[i][0](theta[i]), P_poly_deriv[i][1](theta[i])] for i in range(data_len)])

    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)
    dtheta_dphiR = -0.5
    dtheta_dphiL =  0.5
    dbeta_dphiR  =  0.5
    dbeta_dphiL  =  0.5
    
    dPx_dtheta = P_theta_deriv[:, 0]*cos_beta - P_theta_deriv[:, 1]*sin_beta
    dPy_dtheta = P_theta_deriv[:, 0]*sin_beta + P_theta_deriv[:, 1]*cos_beta
    dPx_dbeta  = P_theta[:, 0]*(-sin_beta) - P_theta[:, 1]*cos_beta
    dPy_dbeta  = P_theta[:, 0]*cos_beta + P_theta[:, 1]*(-sin_beta)
    
    J11 = dPx_dtheta * dtheta_dphiR + dPx_dbeta * dbeta_dphiR
    J12 = dPx_dtheta * dtheta_dphiL + dPx_dbeta * dbeta_dphiL
    J21 = dPy_dtheta * dtheta_dphiR + dPy_dbeta * dbeta_dphiR
    J22 = dPy_dtheta * dtheta_dphiL + dPy_dbeta * dbeta_dphiL
    
    print(J11)
    print(J12)
    print(J21)
    print(J22)

    epsilon = 1e-6
    jacobian = np.array([[J11, J21], [J12, J22]]).T
    jacobian_inv_T = np.empty_like(jacobian)

    for i in range(jacobian.shape[0]):
        try: jacobian_inv_T[i] = np.linalg.inv(jacobian[i]).T
        except np.linalg.LinAlgError: jacobian_inv_T[i] = np.nan

    print(list(np.isnan(jacobian_inv_T).any(axis=(1, 2))).count(True))