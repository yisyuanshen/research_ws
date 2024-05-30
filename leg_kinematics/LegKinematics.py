import numpy as np
from numpy.polynomial import Polynomial

# Initialization
R = 0.1

# Right Plane
Hx_coef = [-0.02986811, 0.10150953, -0.00033831, -0.00676546, -0.00761573, 0.00241217, 0.00006521, -0.00005435]
Hy_coef = [0.09842200, 0.02021071, -0.04976560, -0.00235200, 0.00299106, 0.00091636, -0.00032807, 0.00001824]
Fx_coef = [0.07922198, -0.00642995, -0.00212025, -0.02514313, 0.02091331, -0.00555113, 0.00002911, 0.00013153]
Fy_coef = [-0.04889044, -0.04099782, -0.05057615, 0.05336587, -0.02925680, 0.00442350, 0.00105720, -0.00025475]
Gx_coef = [0, 0, 0, 0, 0, 0, 0]
Gy_coef = [-0.08004471, -0.04301110, -0.10580851, 0.08885408, -0.03103048, -0.00111065, 0.00303460, -0.00046520]
Ux_coef = [-0.00966952, 0.03326879, -0.00141826, -0.00296346, -0.00086992, -0.00075178, 0.00037014, -0.00002506]
Uy_coef = [-0.00066899, 0.01477295, -0.04975541, 0.02979176, -0.01978453, 0.00452661, 0.00037298, -0.00016160]
Lx_coef = [0.00620571, -0.00537372, -0.06028321, 0.02548079, 0.00855482, -0.00870958, 0.00213482, -0.00015989]
Ly_coef = [0.02047847, -0.04889900, -0.08046577, 0.04415016, -0.00771927, -0.00429579, 0.00207711, -0.00021894]

inv_Gy_coef = [-5.95900136, -198.86115887, -2844.17944295, -23374.36177286, -113384.68778569, -325133.48971168, -511744.25427474, -342037.77994577]
inv_U_len_coef = [0.29524048, 31.24201687, -211.52772499, -399.43288897, 27998.38121498, -261547.09889344, 1067959.76233762, -1657663.98172204]
inv_L_len_coef = [0.29530550, 11.02411426, -72.41431581, 986.69737154, -8170.91611485, 38941.64890575, -98163.01663375, 101568.42569989]

Ux_poly = Polynomial(Ux_coef)
Hx_poly = Polynomial(Hx_coef)
Hy_poly = Polynomial(Hy_coef)
Fx_poly = Polynomial(Fx_coef)
Fy_poly = Polynomial(Fy_coef)
Gx_poly = Polynomial(Gx_coef)
Gy_poly = Polynomial(Gy_coef)
Ux_poly = Polynomial(Ux_coef)
Uy_poly = Polynomial(Uy_coef)
Lx_poly = Polynomial(Lx_coef)
Ly_poly = Polynomial(Ly_coef)


def rot_matrix(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle),  np.cos(angle)]])


def get_alpha(theta, beta):
    rot_beta = rot_matrix(beta)
    
    H_r = rot_beta @ np.array([Hx_poly(theta), Hy_poly(theta)])
    F_r = rot_beta @ np.array([Fx_poly(theta), Fy_poly(theta)])
    U_r = rot_beta @ np.array([Ux_poly(theta), Uy_poly(theta)])
    L_r = rot_beta @ np.array([Lx_poly(theta), Ly_poly(theta)])
    G   = rot_beta @ np.array([Gx_poly(theta), Gy_poly(theta)])
    H_l = rot_beta @ np.array([-Hx_poly(theta), Hy_poly(theta)])
    F_l = rot_beta @ np.array([-Fx_poly(theta), Fy_poly(theta)])
    U_l = rot_beta @ np.array([-Ux_poly(theta), Uy_poly(theta)])
    L_l = rot_beta @ np.array([-Lx_poly(theta), Ly_poly(theta)])
    
    contact_height = []
    contact_height.append(U_r[1]-R if H_r[0] >= U_r[0] >= F_r[0] else 0)
    contact_height.append(L_r[1]-R if F_r[0] >= L_r[0] >= G[0] else 0)
    contact_height.append(G[1])
    contact_height.append(L_l[1]-R if G[0] >= L_l[0] >= F_l[0] else 0)
    contact_height.append(U_l[1]-R if F_l[0] >= U_l[0] >= H_l[0] else 0)
    
    if min(contact_height) > -0.09: return 0
    
    contact_rim = np.argmin(contact_height)
    
    if contact_rim == 0:
        return -np.pi/2 - np.arctan2(F_r[1]-U_r[1], F_r[0]-U_r[0]) + np.deg2rad(50)
    
    if contact_rim == 1:
        return -np.pi/2 - np.arctan2(G[1]-L_r[1], G[0]-L_r[0])
    
    if contact_rim == 2:
        return 0
    
    if contact_rim == 3:
        return -np.pi/2 - np.arctan2(G[1]-L_l[1], G[0]-L_l[0])
    
    if contact_rim == 4:
        return -np.pi/2 - np.arctan2(F_l[1]-U_l[1], F_l[0]-U_l[0]) - np.deg2rad(50)
    
    
def get_jacobian(theta, beta, alpha):
    if alpha > np.deg2rad(50):
        print('Right Upper Rim')
        P = rot_matrix(alpha-np.deg2rad(50)) @ np.array([Fx_poly-Ux_poly, Fy_poly-Uy_poly])
        P += np.array([Ux_poly, Uy_poly])
        
    elif alpha > 0:
        print('Right Lower Rim')
        P = rot_matrix(alpha) @ np.array([Gx_poly-Lx_poly, Gy_poly-Ly_poly])
        P += np.array([Lx_poly, Ly_poly])
        
    elif alpha < -np.deg2rad(50):
        print('Left Upper Rim')
        P = rot_matrix(alpha+np.deg2rad(50)) @ np.array([-Fx_poly+Ux_poly, Fy_poly-Uy_poly])
        P += np.array([-Ux_poly, Uy_poly])
             
    elif alpha < 0:
        print('Left Lower Rim')
        P = rot_matrix(alpha) @ np.array([-Gx_poly+Lx_poly, Gy_poly-Ly_poly])
        P += np.array([-Lx_poly, Ly_poly])

    else:
        print('G')
        P = np.array([Gx_poly, Gy_poly])
        
    P_deriv = [P[0].deriv(), P[1].deriv()]
    
    c_beta = np.cos(beta)
    s_beta = np.sin(beta)
    P[0] = P[0](theta)
    P[1] = P[1](theta)
    P_deriv[0] = P_deriv[0](theta)
    P_deriv[1] = P_deriv[1](theta)
    
    jacobian = np.array([[( c_beta*P_deriv[0]-s_beta*P[0]-s_beta*P_deriv[1]-c_beta*P[1])/2,
                          (-c_beta*P_deriv[0]-s_beta*P[0]+s_beta*P_deriv[1]-c_beta*P[1])/2],
                         [( s_beta*P_deriv[0]+c_beta*P[0]+c_beta*P_deriv[1]-s_beta*P[1])/2,
                          (-s_beta*P_deriv[0]+c_beta*P[0]-c_beta*P_deriv[1]-s_beta*P[1])/2]])
    
    return jacobian
    
    
if __name__ == '__main__':
    phi = [2.844888, 1.343904]
    torque = [-0.879571, -0.395425]
    
    theta = (phi[0] - phi[1]) / 2 + np.deg2rad(17)
    beta = (phi[0] + phi[1]) / 2
    
    alpha = get_alpha(theta, beta)
    print(f'Theta = {round(np.rad2deg(theta), 4)}, Beta = {round(np.rad2deg(beta))}, Alpha = {round(np.rad2deg(alpha), 4)}')
    
    jacobian = get_jacobian(theta, beta, alpha)
    
    force = np.linalg.inv(jacobian).T @ np.array(torque)
    print(f'Force = [{round(force[0], 4)}, {round(force[1], 4)}]')
    