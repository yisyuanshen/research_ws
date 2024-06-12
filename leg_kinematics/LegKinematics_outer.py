import numpy as np
from numpy.polynomial import Polynomial

# Right Plane
Gx_coef = [0, 0, 0, 0, 0, 0, 0]
Gy_coef = [-0.09005029, -0.04838749, -0.11903457, 0.09996084, -0.03490929, -0.00124948, 0.00341392, -0.00052335]

Hx_coef = [-0.03239293, 0.11003963, -0.00020331, -0.00724071, -0.00845895, 0.00280766, 0.00002709, -0.00005802]
Hy_coef = [0.11080837, 0.02089043, -0.04976687, -0.00636997, 0.00583801, 0.00046508, -0.00041570, 0.00004072]
Fx_upper_coef = [0.09033342, -0.01139229, -0.00220800, -0.02791559, 0.02363622, -0.00615105, -0.00001352, 0.00015111]
Fy_upper_coef = [-0.05491812, -0.04796917, -0.05067874, 0.05631264, -0.03044084, 0.00441062, 0.00114273, -0.00026639]

Fx_lower_coef = [0.08834901, -0.00656198, 0.00515012, -0.03147112, 0.02245812, -0.00515632, -0.00023411, 0.00016796]
Fy_lower_coef = [-0.05756155, -0.04001018, -0.04683994, 0.05451784, -0.03194900, 0.00551342, 0.00092971, -0.00025922]
Gx_lower_coef = [-0.00077571, 0.00067171, 0.00753540, -0.00318510, -0.00106935, 0.00108870, -0.00026685, 0.00001999]
Gy_lower_coef = [-0.09261010, -0.04227511, -0.10897635, 0.09444207, -0.03394438, -0.00071251, 0.00315428, -0.00049599]

Ux_coef = [-0.00966952, 0.03326879, -0.00141826, -0.00296346, -0.00086992, -0.00075178, 0.00037014, -0.00002506]
Uy_coef = [-0.00066899, 0.01477295, -0.04975541, 0.02979176, -0.01978453, 0.00452661, 0.00037298, -0.00016160]
Lx_coef = [0.00620571, -0.00537372, -0.06028321, 0.02548079, 0.00855482, -0.00870958, 0.00213482, -0.00015989]
Ly_coef = [0.02047847, -0.04889900, -0.08046577, 0.04415016, -0.00771927, -0.00429579, 0.00207711, -0.00021894]

inv_Gy_coef = [-5.95900136, -176.76547455, -2247.25289320, -16416.56135490, -70785.50238850, -180425.98843145, -252428.18260691, -149970.70110426]
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
    
    R = 0.1
    contact_height = []
    contact_height.append(U_r[1]-R if H_r[0] >= U_r[0] >= F_r[0] else 0)
    contact_height.append(L_r[1]-R if F_r[0] >= L_r[0] >= G[0] else 0)
    contact_height.append(G[1])
    contact_height.append(L_l[1]-R if G[0] >= L_l[0] >= F_l[0] else 0)
    contact_height.append(U_l[1]-R if F_l[0] >= U_l[0] >= H_l[0] else 0)
    
    if min(contact_height) > -0.09: return 0
    
    contact_rim = np.argmin(contact_height)
    
    if contact_rim == 0:
        return -np.pi/2 - np.arctan2(F_r[1]-U_r[1], F_r[0]-U_r[0]) + np.deg2rad(50), contact_rim
    
    if contact_rim == 1:
        return -np.pi/2 - np.arctan2(G[1]-L_r[1], G[0]-L_r[0]), contact_rim
    
    if contact_rim == 2:
        return 0, contact_rim
    
    if contact_rim == 3:
        return -np.pi/2 - np.arctan2(G[1]-L_l[1], G[0]-L_l[0]), contact_rim
    
    if contact_rim == 4:
        return -np.pi/2 - np.arctan2(F_l[1]-U_l[1], F_l[0]-U_l[0]) - np.deg2rad(50), contact_rim
    
    
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
    
    # print(c_beta, s_beta)
    # print(P[0], P[1])
    # print(P_deriv[0], P_deriv[1])
    
    jacobian = np.array([[( c_beta*P_deriv[0]-s_beta*P[0]-s_beta*P_deriv[1]-c_beta*P[1])/2,
                          (-c_beta*P_deriv[0]-s_beta*P[0]+s_beta*P_deriv[1]-c_beta*P[1])/2],
                         [( s_beta*P_deriv[0]+c_beta*P[0]+c_beta*P_deriv[1]-s_beta*P[1])/2,
                          (-s_beta*P_deriv[0]+c_beta*P[0]-c_beta*P_deriv[1]-s_beta*P[1])/2]])
    
    # print(np.linalg.inv(jacobian).T)
    
    return jacobian
    
    
if __name__ == '__main__':
    phi = [0, 0]
    trq = [0, 0]
    
    theta = (phi[0] - phi[1]) / 2 + np.deg2rad(17)
    beta = (phi[0] + phi[1]) / 2
    
    alpha, contact_rim = get_alpha(theta, beta)
    print(f'Theta = {round(np.rad2deg(theta), 4)}, Beta = {round(np.rad2deg(beta))}, Alpha = {round(np.rad2deg(alpha), 4)}')
    
    jacobian = get_jacobian(theta, beta, alpha)
    
    force = np.linalg.inv(jacobian).T @ np.array(trq)
    print(f'Force = [{round(force[0], 4)}, {round(force[1], 4)}]')
    