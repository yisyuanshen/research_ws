import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

A_x = [1.1298451515297917e-05, -0.08009406816901488, 0.00031077987117683923, 0.01279780586116366, 0.000529623721428407, -0.0009747107444189707, 0.00010114656462683195, 3.9686105149307733e-07]
A_y = [0.07999814822753314, 1.7151854512279386e-05, -0.04006472074189201, 0.00013130326225181847, 0.0031745271743078922, 0.00011925171552418127, -0.00016662714604901788, 1.5155512108629479e-05]
B_x = [1.4123064393961495e-05, -0.10011758521126767, 0.00038847483896963685, 0.01599725732645494, 0.0006620296517862264, -0.0012183884305244416, 0.0001264332057838159, 4.960763143282633e-07]
B_y = [0.09999768528441662, 2.1439818138853396e-05, -0.050080900927361914, 0.0001641290778146673, 0.003968158967880156, 0.00014906464440965692, -0.00020828393256283677, 1.8944390135981884e-05]
C_x = [-0.08730528366526673, -0.006032982958712443, 0.0019343336102733166, 0.029559693081384952, -0.02240662249085035, 0.005424783364943832, 0.00010570401494254007, -0.00014747932346906734]
C_y = [-0.029355110899534374, -0.04803283842822402, -0.050704451237005635, 0.04823975574127813, -0.02452120341367821, 0.003428094940132397, 0.0009713007194812254, -0.00022015129326877995]
D_x = [3.476446620244001e-06, -0.0246443286673903, 9.562457574964824e-05, 0.003937786418815912, 0.00016296114505717432, -0.0002999109982834234, 3.1122019885248306e-05, 1.2211109276800333e-07]
D_y = [-0.011598137598993386, 0.012042262341325568, -0.057280655639413064, 0.04730077753925416, -0.03352308806304545, 0.010133081111101545, -0.0007682202869026333, -6.530398105226846e-05]
E_x = [2.096498902174112e-16, -1.5626532487718923e-15, 4.038359189262104e-15, -4.883944906039412e-15, 3.0024987769201344e-15, -9.185005768970968e-16, 1.1703549450550424e-16, -2.3097498743265688e-18]
E_y = [-0.05230759796633858, 0.01738675589102002, -0.06493218226053267, 0.06826498832903205, -0.04983313927964629, 0.014583671953580126, -0.001035595016170874, -0.00010106375579044414]
F_x = [-0.0792219724795395, 0.0064299123886048925, 0.0021203529083785215, 0.025142981476038564, -0.02091319282305514, 0.005551074453531046, -2.909495137396668e-05, -0.0001315343091315485]
F_y = [-0.048890447488738886, -0.04099772768916559, -0.05057644869237089, 0.05336634225307657, -0.029257204510924468, 0.0044236934283603795, 0.001057154734805929, -0.0002547437937505522]
G_x = [6.036916649469257e-16, -4.50418527570068e-15, 1.2007613132616358e-14, -1.539199024277966e-14, 1.0429995300919045e-14, -3.779602603732299e-15, 6.800786341117492e-16, -4.6328928134032563e-17]
G_y = [-0.08004472102357749, -0.04301098228831576, -0.10580886558171504, 0.08885462983281482, -0.031030946899122565, -0.0011104345458026294, 0.0030345439893201655, -0.00046519823616068915]
H_x = [0.02986810722712902, -0.10150954623155305, 0.0003383466731623498, 0.00676539672044445, 0.0076157819218680145, -0.0024121975766720574, -6.520433869059816e-05, 5.435360544323393e-05]
H_y = [0.09842199731432363, 0.020210693889343143, -0.049765558993830575, -0.0023520566708253796, 0.0029911101350712636, 0.0009163396527568648, -0.00032806385798062184, 1.8236342494771682e-05]
U_x = [0.009669527950396465, -0.03326883905865388, 0.0014184118289400683, 0.0029632310390750177, 0.0008701125904072712, 0.0007516854757230747, -0.00037011866405118227, 2.505601218368707e-05]
U_y = [-0.0006689948851870158, 0.014772981185930982, -0.049755522265921545, 0.02979194704609264, -0.01978468688936552, 0.0045266839063890915, 0.000372964480330343, -0.00016159420469014]
L_x = [-0.006205708640314474, 0.005373675404611693, 0.06028332658202664, -0.02548094841787018, -0.008554694998187833, 0.00870952509655963, -0.0021348099461122, 0.00015989334953469552]
L_y = [0.02047844986323809, -0.04889885079656741, -0.08046621287998217, 0.044150837154095485, -0.007719832335489967, -0.0042955293496684385, 0.0020770465243536407, -0.00021893289684746635]

A_l_poly = np.array([ np.polynomial.Polynomial(A_x), np.polynomial.Polynomial(A_y)])
A_r_poly = np.array([-np.polynomial.Polynomial(A_x), np.polynomial.Polynomial(A_y)])
B_l_poly = np.array([ np.polynomial.Polynomial(B_x), np.polynomial.Polynomial(B_y)])
B_r_poly = np.array([-np.polynomial.Polynomial(B_x), np.polynomial.Polynomial(B_y)])
C_l_poly = np.array([ np.polynomial.Polynomial(C_x), np.polynomial.Polynomial(C_y)])
C_r_poly = np.array([-np.polynomial.Polynomial(C_x), np.polynomial.Polynomial(C_y)])
D_l_poly = np.array([ np.polynomial.Polynomial(D_x), np.polynomial.Polynomial(D_y)])
D_r_poly = np.array([-np.polynomial.Polynomial(D_x), np.polynomial.Polynomial(D_y)])
E_poly   = np.array([ np.polynomial.Polynomial(E_x), np.polynomial.Polynomial(E_y)])
F_l_poly = np.array([ np.polynomial.Polynomial(F_x), np.polynomial.Polynomial(F_y)])
F_r_poly = np.array([-np.polynomial.Polynomial(F_x), np.polynomial.Polynomial(F_y)])
G_poly   = np.array([ np.polynomial.Polynomial(G_x), np.polynomial.Polynomial(G_y)])
H_l_poly = np.array([ np.polynomial.Polynomial(H_x), np.polynomial.Polynomial(H_y)])
H_r_poly = np.array([-np.polynomial.Polynomial(H_x), np.polynomial.Polynomial(H_y)])
U_l_poly = np.array([ np.polynomial.Polynomial(U_x), np.polynomial.Polynomial(U_y)])
U_r_poly = np.array([-np.polynomial.Polynomial(U_x), np.polynomial.Polynomial(U_y)])
L_l_poly = np.array([ np.polynomial.Polynomial(L_x), np.polynomial.Polynomial(L_y)])
L_r_poly = np.array([-np.polynomial.Polynomial(L_x), np.polynomial.Polynomial(L_y)])

A_l_poly_deriv = np.array([poly.deriv() for poly in A_l_poly])
A_r_poly_deriv = np.array([poly.deriv() for poly in A_r_poly])
B_l_poly_deriv = np.array([poly.deriv() for poly in B_l_poly])
B_r_poly_deriv = np.array([poly.deriv() for poly in B_r_poly])
C_l_poly_deriv = np.array([poly.deriv() for poly in C_l_poly])
C_r_poly_deriv = np.array([poly.deriv() for poly in C_r_poly])
D_l_poly_deriv = np.array([poly.deriv() for poly in D_l_poly])
D_r_poly_deriv = np.array([poly.deriv() for poly in D_r_poly])
E_poly_deriv   = np.array([poly.deriv() for poly in E_poly  ])
F_l_poly_deriv = np.array([poly.deriv() for poly in F_l_poly])
F_r_poly_deriv = np.array([poly.deriv() for poly in F_r_poly])
G_poly_deriv   = np.array([poly.deriv() for poly in G_poly  ])
H_l_poly_deriv = np.array([poly.deriv() for poly in H_l_poly])
H_r_poly_deriv = np.array([poly.deriv() for poly in H_r_poly])
U_l_poly_deriv = np.array([poly.deriv() for poly in U_l_poly])
U_r_poly_deriv = np.array([poly.deriv() for poly in U_r_poly])
L_l_poly_deriv = np.array([poly.deriv() for poly in L_l_poly])
L_r_poly_deriv = np.array([poly.deriv() for poly in L_r_poly])


def get_jacobian(theta, beta):
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)
    rot_beta = np.array([[cos_beta, -sin_beta],[sin_beta, cos_beta]]).transpose(2, 0, 1)

    F_l = np.einsum('ijk,ik->ij', rot_beta, np.array([F_l_poly[0](theta), F_l_poly[1](theta)]).transpose())
    F_r = np.einsum('ijk,ik->ij', rot_beta, np.array([F_r_poly[0](theta), F_r_poly[1](theta)]).transpose())
    G   = np.einsum('ijk,ik->ij', rot_beta, np.array([G_poly  [0](theta), G_poly  [1](theta)]).transpose())
    H_l = np.einsum('ijk,ik->ij', rot_beta, np.array([H_l_poly[0](theta), H_l_poly[1](theta)]).transpose())
    H_r = np.einsum('ijk,ik->ij', rot_beta, np.array([H_r_poly[0](theta), H_r_poly[1](theta)]).transpose())
    U_l = np.einsum('ijk,ik->ij', rot_beta, np.array([U_l_poly[0](theta), U_l_poly[1](theta)]).transpose())
    U_r = np.einsum('ijk,ik->ij', rot_beta, np.array([U_r_poly[0](theta), U_r_poly[1](theta)]).transpose())
    L_l = np.einsum('ijk,ik->ij', rot_beta, np.array([L_l_poly[0](theta), L_l_poly[1](theta)]).transpose())
    L_r = np.einsum('ijk,ik->ij', rot_beta, np.array([L_r_poly[0](theta), L_r_poly[1](theta)]).transpose())
    
    R = 0.1
    eps = 10e-6
    contact_height = np.array([np.where((H_l[:, 0] - eps < U_l[:, 0]) & (U_l[:, 0] < F_l[:, 0] + eps), U_l[:, 1] - R + 0.0045, 0),
                               np.where((F_l[:, 0] - eps < L_l[:, 0]) & (L_l[:, 0] < G[:, 0] + eps), L_l[:, 1] - R, 0),
                               G[:, 1],
                               np.where((G[:, 0] - eps < L_r[:, 0]) & (L_r[:, 0] < F_r[:, 0] + eps), L_r[:, 1] - R, 0),
                               np.where((F_r[:, 0] - eps < U_r[:, 0]) & (U_r[:, 0] < H_r[:, 0] + eps), U_r[:, 1] - R + 0.0045, 0)]).transpose()
    
    contact_rim = np.where(np.min(contact_height, axis=1) >= -eps, None, np.argmin(contact_height, axis=1))
    
    # print(contact_rim)
    
    F_l_theta = np.array([F_l_poly[0](theta), F_l_poly[1](theta)]).transpose()
    F_r_theta = np.array([F_r_poly[0](theta), F_r_poly[1](theta)]).transpose()
    G_theta   = np.array([G_poly[0](theta), G_poly[1](theta)]).transpose()
    U_l_theta = np.array([U_l_poly[0](theta), U_l_poly[1](theta)]).transpose()
    U_r_theta = np.array([U_r_poly[0](theta), U_r_poly[1](theta)]).transpose()
    L_l_theta = np.array([L_l_poly[0](theta), L_l_poly[1](theta)]).transpose()
    L_r_theta = np.array([L_r_poly[0](theta), L_r_poly[1](theta)]).transpose()
    
    F_l_deriv_theta = np.array([F_l_poly_deriv[0](theta), F_l_poly_deriv[1](theta)]).transpose()
    F_r_deriv_theta = np.array([F_r_poly_deriv[0](theta), F_r_poly_deriv[1](theta)]).transpose()
    G_deriv_theta   = np.array([G_poly_deriv[0](theta),   G_poly_deriv[1](theta)]).transpose()
    U_l_deriv_theta = np.array([U_l_poly_deriv[0](theta), U_l_poly_deriv[1](theta)]).transpose()
    U_r_deriv_theta = np.array([U_r_poly_deriv[0](theta), U_r_poly_deriv[1](theta)]).transpose()
    L_l_deriv_theta = np.array([L_l_poly_deriv[0](theta), L_l_poly_deriv[1](theta)]).transpose()
    L_r_deriv_theta = np.array([L_r_poly_deriv[0](theta), L_r_poly_deriv[1](theta)]).transpose()
    
    
    alpha = np.zeros(contact_rim.shape)
    P = G_theta.copy()
    P_deriv = G_deriv_theta.copy()
    
    contact_0 = (contact_rim == 0)
    UF_l = F_l[contact_0]-U_l[contact_0]
    alpha[contact_0] = -np.pi/2 - np.arctan2(UF_l[:,1], UF_l[:,0]) - np.deg2rad(50)
    cos_alpha = np.cos(alpha[contact_0]+np.deg2rad(50))
    sin_alpha = np.sin(alpha[contact_0]+np.deg2rad(50))
    rot_alpha = np.array([[cos_alpha, -sin_alpha],[sin_alpha, cos_alpha]]).transpose(2, 0, 1)
    P[contact_0] = np.einsum('ijk,ik->ij', rot_alpha, (F_l_theta[contact_0]-U_l_theta[contact_0])) + U_l_theta[contact_0]
    P_deriv[contact_0] = np.einsum('ijk,ik->ij', rot_alpha, (F_l_deriv_theta[contact_0]-U_l_deriv_theta[contact_0])) + U_l_deriv_theta[contact_0]
    
    contact_1 = (contact_rim == 1)
    LF_l = F_l[contact_1]-L_l[contact_1]
    alpha[contact_1] = -np.pi/2 - np.arctan2(LF_l[:,1], LF_l[:,0]) - np.deg2rad(50)
    cos_alpha = np.cos(alpha[contact_1]+np.deg2rad(50))
    sin_alpha = np.sin(alpha[contact_1]+np.deg2rad(50))
    rot_alpha = np.array([[cos_alpha, -sin_alpha],[sin_alpha, cos_alpha]]).transpose(2, 0, 1)
    P[contact_1] = np.einsum('ijk,ik->ij', rot_alpha, (F_l_theta[contact_1]-L_l_theta[contact_1])) + L_l_theta[contact_1]
    P_deriv[contact_1] = np.einsum('ijk,ik->ij', rot_alpha, (F_l_deriv_theta[contact_1]-L_l_deriv_theta[contact_1])) + L_l_deriv_theta[contact_1]
    
    contact_3 = (contact_rim == 3)
    LF_r = F_r[contact_3]-L_r[contact_3]
    alpha[contact_3] = -np.pi/2 - np.arctan2(LF_r[:,1], LF_r[:,0]) + np.deg2rad(50)
    cos_alpha = np.cos(alpha[contact_3]-np.deg2rad(50))
    sin_alpha = np.sin(alpha[contact_3]-np.deg2rad(50))
    rot_alpha = np.array([[cos_alpha, -sin_alpha],[sin_alpha, cos_alpha]]).transpose(2, 0, 1)
    P[contact_3] = np.einsum('ijk,ik->ij', rot_alpha, (F_r_theta[contact_3]-L_r_theta[contact_3])) + L_r_theta[contact_3]
    P_deriv[contact_3] = np.einsum('ijk,ik->ij', rot_alpha, (F_r_deriv_theta[contact_3]-L_r_deriv_theta[contact_3])) + L_r_deriv_theta[contact_3]
    
    contact_4 = (contact_rim == 4)
    UF_r = F_r[contact_4]-U_r[contact_4]
    alpha[contact_4] = -np.pi/2 - np.arctan2(UF_r[:,1], UF_r[:,0]) + np.deg2rad(50)
    cos_alpha = np.cos(alpha[contact_4]-np.deg2rad(50))
    sin_alpha = np.sin(alpha[contact_4]-np.deg2rad(50))
    rot_alpha = np.array([[cos_alpha, -sin_alpha],[sin_alpha, cos_alpha]]).transpose(2, 0, 1)
    P[contact_4] = np.einsum('ijk,ik->ij', rot_alpha, (F_r_theta[contact_4]-U_r_theta[contact_4])) + U_r_theta[contact_4]
    P_deriv[contact_4] = np.einsum('ijk,ik->ij', rot_alpha, (F_r_deriv_theta[contact_4]-U_r_deriv_theta[contact_4])) + U_r_deriv_theta[contact_4]
    
    # print(P)
    # print(P_deriv)
    # print(-sin_beta*P[:,0]-cos_beta*P[:,1])
    # print(np.rad2deg(beta).tolist()[::100])
    
    jacobian = np.array([[0.5*(-sin_beta*P[:,0]-cos_beta*P[:,1]) - 0.5*(cos_beta*P_deriv[:,0]-sin_beta*P_deriv[:,1]),
                          0.5*(-sin_beta*P[:,0]-cos_beta*P[:,1]) + 0.5*(cos_beta*P_deriv[:,0]-sin_beta*P_deriv[:,1])],
                         [0.5*( cos_beta*P[:,0]-sin_beta*P[:,1]) - 0.5*(sin_beta*P_deriv[:,0]+cos_beta*P_deriv[:,1]),
                          0.5*( cos_beta*P[:,0]-sin_beta*P[:,1]) + 0.5*(sin_beta*P_deriv[:,0]+cos_beta*P_deriv[:,1])]])
    
    jacobian = jacobian.transpose(2, 0, 1)
    
    return jacobian, contact_rim
  

def get_link_force():
    return -np.array([0, (0.0145704*2 + 0.0364259*2 + 0.112321*2 + 0.052707*2 + 0.047 + 0.046)*9.81])


if __name__ == '__main__':
    sampling_rate = 1000
    theta = np.repeat(np.deg2rad(np.linspace(17, 160, sampling_rate)),sampling_rate)
    beta = np.tile(np.deg2rad(np.linspace(-180, 180, sampling_rate)),sampling_rate)
    
    jacobian, contact_rim = get_jacobian(theta, beta)
    
    determinant = np.linalg.det(jacobian)
    determinant[contact_rim == None] = None
    
    boundary_dict = {'N4': [], '43': [], '32': [],
                     '21': [], '10': [], '0N': []}
    det_zero_dict = {'L0': [], 'R0': []}
    
    for i in range(contact_rim.__len__()-1):
        if contact_rim[i] == None and contact_rim[i+1] ==    4: boundary_dict['N4'].append([theta[i], beta[i]])
        if contact_rim[i] ==    4 and contact_rim[i+1] ==    3: boundary_dict['43'].append([theta[i], beta[i]])
        if contact_rim[i] ==    3 and contact_rim[i+1] ==    2: boundary_dict['32'].append([theta[i], beta[i]])
        if contact_rim[i] ==    2 and contact_rim[i+1] ==    1: boundary_dict['21'].append([theta[i], beta[i]])
        if contact_rim[i] ==    1 and contact_rim[i+1] ==    0: boundary_dict['10'].append([theta[i], beta[i]])
        if contact_rim[i] ==    0 and contact_rim[i+1] == None: boundary_dict['0N'].append([theta[i], beta[i]])

        if determinant[i] >=    0 and determinant[i+1] <=    0: det_zero_dict['L0'].append([theta[i], beta[i]])
        if determinant[i] <=    0 and determinant[i+1] >=    0: det_zero_dict['R0'].append([theta[i], beta[i]])
    
    determinant = determinant.reshape((sampling_rate, sampling_rate))
    
    
    
    
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{mathptmx}')
    plt.rcParams.update({'font.size': 22})
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.figure(figsize=(24, 10))
    
    cmap_modified = LinearSegmentedColormap.from_list('cmap_modified', plt.cm.coolwarm_r(np.linspace(0.2, 0.8, 256)))
    
    img = plt.imshow(determinant, extent=[np.rad2deg(beta.min()), np.rad2deg(beta.max()), np.rad2deg(theta.min()), np.rad2deg(theta.max())], 
                     origin='lower', cmap=cmap_modified, aspect='auto')
    
    cbar = plt.colorbar(img, label='value of determinants')
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    
    for tb in boundary_dict.values():
        tb = np.rad2deg(np.array(tb))
        plt.plot(tb[:, 1], tb[:, 0], label=f'', linestyle='-', color='gray', linewidth=1.5)
    
    for tb in det_zero_dict.values():
        tb = np.rad2deg(np.array(tb))
        plt.plot(tb[:, 1], tb[:, 0], label=f'', linestyle='-', color='blue', linewidth=2.5)
        
        
        
    df_traj = pd.read_csv(f'{os.path.dirname(os.path.abspath(__file__))}/Trajectory/traj_walk_h290.csv')
    traj_phi = df_traj.iloc[:, [0,1]].to_numpy()
    traj_theta = (traj_phi[:,0]-traj_phi[:,1])/2+np.deg2rad(17)
    traj_beta = (traj_phi[:,0]+traj_phi[:,1])/2
    traj_beta += np.pi
    traj_beta %= np.pi*2
    traj_beta -= np.pi
    plt.plot(np.rad2deg(traj_beta)[4240:5200], np.rad2deg(traj_theta)[4240:5200], label=f'Walk (Stance)', linestyle=(0, (1, 1)), color='lightgreen', linewidth=4, zorder=10)
    plt.plot(np.rad2deg(traj_beta)[4000:4240], np.rad2deg(traj_theta)[4000:4240], label=f'Walk (Swing)', linestyle=(0, (1, 1)), color='darkgreen', linewidth=4, zorder=10)
    
    
    df_traj = pd.read_csv(f'{os.path.dirname(os.path.abspath(__file__))}/Trajectory/traj_rot_h169.csv')
    traj_phi = df_traj.iloc[:, [0,1]].to_numpy()[2802:4934]
    traj_phi = traj_phi[df_traj.iloc[2802:4934, -4] == 0]
    traj_theta = (traj_phi[:,0]-traj_phi[:,1])/2+np.deg2rad(17)
    traj_beta = (traj_phi[:,0]+traj_phi[:,1])/2
    traj_beta += np.pi
    traj_beta %= np.pi*2
    traj_beta -= np.pi
    sorted_indices = np.argsort(traj_beta)
    traj_theta = traj_theta[sorted_indices]
    traj_beta = traj_beta[sorted_indices]
    plt.plot(np.rad2deg(traj_beta), np.rad2deg(traj_theta), label=f'Hybrid Walk (Stance)', linestyle='--', color='red', linewidth=3.5, zorder=10)
    
    df_traj = pd.read_csv(f'{os.path.dirname(os.path.abspath(__file__))}/Trajectory/traj_rot_h169.csv')
    traj_phi = df_traj.iloc[:, [6,7]].to_numpy()[2802:4934]
    traj_phi = traj_phi[df_traj.iloc[2802:4934, -1] == 0]
    traj_theta = (traj_phi[:,0]-traj_phi[:,1])/2+np.deg2rad(17)
    traj_beta = (traj_phi[:,0]+traj_phi[:,1])/2
    traj_beta += np.pi
    traj_beta %= np.pi*2
    traj_beta -= np.pi
    sorted_indices = np.argsort(traj_beta)
    traj_theta = traj_theta[sorted_indices]
    traj_beta = traj_beta[sorted_indices]
    plt.plot(np.rad2deg(traj_beta), np.rad2deg(traj_theta), label=f'', linestyle='--', color='red', linewidth=3.5, zorder=10)
    df_traj = pd.read_csv(f'{os.path.dirname(os.path.abspath(__file__))}/Trajectory/traj_rot_h169.csv')
    traj_phi = df_traj.iloc[:, [0,1]].to_numpy()[2802:4934]
    traj_theta = (traj_phi[:,0]-traj_phi[:,1])/2+np.deg2rad(17)
    traj_beta = (traj_phi[:,0]+traj_phi[:,1])/2
    traj_beta += np.pi
    traj_beta %= np.pi*2
    traj_beta -= np.pi
    sorted_indices = np.argsort(traj_beta)
    traj_theta = traj_theta[sorted_indices]
    traj_beta = traj_beta[sorted_indices]
    plt.plot(np.rad2deg(traj_beta)[:266], np.rad2deg(traj_theta)[:266], label=f'Hybrid Walk (Swing)', linestyle='--', color='darkred', linewidth=3.5, zorder=10)
    
    df_traj = pd.read_csv(f'{os.path.dirname(os.path.abspath(__file__))}/Trajectory/traj_rot_h169.csv')
    traj_phi = df_traj.iloc[:, [6,7]].to_numpy()[2802:4934]
    traj_theta = (traj_phi[:,0]-traj_phi[:,1])/2+np.deg2rad(17)
    traj_beta = (traj_phi[:,0]+traj_phi[:,1])/2
    traj_beta += np.pi
    traj_beta %= np.pi*2
    traj_beta -= np.pi
    sorted_indices = np.argsort(traj_beta)
    traj_theta = traj_theta[sorted_indices]
    traj_beta = traj_beta[sorted_indices]
    plt.plot(np.rad2deg(traj_beta)[-266:], np.rad2deg(traj_theta)[-266:], label=f'', linestyle='--', color='darkred', linewidth=3.5, zorder=10)
    
    

    df_traj = pd.read_csv(f'{os.path.dirname(os.path.abspath(__file__))}/Trajectory/traj_wave.csv')
    traj_phi = df_traj.iloc[:, [0,1]].to_numpy()[2800:]
    traj_theta = (traj_phi[:,0]-traj_phi[:,1])/2+np.deg2rad(17)
    traj_beta = (traj_phi[:,0]+traj_phi[:,1])/2
    traj_beta += np.pi
    traj_beta %= np.pi*2
    traj_beta -= np.pi
    sorted_indices = np.argsort(traj_beta)
    traj_theta = traj_theta[sorted_indices]
    traj_beta = traj_beta[sorted_indices]
    plt.plot(np.rad2deg(traj_beta), np.rad2deg(traj_theta), label=f'Wave (Stance)', linestyle='-.', color='darkmagenta', linewidth=3.5, zorder=10)
    
    
    
    
    
    
    plt.text(-90, 130, 'Right\nUpper Rim', ha='center', va='center', fontsize=16)
    plt.text(-36,  49, 'Right\nLower Rim', ha='center', va='center', fontsize=16)
    plt.text(  0,  140, 'G', ha='center', va='center', fontsize=16)
    plt.text( 36,  49, 'Left\nLower Rim', ha='center', va='center', fontsize=16)
    plt.text( 90, 130, 'Left\nUpper Rim', ha='center', va='center', fontsize=16)
    
    
    plt.title('Trajectories and Jacobian Determinants on the Contact Map')
    plt.xlabel(r'$\beta (\deg)$')
    plt.ylabel(r'$\theta (\deg)$')
    plt.legend(loc='upper left', fontsize=16)
    plt.savefig('Jacobian.pdf', format='pdf', bbox_inches='tight')
    plt.show()