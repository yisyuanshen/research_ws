import numpy as np
import matplotlib.pyplot as plt


# Left Plane Curve Fitting Coefficients
A_x = [1.1292886482058312e-05, -0.08009403713692212, 0.0003107144391486884, 0.012797873528973849, 0.0005295870938137342, -0.0009747008203945364, 0.00010114551418841818, 3.968574388893076e-07]
A_y = [0.07999814924773244, 1.7145219592933924e-05, -0.04006470379518947, 0.0001312809867631146, 0.003174543615642159, 0.00011924483963082137, -0.00016662562582335288, 1.5155374145949843e-05]
B_x = [1.4116108102294423e-05, -0.10011754642115084, 0.0003883930489312213, 0.015997341911223414, 0.0006619838672625973, -0.0012183760254912332, 0.00012643189273509628, 4.960717986490948e-07]
B_y = [0.09999768655966541, 2.143152449250079e-05, -0.0500808797439909, 0.00016410123345931993, 0.003968179519549091, 0.00014905604953972408, -0.00020828203227936626, 1.8944217682444055e-05]
C_x = [-0.0873044410952548, -0.006038411912847815, 0.001948064087783593, 0.02954182547086227, -0.022393565523069883, 0.0054193758788821725, 0.00010688833690089704, -0.00014758582950897536]
C_y = [-0.029354309143722876, -0.04803664310193761, -0.05069851096684769, 0.04823695695977528, -0.024522775370336612, 0.0034302257248839787, 0.0009705172019776995, -0.00022005316153178318]
D_x = [3.474734302376189e-06, -0.024644319119054316, 9.560444281846218e-05, 0.003937807239679968, 0.00016294987502238215, -0.0002999079447377171, 3.1121696673510875e-05, 1.2210998118812034e-07]
D_y = [-0.01159713358947484, 0.012036792459023461, -0.05726952461757834, 0.04728991254518538, -0.03351780358841137, 0.010131978808778506, -0.0007682114483523003, -6.528446438716484e-05]
E_x = [1.9284883733625186e-16, -1.4324042428771431e-15, 3.6555203154059035e-15, -4.3189076651658346e-15, 2.542849414643894e-15, -7.098433225234169e-16, 6.760715836529259e-17, 2.4439010417800376e-18]
E_y = [-0.05230614818378873, 0.017378857898768023, -0.06491611164974555, 0.06824930434892112, -0.04982551345687386, 0.014582082795063848, -0.0010355829250312493, -0.00010103550373527039]
F_x = [-0.07922115681627032, 0.0064245034694379, 0.0021344474707458874, 0.02512408462144383, -0.020898976289604555, 0.005545020060855934, -2.7733258482419835e-05, -0.00013165988282582016]
F_y = [-0.048889489917494255, -0.041002433863297126, -0.05056850559047525, 0.053361308364445574, -0.02925741541444968, 0.004425371925485272, 0.001056446425426396, -0.0002546503426687935]
G_x = [5.692831173187612e-16, -4.2389959233541044e-15, 1.1233596662274271e-14, -1.4258205079718426e-14, 9.514532901870432e-15, -3.366933068026824e-15, 5.829486054269246e-16, -3.704221547207857e-17]
G_y = [-0.08004422213631104, -0.04301265930703809, -0.10580878294257633, 0.08886009184138272, -0.03103902364866719, -0.001105418324450194, 0.0030330875126614357, -0.0004650360042865928]
H_x = [0.029867885022260846, -0.10150834465186069, 0.00033593029457692076, 0.006767707421176299, 0.0076147049198291485, -0.002412001729067313, -6.51936627639469e-05, 5.434796710925797e-05]
H_y = [0.0984220591037217, 0.020210162361820168, -0.049763853865694395, -0.0023547588103735376, 0.0029934392163595675, 0.0009152300669897989, -0.00032778979330312056, 1.820898439207874e-05]
U_x = [0.009669615824629486, -0.03326996939355048, 0.0014227965133679344, 0.002955481618362319, 0.0008772745633062714, 0.000748106150981612, -0.00036920343536796214, 2.496223904549841e-05]
U_y = [-0.0006682432219356628, 0.014768821071984477, -0.049746848561805015, 0.029783134409277722, -0.01978006205412676, 0.004525511094614576, 0.00037306235275785175, -0.00016158912158911357]
L_x = [-0.006204808983197421, 0.00536772292266007, 0.060298802271602565, -0.025501651107264288, -0.00853915255084279, 0.008702919013865378, -0.0021333268747112232, 0.00015975681224483941]
L_y = [0.020478303494730617, -0.048896242660697496, -0.08047731295274936, 0.04417131343181672, -0.007739219889232588, -0.004285690146923471, 0.0020745040514171436, -0.00021867040854108426]


# Polynomial Fitting of Each Point
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


def rot_matrix(angle): return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle),  np.cos(angle)]])


def get_contact_point(theta, beta):
    rot_beta = rot_matrix(beta)
    
    A_l = rot_beta @ [A_l_poly[0](theta), A_l_poly[1](theta)]
    A_r = rot_beta @ [A_r_poly[0](theta), A_r_poly[1](theta)]
    B_l = rot_beta @ [B_l_poly[0](theta), B_l_poly[1](theta)]
    B_r = rot_beta @ [B_r_poly[0](theta), B_r_poly[1](theta)]
    C_l = rot_beta @ [C_l_poly[0](theta), C_l_poly[1](theta)]
    C_r = rot_beta @ [C_r_poly[0](theta), C_r_poly[1](theta)]
    D_l = rot_beta @ [D_l_poly[0](theta), D_l_poly[1](theta)]
    D_r = rot_beta @ [D_r_poly[0](theta), D_r_poly[1](theta)]
    E   = rot_beta @ [E_poly  [0](theta), E_poly  [1](theta)]
    F_l = rot_beta @ [F_l_poly[0](theta), F_l_poly[1](theta)]
    F_r = rot_beta @ [F_r_poly[0](theta), F_r_poly[1](theta)]
    G   = rot_beta @ [G_poly  [0](theta), G_poly  [1](theta)]
    H_l = rot_beta @ [H_l_poly[0](theta), H_l_poly[1](theta)]
    H_r = rot_beta @ [H_r_poly[0](theta), H_r_poly[1](theta)]
    U_l = rot_beta @ [U_l_poly[0](theta), U_l_poly[1](theta)]
    U_r = rot_beta @ [U_r_poly[0](theta), U_r_poly[1](theta)]
    L_l = rot_beta @ [L_l_poly[0](theta), L_l_poly[1](theta)]
    L_r = rot_beta @ [L_r_poly[0](theta), L_r_poly[1](theta)]
    
    R = 0.1
    eps = 10e-4
    
    contact_height = []
    contact_height.append(U_l[1]-R if (H_l[0]-eps < U_l[0] < F_l[0]+eps) else 0)
    contact_height.append(L_l[1]-R if (F_l[0]-eps < L_l[0] < G[0]+eps)   else 0)
    contact_height.append(G[1])
    contact_height.append(L_r[1]-R if (G[0]-eps < L_r[0] < F_r[0]+eps)   else 0)
    contact_height.append(U_r[1]-R if (F_r[0]-eps < U_r[0] < H_r[0]+eps) else 0)
    
    if min(contact_height) >= 0: return None, None, None
    
    contact_rim = np.argmin(contact_height)
    
    if contact_rim == 0:
        alpha = -np.pi/2 - np.arctan2(F_l[1]-U_l[1], F_l[0]-U_l[0]) - np.deg2rad(50)
        P = rot_matrix(alpha+np.deg2rad(50)) @ (F_l_poly-U_l_poly) + U_l_poly
    
    elif contact_rim == 1:
        alpha = -np.pi/2 - np.arctan2(G[1]-L_l[1], G[0]-L_l[0])
        P = rot_matrix(alpha) @ (G_poly-L_l_poly) + L_l_poly
    
    elif contact_rim == 2:
        alpha = 0
        P = G_poly.copy()
    
    elif contact_rim == 3:
        alpha = -np.pi/2 - np.arctan2(G[1]-L_r[1], G[0]-L_r[0])
        P = rot_matrix(alpha) @ (G_poly-L_r_poly) + L_r_poly
    
    elif contact_rim == 4:
        alpha = -np.pi/2 - np.arctan2(F_r[1]-U_r[1], F_r[0]-U_r[0]) + np.deg2rad(50)
        P = rot_matrix(alpha-np.deg2rad(50)) @ (F_r_poly-U_r_poly) + U_r_poly
    
    return contact_rim, alpha, P


def get_jacobian(theta, beta):
    contact_rim, alpha, P = get_contact_point(theta, beta)
    
    if contact_rim is None: return None
    
    P_deriv = np.array([P[0].deriv(), P[1].deriv()])
    
    # print('\n= = = = =')
    # print(f'TB = ({np.rad2deg(theta)}, {np.rad2deg(beta)})')
    # print(f'Alpha = {np.rad2deg(alpha)}')
    # print(f'Contact Rim = {contact_rim}')
    
    c_beta = np.cos(beta)
    s_beta = np.sin(beta)
    P[0] = P[0](theta)
    P[1] = P[1](theta)
    P_deriv[0] = P_deriv[0](theta)
    P_deriv[1] = P_deriv[1](theta)
    
    jacobian = np.array([[-(c_beta*P_deriv[0]-s_beta*P_deriv[1])/2 + (-s_beta*P[0]-c_beta*P[1])/2,
                           (c_beta*P_deriv[0]-s_beta*P_deriv[1])/2 + (-s_beta*P[0]-c_beta*P[1])/2],
                         [-(s_beta*P_deriv[0]+c_beta*P_deriv[1])/2 +  (c_beta*P[0]-s_beta*P[1])/2,
                           (s_beta*P_deriv[0]+c_beta*P_deriv[1])/2 +  (c_beta*P[0]-s_beta*P[1])/2]])
    
    return jacobian


def get_link_force(enable):
    return np.array([0, (0.0145704*2 + 0.0364259*2 + 0.112321*2 + 0.052707*2 + 0.047 + 0.046)*9.81]) if enable else 0


if __name__ == '__main__':
    # '''
    # Validate Jacobian Calculation
    theta_values = np.linspace(np.deg2rad(17), np.deg2rad(160), 1000)
    beta_values = np.linspace(np.deg2rad(-180), np.deg2rad(180), 1000)
    
    # Initialize the determinant matrix
    determinants = np.zeros((len(theta_values), len(beta_values)))

    # Calculate determinants
    for i, theta in enumerate(theta_values):
        for j, beta in enumerate(beta_values):
            jacobian = get_jacobian(theta, beta)
            if jacobian is None:
                determinants[i, j] = np.inf
            else:
                det = np.linalg.det(jacobian)
                determinants[i, j] = det
                
                if 10e-4 > det > -10e-4:
                    print(np.rad2deg(theta), np.rad2deg(beta), det)
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.imshow(determinants, extent=[-180, 180, 160, 17], aspect='auto', cmap=plt.cm.jet)
    
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label('Determinant of Jacobian', size=24)
    
    plt.xlabel('Beta (degrees)', fontsize=24)
    plt.ylabel('Theta (degrees)', fontsize=24)
    plt.tick_params(labelsize=20)
    plt.title('Determinant of the Jacobian Matrix', fontsize=28)
    plt.gca().invert_yaxis()
    plt.show()
    # '''