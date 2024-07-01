import numpy as np
from numpy.polynomial import Polynomial

# coefficients of all the points on the right plane
A_x = [-1.129288648260761e-05, 0.08009403713692657, -0.0003107144391614336, -0.012797873528955785, -0.0005295870938277623, 0.0009747008204006224, -0.00010114551418980388, -3.968574387602764e-07]
A_y = [0.07999814924773316, 1.7145219587770126e-05, -0.04006470379517542, 0.00013128098674423447, 0.0031745436156560103, 0.00011924483962515575, -0.0001666256258221374, 1.515537414584306e-05]
B_x = [-1.411610810312764e-05, 0.1001175464211573, -0.0003883930489497228, -0.015997341911197397, -0.0006619838672826222, 0.001218376025499842, -0.00012643189273703638, -4.960717984706521e-07]
B_y = [0.09999768655966619, 2.14315244865344e-05, -0.050080879743974474, 0.00016410123343752557, 0.003968179519564681, 0.00014905604953358207, -0.000208282032278111, 1.8944217682340097e-05]
C_x = [0.08730444109525554, 0.006038411912842335, -0.0019480640877697262, -0.02954182547087869, 0.02239356552307967, -0.005419375878885002, -0.00010688833690058729, 0.00014758582950897729]
C_y = [-0.02935430914372217, -0.04803664310194349, -0.05069851096683131, 0.04823695695975268, -0.024522775370319442, 0.0034302257248766694, 0.000970517201979338, -0.00022005316153193402]
D_x = [-3.4747343022248285e-06, 0.02464431911905334, -9.560444281634296e-05, -0.003937807239681712, -0.000162949875022163, 0.00029990794473815584, -3.1121696673735556e-05, -1.221099811557898e-07]
D_y = [-0.011597133589474484, 0.01203679245902142, -0.05726952461757285, 0.04728991254517789, -0.03351780358840571, 0.010131978808776108, -0.0007682114483517649, -6.52844643872138e-05]
E_x = [1.9284883733625186e-16, -1.4324042428771431e-15, 3.6555203154059035e-15, -4.3189076651658346e-15, 2.542849414643894e-15, -7.098433225234169e-16, 6.760715836529259e-17, 2.4439010417800376e-18]
E_y = [-0.05230614818378873, 0.017378857898768023, -0.06491611164974555, 0.06824930434892112, -0.04982551345687386, 0.014582082795063848, -0.0010355829250312493, -0.00010103550373527039]
F_x = [0.07922115681627133, -0.0064245034694449385, -0.0021344474707275475, -0.02512408462146725, 0.020898976289620615, -0.005545020060861925, 2.7733258483555903e-05, 0.0001316598828257355]
F_y = [-0.04888948991749337, -0.04100243386330336, -0.050568505590458196, 0.05336130836442207, -0.02925741541443177, 0.004425371925477578, 0.001056446425428136, -0.0002546503426689543]
G_x = [5.692831173187612e-16, -4.2389959233541044e-15, 1.1233596662274271e-14, -1.4258205079718426e-14, 9.514532901870432e-15, -3.366933068026824e-15, 5.829486054269246e-16, -3.704221547207857e-17]
G_y = [-0.08004422213631104, -0.04301265930703809, -0.10580878294257633, 0.08886009184138272, -0.03103902364866719, -0.001105418324450194, 0.0030330875126614357, -0.0004650360042865928]
H_x = [-0.029867885022261887, 0.1015083446518684, -0.0003359302945983391, -0.006767707421146685, -0.007614704919851656, 0.002412001729076878, 6.519366276181607e-05, -5.434796710906423e-05]
H_y = [0.09842205910372219, 0.020210162361816802, -0.04976385386568556, -0.0023547588103848085, 0.0029934392163673065, 0.000915230066986905, -0.00032778979330257043, 1.8208984392037498e-05]
U_x = [-0.009669615824629486, 0.03326996939355064, -0.0014227965133690036, -0.0029554816183596563, -0.0008772745633094239, -0.0007481061509797202, 0.00036920343536740816, -2.4962239045435592e-05]
U_y = [-0.0006682432219354362, 0.014768821071982609, -0.04974684856179978, 0.02978313440927041, -0.0197800620541211, 0.004525511094612111, 0.00037306235275841695, -0.00016158912158916643]
L_x = [0.006204808983197836, -0.005367722922662346, -0.060298802271597424, 0.025501651107259254, 0.008539152550844633, -0.008702919013865187, 0.002133326874710949, -0.0001597568122447928]
L_y = [0.02047830349473066, -0.048896242660697795, -0.08047731295274864, 0.0441713134318154, -0.007739219889231084, -0.004285690146924385, 0.002074504051417415, -0.00021867040854111553]

# Polynominal of all the points on the right plane
A_poly = np.array([Polynomial(A_x), Polynomial(A_y)])
B_poly = np.array([Polynomial(B_x), Polynomial(B_y)])
C_poly = np.array([Polynomial(C_x), Polynomial(C_y)])
D_poly = np.array([Polynomial(D_x), Polynomial(D_y)])
E_poly = np.array([Polynomial(E_x), Polynomial(E_y)])
F_poly = np.array([Polynomial(F_x), Polynomial(F_y)])
G_poly = np.array([Polynomial(G_x), Polynomial(G_y)])
H_poly = np.array([Polynomial(H_x), Polynomial(H_y)])
U_poly = np.array([Polynomial(U_x), Polynomial(U_y)])
L_poly = np.array([Polynomial(L_x), Polynomial(L_y)])

# Other global parameters
eps = 10e-6


def rot_matrix(angle):
        return np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle),  np.cos(angle)]])


def get_alpha(theta, beta):
    rot_beta = rot_matrix(beta)
    
    H_r = rot_beta @ [ H_poly[0](theta), H_poly[1](theta)]
    U_r = rot_beta @ [ U_poly[0](theta), U_poly[1](theta)]
    F_r = rot_beta @ [ F_poly[0](theta), F_poly[1](theta)]
    L_r = rot_beta @ [ L_poly[0](theta), L_poly[1](theta)]
    G   = rot_beta @ [ G_poly[0](theta), G_poly[1](theta)]
    L_l = rot_beta @ [-L_poly[0](theta), L_poly[1](theta)]
    F_l = rot_beta @ [-F_poly[0](theta), F_poly[1](theta)]
    U_l = rot_beta @ [-U_poly[0](theta), U_poly[1](theta)]
    H_l = rot_beta @ [-H_poly[0](theta), H_poly[1](theta)]
    
    R = 0.1
    contact_height = []
    contact_height.append(U_r[1]-R if U_r[0] - H_r[0] < eps and F_r[0] - U_r[0] < eps else 0)
    contact_height.append(L_r[1]-R if L_r[0] - F_r[0] < eps and   G[0] - L_r[0] < eps else 0)
    contact_height.append(G[1])
    contact_height.append(L_l[1]-R if F_l[0] - L_l[0] < eps and L_l[0] -   G[0] < eps else 0)
    contact_height.append(U_l[1]-R if H_l[0] - U_l[0] < eps and U_l[0] - F_l[0] < eps else 0)

    contact_rim = np.argmin(contact_height) if min(contact_height) < 0 else np.nan
    if   contact_rim == 0: alpha = -np.pi/2 - np.arctan2(F_r[1]-U_r[1], F_r[0]-U_r[0]) + np.deg2rad(50)
    elif contact_rim == 1: alpha = -np.pi/2 - np.arctan2(  G[1]-L_r[1],   G[0]-L_r[0])
    elif contact_rim == 2: alpha = 0
    elif contact_rim == 3: alpha = -np.pi/2 - np.arctan2(  G[1]-L_l[1],   G[0]-L_l[0])
    elif contact_rim == 4: alpha = -np.pi/2 - np.arctan2(F_l[1]-U_l[1], F_l[0]-U_l[0]) - np.deg2rad(50)
    else: alpha = np.nan

    return alpha, contact_rim


def get_jacobian(theta, beta, alpha):
    if alpha > np.deg2rad(50):
        print('Right Upper Rim')
        P = rot_matrix(alpha-np.deg2rad(50)) @ (F_poly-U_poly) + U_poly
    elif alpha > 0:
        print('Right Lower Rim')
        P = rot_matrix(alpha) @ (G_poly-L_poly) + L_poly
    elif alpha == 0:
        print('G')
        P = G_poly.copy()
    elif alpha < -np.deg2rad(50):
        print('Left Upper Rim')
        P = (rot_matrix(-alpha-np.deg2rad(50)) @ (F_poly-U_poly) + U_poly) * np.array([-1, 1])
    elif alpha < 0:
        print('Left Lower Rim')
        P = (rot_matrix(-alpha) @ (G_poly-L_poly) + L_poly) * np.array([-1, 1])
    
    P_deriv = np.array([P[0].deriv(), P[1].deriv()])
    
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


def get_link_force(theta, beta):
    g = 9.81
    
    l12_acc = np.array([B_poly[0].deriv().deriv()(theta)/2, B_poly[1].deriv().deriv()(theta)/2 - g])

    l4_acc = np.array([(D_poly[0].deriv().deriv()(theta) + C_poly[0].deriv().deriv()(theta))/2,
                       (D_poly[1].deriv().deriv()(theta) + C_poly[1].deriv().deriv()(theta))/2 - g])
    
    l56_acc = np.array([(A_poly[0].deriv().deriv()(theta) + E_poly[0].deriv().deriv()(theta))/2,
                        (A_poly[1].deriv().deriv()(theta) + E_poly[1].deriv().deriv()(theta))/2 - g])
    
    l37_acc = np.array([(H_poly[0].deriv().deriv()(theta) + F_poly[0].deriv().deriv()(theta))/2,
                        (H_poly[1].deriv().deriv()(theta) + F_poly[1].deriv().deriv()(theta))/2 - g])
    
    l8_acc = np.array([(F_poly[0].deriv().deriv()(theta) + G_poly[0].deriv().deriv()(theta))/2,
                       (F_poly[1].deriv().deriv()(theta) + G_poly[1].deriv().deriv()(theta))/2 - g])
    
    # force = 0.047*l12_acc + 0.014*l4_acc + 0.035*l56_acc + 0.112*l37_acc + 0.053*l8_acc
    force = (0.047 + 0.014 + 0.035 + 0.112 + 0.053) * (-g)
    
    return force * 2
    
    
if __name__ == '__main__':
    
    phi = [1.46677191235467,-1.4667719123209109]
    trq = [2.7922063369220687,-2.79241478976431]
    
    theta = (phi[0]-phi[1])/2 + np.deg2rad(17)
    beta = -(phi[0]+phi[1])/2
    
    alpha, contact_rim = get_alpha(theta=theta, beta=beta)
    
    print(f'Theta = {round(np.rad2deg(theta), 4)}; Beta = {round(np.rad2deg(beta), 4)}; Alpha = {round(np.rad2deg(alpha), 4)}, Contact Rim = {contact_rim}')
    
    jacobian = get_jacobian(theta=theta, beta=beta, alpha=alpha)
    
    action_force = np.linalg.inv(jacobian).T @ trq
    
    link_force = 0 # get_link_force(theta=theta, beta=beta)
    
    meas_force = (action_force - link_force)
    
    print([meas_force[0], meas_force[1]])
    