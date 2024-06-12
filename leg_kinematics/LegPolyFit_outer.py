import LegModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

leg = LegModel.Leg()

theta_list = []

# right plane
G_y = []

H_x = []
H_y = []
F_x_upper = []
F_y_upper = []

F_x_lower = []
F_y_lower = []
G_x_lower = []
G_y_lower = []

U_x = []
U_y = []
U_len = []

L_x = []  
L_y = []
L_len = []

sample_rate = 10000

for theta in range(17*sample_rate, 160*sample_rate):
    theta = np.deg2rad(theta/sample_rate) # + leg.theta_0
    beta = leg.beta_0
    leg.set_tb(theta, beta)
    
    theta_list.append(theta)
    
    G_y.append((leg.R+leg.r)/leg.R*leg.vec_OG[1])
    
    H_x.append(leg.vec_upper_rim_r[0] + (leg.R+leg.r)/leg.R*(leg.vec_OH_r[0]-leg.vec_upper_rim_r[0]))
    H_y.append(leg.vec_upper_rim_r[1] + (leg.R+leg.r)/leg.R*(leg.vec_OH_r[1]-leg.vec_upper_rim_r[1]))
    F_x_upper.append(leg.vec_upper_rim_r[0] + (leg.R+leg.r)/leg.R*(leg.vec_OF_r[0]-leg.vec_upper_rim_r[0]))
    F_y_upper.append(leg.vec_upper_rim_r[1] + (leg.R+leg.r)/leg.R*(leg.vec_OF_r[1]-leg.vec_upper_rim_r[1]))
    
    F_x_lower.append(leg.vec_lower_rim_r[0] + (leg.R+leg.r)/leg.R*(leg.vec_OF_r[0]-leg.vec_lower_rim_r[0]))
    F_y_lower.append(leg.vec_lower_rim_r[1] + (leg.R+leg.r)/leg.R*(leg.vec_OF_r[1]-leg.vec_lower_rim_r[1]))
    G_x_lower.append(leg.vec_lower_rim_r[0] + (leg.R+leg.r)/leg.R*(leg.vec_OG[0]-leg.vec_lower_rim_r[0]))
    G_y_lower.append(leg.vec_lower_rim_r[1] + (leg.R+leg.r)/leg.R*(leg.vec_OG[1]-leg.vec_lower_rim_r[1]))
    
    U_x.append(leg.vec_upper_rim_r[0])
    U_y.append(leg.vec_upper_rim_r[1])
    U_len.append(np.linalg.norm(leg.vec_upper_rim_r))
    
    L_x.append(leg.vec_lower_rim_r[0])
    L_y.append(leg.vec_lower_rim_r[1])
    L_len.append(np.linalg.norm(leg.vec_lower_rim_r))
    

'''
data = {
    'theta': theta_list,
    'G_y': G_y
}

df = pd.DataFrame(data)
df.to_csv('theta_Gy.csv', index=False)
'''


# forward poly fit
degree = 7
coef = np.polyfit(theta_list, G_y, degree)
print(f"Gy_coef = [{', '.join([f'{c:.8f}' for c in coef[::-1]])}]")
coef = np.polyfit(theta_list, H_x, degree)
print(f"Hx_coef = [{', '.join([f'{c:.8f}' for c in coef[::-1]])}]")
coef = np.polyfit(theta_list, H_y, degree)
print(f"Hy_coef = [{', '.join([f'{c:.8f}' for c in coef[::-1]])}]")
coef = np.polyfit(theta_list, F_x_upper, degree)
print(f"Fx_upper_coef = [{', '.join([f'{c:.8f}' for c in coef[::-1]])}]")
coef = np.polyfit(theta_list, F_y_upper, degree)
print(f"Fy_upper_coef = [{', '.join([f'{c:.8f}' for c in coef[::-1]])}]")
coef = np.polyfit(theta_list, F_x_lower, degree)
print(f"Fx_lower_coef = [{', '.join([f'{c:.8f}' for c in coef[::-1]])}]")
coef = np.polyfit(theta_list, F_y_lower, degree)
print(f"Fy_lower_coef = [{', '.join([f'{c:.8f}' for c in coef[::-1]])}]")
coef = np.polyfit(theta_list, G_x_lower, degree)
print(f"Gx_lower_coef = [{', '.join([f'{c:.8f}' for c in coef[::-1]])}]")
coef = np.polyfit(theta_list, G_y_lower, degree)
print(f"Gy_lower_coef = [{', '.join([f'{c:.8f}' for c in coef[::-1]])}]")
coef = np.polyfit(theta_list, U_x, degree)
print(f"Ux_coef = [{', '.join([f'{c:.8f}' for c in coef[::-1]])}]")
coef = np.polyfit(theta_list, U_y, degree)
print(f"Uy_coef = [{', '.join([f'{c:.8f}' for c in coef[::-1]])}]")
coef = np.polyfit(theta_list, L_x, degree)
print(f"Lx_coef = [{', '.join([f'{c:.8f}' for c in coef[::-1]])}]")
coef = np.polyfit(theta_list, L_y, degree)
print(f"Ly_coef = [{', '.join([f'{c:.8f}' for c in coef[::-1]])}]")


# inverse poly fit
degree = 7
coef = np.polyfit(G_y, theta_list, degree)
print(f"inv_Gy_coef = [{', '.join([f'{c:.8f}' for c in coef[::-1]])}]")
coef = np.polyfit(U_len, theta_list, degree)
print(f"inv_U_len_coef = [{', '.join([f'{c:.8f}' for c in coef[::-1]])}]")
coef = np.polyfit(L_len, theta_list, degree)
print(f"inv_L_len_coef = [{', '.join([f'{c:.8f}' for c in coef[::-1]])}]")


'''
plt.figure(figsize=(10, 6))
plt.plot(theta_list, lower_x, color='blue', label='Data', linewidth=0.5)
# plt.plot(theta_list, lower_len, color='red', label='Data', linewidth=0.5)

degree = 7
coefficients = np.polyfit(G_y, theta_list, degree)
print(coefficients)

polynomial = np.poly1d(coefficients)

# theta_line = np.linspace(min(theta_list), max(theta_list), 1000)
theta_line = np.linspace(max(G_y), min(G_y), 1000)
vec_OG_y_fit = polynomial(theta_line)


# plt.plot(theta_line, vec_OG_y_fit, color='red', label=f'{degree} Degree Polynomial Fit', linewidth=0.5)
plt.xlabel('Theta (radians)')
plt.ylabel('Data')
plt.title('Theta vs. Data')
plt.legend()
plt.grid(True)
plt.show()
'''