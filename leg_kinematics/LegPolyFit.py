import LegModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

leg = LegModel.Leg()

theta_list = []
G_y = []
H_x = []
H_y = []
F_x = []
F_y = []
upper_x = []  # right plane
upper_y = []
upper_len = []
lower_x = []  
lower_y = []
lower_len = []

sample_rate = 10000

for theta in range(17*sample_rate, 160*sample_rate):
    theta = np.deg2rad(theta/sample_rate) # + leg.theta_0
    beta = leg.beta_0
    leg.set_tb(theta, beta)
    
    
    theta_list.append(theta)
    G_y.append(leg.vec_OG[1])
    H_x.append(leg.vec_OH_r[0])
    H_y.append(leg.vec_OH_r[1])
    F_x.append(leg.vec_OF_r[0])
    F_y.append(leg.vec_OF_r[1])
    upper_x.append(leg.vec_upper_rim_r[0])
    upper_y.append(leg.vec_upper_rim_r[1])
    upper_len.append(np.linalg.norm(leg.vec_upper_rim_r))
    lower_x.append(leg.vec_lower_rim_r[0])
    lower_y.append(leg.vec_lower_rim_r[1])
    lower_len.append(np.linalg.norm(leg.vec_lower_rim_r))

theta_list = np.array(theta_list)
G_y = np.array(G_y)
H_x = np.array(H_x)
H_y = np.array(H_y)
F_x = np.array(F_x)
F_y = np.array(F_y)
upper_x = np.array(upper_x)
upper_y = np.array(upper_y)
upper_len = np.array(upper_len)
lower_x = np.array(lower_x)
lower_y = np.array(lower_y)
lower_len = np.array(lower_len)

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
print(f"[{', '.join([f'{c:.8f}' for c in coef[::-1]])}]")
coef = np.polyfit(theta_list, H_x, degree)
print(f"[{', '.join([f'{c:.8f}' for c in coef[::-1]])}]")
coef = np.polyfit(theta_list, H_y, degree)
print(f"[{', '.join([f'{c:.8f}' for c in coef[::-1]])}]")
coef = np.polyfit(theta_list, F_x, degree)
print(f"[{', '.join([f'{c:.8f}' for c in coef[::-1]])}]")
coef = np.polyfit(theta_list, F_y, degree)
print(f"[{', '.join([f'{c:.8f}' for c in coef[::-1]])}]")
coef = np.polyfit(theta_list, upper_x, degree)
print(f"[{', '.join([f'{c:.8f}' for c in coef[::-1]])}]")
coef = np.polyfit(theta_list, upper_y, degree)
print(f"[{', '.join([f'{c:.8f}' for c in coef[::-1]])}]")
coef = np.polyfit(theta_list, lower_x, degree)
print(f"[{', '.join([f'{c:.8f}' for c in coef[::-1]])}]")
coef = np.polyfit(theta_list, lower_y, degree)
print(f"[{', '.join([f'{c:.8f}' for c in coef[::-1]])}]")


# inverse poly fit
degree = 7
coef = np.polyfit(G_y, theta_list, degree)
print(f"[{', '.join([f'{c:.8f}' for c in coef[::-1]])}]")
coef = np.polyfit(upper_len, theta_list, degree)
print(f"[{', '.join([f'{c:.8f}' for c in coef[::-1]])}]")
coef = np.polyfit(lower_len, theta_list, degree)
print(f"[{', '.join([f'{c:.8f}' for c in coef[::-1]])}]")



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