import numpy as np

class Leg:
    def __init__(self):
        self.phi_r = 0
        self.phi_l = 0
        
        self.theta = 0
        self.beta = 0
        
        self.theta_0 = np.deg2rad(17)
        self.beta_0  = np.deg2rad(90)

        self.R = 0.1
        self.r = 0.0125
        self.n_HF = np.deg2rad(130)
        self.n_BC = np.deg2rad(101)
        
        self.l1 = 0.8 * self.R
        self.l2 = self.R - self.l1
        self.l3 = 2 * self.R * np.sin(self.n_BC / 2)
        self.l4 = 0.88296634 * self.R
        self.l5 = 0.9 * self.R
        self.l6 = 0.4 * self.R
        self.l7 = 2 * self.R * np.sin((self.n_HF - self.n_BC - self.theta_0) / 2)
        self.l8 = 2 * self.R * np.sin((np.pi - self.n_HF) / 2)
        
        self.set_tb(theta=self.theta_0, beta=self.beta_0)
    
    
    def set_tb(self, theta, beta):
        self.theta = theta
        self.beta = beta + self.beta_0
        self.phi_r = self.beta - self.theta + self.theta_0 - self.beta_0
        self.phi_l = self.beta + self.theta - self.theta_0 - self.beta_0
        self.calculate_forward()
    
    def set_phi(self, phi_r, phi_l):
        self.phi_r = phi_r
        self.phi_l = phi_l
        self.theta = (phi_l - phi_r) / 2 + self.theta_0
        self.beta = (phi_l + phi_r) / 2 - self.beta_0
        self.calculate_forward()
    
    
    def rot_matrix(self, angle):
        return np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle),  np.cos(angle)]])
    
    
    def calculate_forward(self):
        self.O = np.array([0, 0])

        # A
        vec_OA_r = self.l1 * np.array([np.cos(-self.theta), np.sin(-self.theta)])
        
        # B
        vec_OB_r = (self.l1 + self.l2) * np.array([np.cos(-self.theta), np.sin(-self.theta)])
        
        # E
        ang_AOE = np.pi - self.theta
        ang_OAE = np.pi - ang_AOE - np.arcsin(self.l1 * np.sin(ang_AOE) / (self.l5 + self.l6))
        rot_OAE = self.rot_matrix(ang_OAE)
        vec_AE_r = rot_OAE @ (-vec_OA_r) * (self.l5 + self.l6) / np.linalg.norm(vec_OA_r)
        vec_OE = vec_OA_r + vec_AE_r
        
        # D
        vec_OD_r = (self.l5 * vec_OE + self.l6 * vec_OA_r) / (self.l5 + self.l6)
        
        # C
        vec_BD_r = vec_OD_r - vec_OB_r
        ang_DBC = np.arccos((self.l3**2 + np.linalg.norm(vec_BD_r)**2 - self.l4**2) / (2 * self.l3 * np.linalg.norm(vec_BD_r)))
        rot_DBC = self.rot_matrix(ang_DBC)
        vec_BC_r = rot_DBC @ vec_BD_r * self.l3 / np.linalg.norm(vec_BD_r)
        vec_OC_r = vec_OB_r + vec_BC_r
        
        # F
        arc_BF = self.n_HF - self.theta_0
        ang_BCF = np.arccos((self.l3**2 + self.l7**2 - 2 * self.R**2 + 2 * self.R**2 * np.cos(arc_BF)) / (2 * self.l3 * self.l7))
        rot_BCF = self.rot_matrix(ang_BCF)
        vec_CB_r = vec_OB_r - vec_OC_r
        vec_CF_r = rot_BCF @ vec_CB_r * self.l7 / np.linalg.norm(vec_CB_r)
        vec_OF_r = vec_OC_r + vec_CF_r
        
        # G
        ang_GOF = np.arccos(vec_OE @ vec_OF_r / np.linalg.norm(vec_OE) / np.linalg.norm(vec_OF_r))
        ang_OGF = np.arcsin(np.linalg.norm(vec_OF_r) * np.sin(ang_GOF) / self.l8)
        ang_OFG = np.pi - ang_GOF - ang_OGF
        norm_OG = self.l8 * np.sin(ang_OFG) / np.sin(ang_GOF)
        vec_OG = vec_OE * norm_OG / np.linalg.norm(vec_OE)
        
        # U
        ang_B_upper_C = -(np.pi - self.n_BC) / 2
        rot_B_upper_C = self.rot_matrix(ang_B_upper_C)
        vec_upper_rim_r = vec_OB_r + rot_B_upper_C @ vec_BC_r * self.R / np.linalg.norm(vec_BC_r)
        
        # L
        ang_F_lower_G = -self.n_HF / 2
        rot_F_lower_G = self.rot_matrix(ang_F_lower_G)
        vec_FG_r = vec_OG - vec_OF_r
        vec_lower_rim_r = vec_OF_r + rot_F_lower_G @ vec_FG_r * self.R / np.linalg.norm(vec_FG_r)
        
        # H
        vec_OH_r = vec_upper_rim_r + self.rot_matrix(self.theta_0) @ (vec_OB_r - vec_upper_rim_r)
        
        # Rotate all the points with beta
        rot_beta = self.rot_matrix(self.beta)
        
        self.A_r = rot_beta @ vec_OA_r
        self.B_r = rot_beta @ vec_OB_r
        self.C_r = rot_beta @ vec_OC_r
        self.D_r = rot_beta @ vec_OD_r
        self.E   = rot_beta @ vec_OE
        self.F_r = rot_beta @ vec_OF_r
        self.G   = rot_beta @ vec_OG
        self.H_r = rot_beta @ vec_OH_r
        self.U_r = rot_beta @ vec_upper_rim_r
        self.L_r = rot_beta @ vec_lower_rim_r
        
        # Reflect all the points through OG
        ang_OG = np.arctan2(self.G[1], self.G[0])
        ref_OG = np.array([[np.cos(2*ang_OG), np.sin(2*ang_OG)], [np.sin(2*ang_OG),  -np.cos(2*ang_OG)]])
        
        self.A_l = ref_OG @ self.A_r
        self.B_l = ref_OG @ self.B_r
        self.C_l = ref_OG @ self.C_r
        self.D_l = ref_OG @ self.D_r
        self.F_l = ref_OG @ self.F_r
        self.H_l = ref_OG @ self.H_r
        self.U_l = ref_OG @ self.U_r
        self.L_l = ref_OG @ self.L_r
        
if __name__ == '__main__':
    leg = Leg()
    
    theta = 17
    beta = 0
    
    leg.set_tb(theta=np.deg2rad(theta), beta=np.deg2rad(beta))
    print(leg.G)
    