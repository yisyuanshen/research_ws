import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation


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
        U_r = vec_OB_r + rot_B_upper_C @ vec_BC_r * self.R / np.linalg.norm(vec_BC_r)
        
        # L
        ang_F_lower_G = -self.n_HF / 2
        rot_F_lower_G = self.rot_matrix(ang_F_lower_G)
        vec_FG_r = vec_OG - vec_OF_r
        L_r = vec_OF_r + rot_F_lower_G @ vec_FG_r * self.R / np.linalg.norm(vec_FG_r)
        
        # H
        vec_OH_r = U_r + self.rot_matrix(self.theta_0) @ (vec_OB_r - U_r)
        
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
        self.U_r = rot_beta @ U_r
        self.L_r = rot_beta @ L_r
        
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
    
    
    def draw(self, ax):
        ax.clear()
        
        linewidth = 1.5
        
        line_OA_r = patches.Polygon((self.O, self.A_r),   closed=False, edgecolor='cornflowerblue', linewidth=linewidth, label='OA_r')
        line_AB_r = patches.Polygon((self.A_r, self.B_r), closed=False, edgecolor='cornflowerblue', linewidth=linewidth, label='AB_r')
        line_AD_r = patches.Polygon((self.A_r, self.D_r), closed=False, edgecolor='cornflowerblue', linewidth=linewidth, label='AD_r')
        line_DE_r = patches.Polygon((self.D_r, self.E),   closed=False, edgecolor='cornflowerblue', linewidth=linewidth, label='DE_r')
        line_DC_r = patches.Polygon((self.D_r, self.C_r), closed=False, edgecolor='cornflowerblue', linewidth=linewidth, label='DC_r')
                
        line_OA_l = patches.Polygon((self.O, self.A_l),   closed=False, edgecolor='cornflowerblue', linewidth=linewidth, label='OA_l')
        line_AB_l = patches.Polygon((self.A_l, self.B_l), closed=False, edgecolor='cornflowerblue', linewidth=linewidth, label='AB_l')
        line_AD_l = patches.Polygon((self.A_l, self.D_l), closed=False, edgecolor='cornflowerblue', linewidth=linewidth, label='AD_l')
        line_DE_l = patches.Polygon((self.D_l, self.E),   closed=False, edgecolor='cornflowerblue', linewidth=linewidth, label='DE_l')
        line_DC_l = patches.Polygon((self.D_l, self.C_l), closed=False, edgecolor='cornflowerblue', linewidth=linewidth, label='DC_l')
        
        line_UH_l = patches.Polygon((self.U_l, self.H_l), closed=False, edgecolor='indigo', linestyle=':', linewidth=linewidth, label='UH_l')
        line_UF_l = patches.Polygon((self.U_l, self.F_l), closed=False, edgecolor='indigo', linestyle=':', linewidth=linewidth, label='UF_l')
        line_LF_l = patches.Polygon((self.L_l, self.F_l), closed=False, edgecolor='indigo', linestyle=':', linewidth=linewidth, label='LF_l')
        line_LG_l = patches.Polygon((self.L_l, self.G),   closed=False, edgecolor='indigo', linestyle=':', linewidth=linewidth, label='LG_l')
        line_UH_r = patches.Polygon((self.U_r, self.H_r), closed=False, edgecolor='indigo', linestyle=':', linewidth=linewidth, label='UH_r')
        line_UF_r = patches.Polygon((self.U_r, self.F_r), closed=False, edgecolor='indigo', linestyle=':', linewidth=linewidth, label='UF_r')
        line_LF_r = patches.Polygon((self.L_r, self.F_r), closed=False, edgecolor='indigo', linestyle=':', linewidth=linewidth, label='LF_r')
        line_LG_r = patches.Polygon((self.L_r, self.G),   closed=False, edgecolor='indigo', linestyle=':', linewidth=linewidth, label='LG_r')
        
        line_beta = patches.Polygon((-self.G*1, self.G*1.3), closed=False, edgecolor='orange', linewidth=linewidth, label='OG', linestyle='-.')
        
        arc_HF_angle_r = self.F_r - self.U_r
        arc_FG_angle_r = self.G   - self.L_r
        arc_HF_angle_l = self.F_l - self.U_l
        arc_FG_angle_l = self.G   - self.L_l
        angle_G = np.rad2deg(np.arctan2(self.G[1], self.G[0])-np.arctan2(self.G[1]-self.L_l[1], self.G[0]-self.L_l[0]))
        
        arc_HF_r = patches.Arc(self.U_r, 2 * self.R, 2 * self.R, angle=np.rad2deg(np.arctan2(arc_HF_angle_r[1], arc_HF_angle_r[0])), 
                               theta1=0, theta2=np.rad2deg(self.n_HF), edgecolor='red', linewidth=linewidth, label='Upper_Rim_r')
        arc_FG_r = patches.Arc(self.L_r, 2 * self.R, 2 * self.R, angle=np.rad2deg(np.arctan2(arc_FG_angle_r[1], arc_FG_angle_r[0])), 
                               theta1=0, theta2=np.rad2deg(np.pi-self.n_HF), edgecolor='black', linewidth=linewidth, label='Lower_Rim_r')
        arc_HF_l = patches.Arc(self.U_l, 2 * self.R, 2 * self.R, angle=np.rad2deg(np.arctan2(arc_HF_angle_l[1], arc_HF_angle_l[0])), 
                               theta1=-np.rad2deg(self.n_HF), theta2=0, edgecolor='red', linewidth=linewidth, label='Upper_Rim_l')
        arc_FG_l = patches.Arc(self.L_l, 2 * self.R, 2 * self.R, angle=np.rad2deg(np.arctan2(arc_FG_angle_l[1], arc_FG_angle_l[0])), 
                               theta1=-np.rad2deg(np.pi-self.n_HF), theta2=0, edgecolor='black', linewidth=linewidth, label='Lower_Rim_l')
        
        arc_HF_r_outer = patches.Arc(self.U_r, 2 * (self.R + self.r), 2 * (self.R + self.r), angle=np.rad2deg(np.arctan2(arc_HF_angle_r[1], arc_HF_angle_r[0])), 
                                     theta1=0, theta2=np.rad2deg(self.n_HF), edgecolor='red', linewidth=linewidth, label='Upper_Rim_r')
        arc_FG_r_outer = patches.Arc(self.L_r, 2 * (self.R + self.r), 2 *(self.R + self.r), angle=np.rad2deg(np.arctan2(arc_FG_angle_r[1], arc_FG_angle_r[0])), 
                                     theta1=0, theta2=np.rad2deg(np.pi-self.n_HF), edgecolor='black', linewidth=linewidth, label='Lower_Rim_r')
        arc_HF_l_outer = patches.Arc(self.U_l, 2 * (self.R + self.r), 2 * (self.R + self.r), angle=np.rad2deg(np.arctan2(arc_HF_angle_l[1], arc_HF_angle_l[0])), 
                                     theta1=-np.rad2deg(self.n_HF), theta2=0, edgecolor='red', linewidth=linewidth, label='Upper_Rim_l')
        arc_FG_l_outer = patches.Arc(self.L_l, 2 * (self.R + self.r), 2 * (self.R + self.r), angle=np.rad2deg(np.arctan2(arc_FG_angle_l[1], arc_FG_angle_l[0])), 
                                     theta1=-np.rad2deg(np.pi-self.n_HF), theta2=0, edgecolor='black', linewidth=linewidth, label='Lower_Rim_l')
        
        arc_HF_r_inner = patches.Arc(self.U_r, 2 * (self.R - self.r), 2 * (self.R - self.r), angle=np.rad2deg(np.arctan2(arc_HF_angle_r[1], arc_HF_angle_r[0])), 
                                     theta1=0, theta2=np.rad2deg(self.n_HF), edgecolor='red', linewidth=linewidth, label='Upper_Rim_r')
        arc_FG_r_inner = patches.Arc(self.L_r, 2 * (self.R - self.r), 2 *(self.R - self.r), angle=np.rad2deg(np.arctan2(arc_FG_angle_r[1], arc_FG_angle_r[0])), 
                                     theta1=0, theta2=np.rad2deg(np.pi-self.n_HF), edgecolor='black', linewidth=linewidth, label='Lower_Rim_r')
        arc_HF_l_inner = patches.Arc(self.U_l, 2 * (self.R - self.r), 2 * (self.R - self.r), angle=np.rad2deg(np.arctan2(arc_HF_angle_l[1], arc_HF_angle_l[0])), 
                                     theta1=-np.rad2deg(self.n_HF), theta2=0, edgecolor='red', linewidth=linewidth, label='Upper_Rim_l')
        arc_FG_l_inner = patches.Arc(self.L_l, 2 * (self.R - self.r), 2 * (self.R - self.r), angle=np.rad2deg(np.arctan2(arc_FG_angle_l[1], arc_FG_angle_l[0])), 
                                     theta1=-np.rad2deg(np.pi-self.n_HF), theta2=0, edgecolor='black', linewidth=linewidth, label='Lower_Rim_l')
        
        
        arc_G_outer = patches.Arc(self.G, 2 * self.r, 2 * self.r, angle=np.rad2deg(np.arctan2(self.G[1], self.G[0])), 
                                     theta1=-angle_G-0.001, theta2=angle_G+0.001, edgecolor='red', linewidth=linewidth, label='Arc G')

        # create end point G
        circle_F_r = patches.Circle(self.F_r, self.r, edgecolor='green', linewidth=linewidth, facecolor='none', label='F_r')
        circle_F_l = patches.Circle(self.F_l, self.r, edgecolor='green', linewidth=linewidth, facecolor='none', label='F_l')
        circle_G = patches.Circle(self.G, self.r, edgecolor='green', linewidth=linewidth, facecolor='none', label='G')

        # axis
        ax.annotate('', xy=(0.19, 0), xytext=(-0.19, 0), arrowprops=dict(facecolor='gray', edgecolor='gray', shrink=0.0, width=0.5, zorder=1))
        ax.annotate('', xy=(0, 0.19), xytext=(0, -0.19), arrowprops=dict(facecolor='gray', edgecolor='gray', shrink=0.0, width=0.5, zorder=1))
        
        # labels
        arrow_contact_r = patches.FancyArrowPatch((0.005, -0.15), (self.F_r/np.linalg.norm(self.F_r)*0.15), connectionstyle=f"arc3,rad=0.24", facecolor='orange', edgecolor='orange', linewidth=linewidth*1.5, linestyle='-', arrowstyle="->", mutation_scale=20)
        arrow_contact_l = patches.FancyArrowPatch((-0.005, -0.15), (self.F_l/np.linalg.norm(self.F_l)*0.15), connectionstyle=f"arc3,rad=-0.24", facecolor='orange', edgecolor='orange', linewidth=linewidth*1.5, linestyle='-', arrowstyle="->", mutation_scale=20)
        
        circle_alpha_0 = patches.Circle((0, -self.R-self.r), 0.004, edgecolor='indigo', linewidth=linewidth, facecolor='indigo', label='', zorder=10)
        circle_alpha_60 = patches.Circle(((self.R+self.r)*np.cos(np.pi/6), -(self.R+self.r)*np.sin(np.pi/6)), 0.004, edgecolor='indigo', linewidth=linewidth, facecolor='indigo', label='', zorder=10)
        circle_alpha_90 = patches.Circle((-self.R-self.r, 0), 0.004, edgecolor='indigo', linewidth=linewidth, facecolor='indigo', label='', zorder=10)
        circle_alpha_120 = patches.Circle(((self.R+self.r)*np.cos(np.pi/6), (self.R+self.r)*np.sin(np.pi/6)), 0.004, edgecolor='indigo', linewidth=linewidth, facecolor='indigo', label='', zorder=10)
        
        # '''
        # TB = (17, 0)
        ax.text( 0.120, -0.130, r'$+\alpha$', ha='center', va='center')
        ax.text(-0.120, -0.130, r'$-\alpha$', ha='center', va='center')
        ax.text( 0.006, -0.130, r'$\alpha=0$', ha='center', va='center')
        ax.text( 0.135, -0.060, r'$\displaystyle \alpha=\frac{\pi}{3}$', ha='center', va='center')
        ax.text( -0.150, 0.020, r'$\displaystyle \alpha=\frac{-\pi}{2}$', ha='center', va='center')
        ax.text( 0.135,  0.055, r'$\displaystyle \alpha=\frac{2\pi}{3}$', ha='center', va='center')
        # '''
        
        
        arc_beta = patches.Arc(self.O, 0.24, 0.24, theta1=90, theta2=90+beta, linestyle='--', edgecolor='black', linewidth=linewidth)
        arc_theta = patches.Arc(self.O, 0.185, 0.185, theta1=90+beta, theta2=90+beta+theta, linestyle='--', edgecolor='black', linewidth=linewidth)
        arc_phi_r = patches.Arc(self.O, 0.08, 0.08, theta1=90+beta-theta-self.theta_0, theta2=90, linestyle='--', edgecolor='black', linewidth=linewidth)
        arc_phi_l = patches.Arc(self.O, 0.11, 0.11, theta1=90, theta2=90+beta+theta+self.theta_0, linestyle='--', edgecolor='black', linewidth=linewidth)
        
        '''
        # TB = (70, 40)
        ax.text( 0.014,  0.052, r'$\phi_R$', ha='center', va='center')
        ax.text(-0.058,  0.040, r'$\phi_L$', ha='center', va='center')
        ax.text(-0.045,  0.125, r'$\theta$', ha='center', va='center')
        ax.text(-0.090,  0.055, r'$\beta$', ha='center', va='center')
        ax.text( 0.018,  0.008, r'$\mathbf{O}$', ha='center', va='center')
        ax.text(-0.110,  0.010, r'$\mathbf{H_L}$', ha='center', va='center')
        ax.text(-0.122, -0.040, r'$\mathbf{A_L}$', ha='center', va='center')
        ax.text( 0.013, -0.035, r'$\mathbf{U_L}$', ha='center', va='center')
        ax.text( 0.040, -0.170, r'$\mathbf{F_L}$', ha='center', va='center')
        ax.text( 0.082, -0.047, r'$\mathbf{L_L}$', ha='center', va='center')
        ax.text( 0.130, -0.175, r'$\mathbf{G}$', ha='center', va='center')
        ax.text( 0.160, -0.070, r'$\mathbf{F_R}$', ha='center', va='center')
        ax.text( 0.057,  0.110, r'$\mathbf{A_R}$', ha='center', va='center')
        ax.text( 0.020,  0.110, r'$\mathbf{H_R}$', ha='center', va='center')
        # '''
        
        # Add all elements
        ax.add_patch(line_OA_r)
        ax.add_patch(line_AB_r)
        ax.add_patch(line_AD_r)
        ax.add_patch(line_DE_r)
        ax.add_patch(line_DC_r)
        
        ax.add_patch(line_OA_l)
        ax.add_patch(line_AB_l)
        ax.add_patch(line_AD_l)
        ax.add_patch(line_DE_l)
        ax.add_patch(line_DC_l)
        
        
        # ax.add_patch(arc_HF_r)
        # ax.add_patch(arc_FG_r)
        # ax.add_patch(arc_HF_l)
        # ax.add_patch(arc_FG_l)
        
        ax.add_patch(arc_HF_r_inner)
        ax.add_patch(arc_FG_r_inner)
        ax.add_patch(arc_HF_l_inner)
        ax.add_patch(arc_FG_l_inner)
        
        ax.add_patch(arc_HF_r_outer)
        ax.add_patch(arc_FG_r_outer)
        ax.add_patch(arc_HF_l_outer)
        ax.add_patch(arc_FG_l_outer)
        
        ax.add_patch(circle_F_r)
        ax.add_patch(circle_F_l)
        ax.add_patch(circle_G)
        # ax.add_patch(arc_G_outer)
        
        
        ax.add_patch(arrow_contact_r)
        ax.add_patch(arrow_contact_l)
        ax.add_patch(circle_alpha_0)
        ax.add_patch(circle_alpha_60)
        ax.add_patch(circle_alpha_90)
        ax.add_patch(circle_alpha_120)
        
        # '''
        # ax.add_patch(line_UH_l)
        # ax.add_patch(line_UF_l)
        # ax.add_patch(line_LF_l)
        # ax.add_patch(line_LG_l)
        # ax.add_patch(line_UH_r)
        # ax.add_patch(line_UF_r)
        # ax.add_patch(line_LF_r)
        # ax.add_patch(line_LG_r)
        
        # ax.add_patch(line_beta)
        # ax.add_patch(arc_beta)
        # ax.add_patch(arc_theta)
        # ax.add_patch(arc_phi_r)
        # ax.add_patch(arc_phi_l)
        # '''
        
        # Setup the ax
        
        ax.set_aspect('equal')
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.2, 0.2)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Leg Contact Angle Definition')
        # ax.set_title('Leg Configuration')
        
        plt.xticks(np.arange(-2, 3, 1)/10, fontsize=11)
        plt.yticks(np.arange(-2, 3, 1)/10, fontsize=11)
        
        
    def plot_once(self):
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{mathptmx}')
        plt.rcParams.update({'font.size': 11})
        plt.rcParams['font.family'] = 'DeJavu Serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        fig, ax = plt.subplots()
        self.draw(ax)
        plt.savefig('LegPlot.pdf', format='pdf', bbox_inches='tight')
        plt.show()
        
        
    def plot_animation(self, tb_list=None):
        plt.rcParams['font.family'] = 'DeJavu Serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        fig, ax = plt.subplots(figsize=(6, 6))
        self.tb_list = tb_list
        frames = len(self.tb_list)
        ani = FuncAnimation(fig, self.update_animation, frames=frames, fargs=(ax,), interval=1)
        plt.show()
        
        
    def update_animation(self, frame, ax):
        theta, beta = self.tb_list[frame]
        self.set_tb(theta, beta)
        self.draw(ax)
        
        
if __name__ == '__main__':
    leg = Leg()
    
    theta = 17
    beta = 0
    
    leg.set_tb(theta=np.deg2rad(theta), beta=np.deg2rad(beta))
    
    leg.plot_once()
    
    '''
    tb_list = []
    for i in range(360):
        theta = 45 * (np.cos(i/45*np.pi)+1)
        if i < 180:beta = -90 - 50 * (np.cos(i/90*np.pi)-1)
        else: beta = -90 + 50 * (np.cos(i/90*np.pi)-1)
        
        theta = np.deg2rad(theta) + leg.theta_0 
        beta = np.deg2rad(beta) + leg.beta_0
        
        tb_list.append((theta, beta))
    
    leg.plot_animation(tb_list=tb_list)
    # '''