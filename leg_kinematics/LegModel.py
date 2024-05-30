import numpy as np
from numpy import pi, sin, cos, arcsin, arccos, arctan2, deg2rad
from numpy.linalg import norm

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

class Leg:
    def __init__(self):
        self.theta = 0
        self.beta = 0
        
        self.theta_0 = deg2rad(17)
        self.beta_0 = deg2rad(90)
        
        self.R = 0.1  # wheel radius
        self.r = 0.0125  # rim thickness
        self.n_HF = deg2rad(130)
        self.n_BC = deg2rad(101)
        
        self.l1 = 0.8 * self.R
        self.l2 = self.R - self.l1
        self.l3 = 2 * self.R * sin(self.n_BC / 2)
        self.l4 = 0.88296634 * self.R
        self.l5 = 0.9 * self.R
        self.l6 = 0.4 * self.R
        self.l7 = 2 * self.R * sin((self.n_HF - self.n_BC - self.theta_0) / 2)
        self.l8 = 2 * self.R * sin((pi - self.n_HF) / 2)
        
        self.set_tb(theta=self.theta_0, beta=self.beta_0)
        
        
    def set_tb(self, theta, beta):
        self.theta = theta
        self.beta = beta
        self.calculate_forward()
        
    def rot_matrix(self, angle):
        return np.array([[cos(angle), -sin(angle)], [sin(angle),  cos(angle)]])

    
    def calculate_forward(self):
        self.vec_O = np.array([0, 0])
        
        vec_OA_r = self.l1 * np.array([cos(-self.theta), sin(-self.theta)])
        vec_OB_r = (self.l1 + self.l2) * np.array([cos(-self.theta), sin(-self.theta)])
        
        ang_AOE = pi - self.theta
        ang_OAE = pi - ang_AOE - arcsin(self.l1 * sin(ang_AOE) / (self.l5 + self.l6))
        rot_OAE = self.rot_matrix(ang_OAE)
        vec_AE_r = rot_OAE @ (-vec_OA_r) * (self.l5 + self.l6) / norm(vec_OA_r)
        vec_OE = vec_OA_r + vec_AE_r

        vec_OD_r = (self.l5 * vec_OE + self.l6 * vec_OA_r) / (self.l5 + self.l6)

        vec_BD_r = vec_OD_r - vec_OB_r
        ang_DBC = arccos((self.l3**2 + norm(vec_BD_r)**2 - self.l4**2) / (2 * self.l3 * norm(vec_BD_r)))
        rot_DBC = self.rot_matrix(ang_DBC)
        vec_BC_r = rot_DBC @ vec_BD_r * self.l3 / norm(vec_BD_r)
        vec_OC_r = vec_OB_r + vec_BC_r

        arc_BF = self.n_HF - self.theta_0
        ang_BCF = arccos((self.l3**2 + self.l7**2 - 2 * self.R**2 + 2 * self.R**2 * cos(arc_BF)) / (2 * self.l3 * self.l7))
        rot_BCF = self.rot_matrix(ang_BCF)
        vec_CB_r = vec_OB_r - vec_OC_r
        vec_CF_r = rot_BCF @ vec_CB_r * self.l7 / norm(vec_CB_r)
        vec_OF_r = vec_OC_r + vec_CF_r

        ang_GOF = arccos(vec_OE @ vec_OF_r / norm(vec_OE) / norm(vec_OF_r))
        ang_OGF = arcsin(norm(vec_OF_r) * sin(ang_GOF) / self.l8)
        ang_OFG = pi - ang_GOF - ang_OGF
        norm_OG = self.l8 * sin(ang_OFG) / sin(ang_GOF)
        vec_OG = vec_OE * norm_OG / norm(vec_OE)
        
        
        ang_B_upper_C = -(pi - self.n_BC) / 2
        rot_B_upper_C = self.rot_matrix(ang_B_upper_C)
        vec_upper_rim_r = vec_OB_r + rot_B_upper_C @ vec_BC_r * self.R / norm(vec_BC_r)
        
        ang_F_lower_G = -self.n_HF / 2
        rot_F_lower_G = self.rot_matrix(ang_F_lower_G)
        vec_FG_r = vec_OG - vec_OF_r
        vec_lower_rim_r = vec_OF_r + rot_F_lower_G @ vec_FG_r * self.R / norm(vec_FG_r)

        rot_theta_0 = self.rot_matrix(self.theta_0)
        vec_OH_r = vec_upper_rim_r + rot_theta_0 @ (vec_OB_r - vec_upper_rim_r)
        
        
        rot_beta = self.rot_matrix(self.beta)

        self.vec_OA_r = rot_beta @ vec_OA_r
        self.vec_OB_r = rot_beta @ vec_OB_r
        self.vec_OC_r = rot_beta @ vec_OC_r
        self.vec_OD_r = rot_beta @ vec_OD_r
        self.vec_OE   = rot_beta @ vec_OE
        self.vec_OF_r = rot_beta @ vec_OF_r
        self.vec_OG   = rot_beta @ vec_OG
        self.vec_OH_r = rot_beta @ vec_OH_r
        self.vec_upper_rim_r = rot_beta @ vec_upper_rim_r
        self.vec_lower_rim_r = rot_beta @ vec_lower_rim_r
        
        ang_OG = arctan2(self.vec_OG[1], self.vec_OG[0])
        ref_OG = np.array([[cos(2*ang_OG), sin(2*ang_OG)], [sin(2*ang_OG),  -cos(2*ang_OG)]])
        
        self.vec_OA_l = ref_OG @ self.vec_OA_r
        self.vec_OB_l = ref_OG @ self.vec_OB_r
        self.vec_OC_l = ref_OG @ self.vec_OC_r
        self.vec_OD_l = ref_OG @ self.vec_OD_r
        self.vec_OF_l = ref_OG @ self.vec_OF_r
        self.vec_OH_l = ref_OG @ self.vec_OH_r
        self.vec_upper_rim_l = ref_OG @ self.vec_upper_rim_r
        self.vec_lower_rim_l = ref_OG @ self.vec_lower_rim_r
        
          
    def draw(self, ax):
        ax.clear()
        
        linewidth = 1.5
        
        line_OA_r = patches.Polygon((self.vec_O,    self.vec_OA_r), closed=False, edgecolor='blue', linewidth=linewidth, label='OA_r')
        line_AB_r = patches.Polygon((self.vec_OA_r, self.vec_OB_r), closed=False, edgecolor='blue', linewidth=linewidth, label='AB_r')
        line_AD_r = patches.Polygon((self.vec_OA_r, self.vec_OD_r), closed=False, edgecolor='blue', linewidth=linewidth, label='AD_r')
        line_DE_r = patches.Polygon((self.vec_OD_r, self.vec_OE),   closed=False, edgecolor='blue', linewidth=linewidth, label='DE_r')
        line_DC_r = patches.Polygon((self.vec_OD_r, self.vec_OC_r), closed=False, edgecolor='blue', linewidth=linewidth, label='DC_r')
                
        line_OA_l = patches.Polygon((self.vec_O,    self.vec_OA_l), closed=False, edgecolor='blue', linewidth=linewidth, label='OA_l')
        line_AB_l = patches.Polygon((self.vec_OA_l, self.vec_OB_l), closed=False, edgecolor='blue', linewidth=linewidth, label='AB_l')
        line_AD_l = patches.Polygon((self.vec_OA_l, self.vec_OD_l), closed=False, edgecolor='blue', linewidth=linewidth, label='AD_l')
        line_DE_l = patches.Polygon((self.vec_OD_l, self.vec_OE),   closed=False, edgecolor='blue', linewidth=linewidth, label='DE_l')
        line_DC_l = patches.Polygon((self.vec_OD_l, self.vec_OC_l), closed=False, edgecolor='blue', linewidth=linewidth, label='DC_l')
        
        arc_HF_angle_r = self.vec_OF_r - self.vec_upper_rim_r
        arc_FG_angle_r = self.vec_OG   - self.vec_lower_rim_r
        arc_HF_angle_l = self.vec_OF_l - self.vec_upper_rim_l
        arc_FG_angle_l = self.vec_OG   - self.vec_lower_rim_l
        
        arc_HF_r = patches.Arc(self.vec_upper_rim_r, 2 * self.R, 2 * self.R, angle=np.rad2deg(np.arctan2(arc_HF_angle_r[1], arc_HF_angle_r[0])), 
                               theta1=0, theta2=np.rad2deg(self.n_HF), edgecolor='red', linewidth=linewidth, label='Upper_Rim_r')
        arc_FG_r = patches.Arc(self.vec_lower_rim_r, 2 * self.R, 2 * self.R, angle=np.rad2deg(np.arctan2(arc_FG_angle_r[1], arc_FG_angle_r[0])), 
                               theta1=0, theta2=np.rad2deg(np.pi-self.n_HF), edgecolor='black', linewidth=linewidth, label='Lower_Rim_r')
        arc_HF_l = patches.Arc(self.vec_upper_rim_l, 2 * self.R, 2 * self.R, angle=np.rad2deg(np.arctan2(arc_HF_angle_l[1], arc_HF_angle_l[0])), 
                               theta1=-np.rad2deg(self.n_HF), theta2=0, edgecolor='red', linewidth=linewidth, label='Upper_Rim_l')
        arc_FG_l = patches.Arc(self.vec_lower_rim_l, 2 * self.R, 2 * self.R, angle=np.rad2deg(np.arctan2(arc_FG_angle_l[1], arc_FG_angle_l[0])), 
                               theta1=-np.rad2deg(np.pi-self.n_HF), theta2=0, edgecolor='black', linewidth=linewidth, label='Lower_Rim_l')
        
        arc_HF_r_outer = patches.Arc(self.vec_upper_rim_r, 2 * (self.R + self.r), 2 * (self.R + self.r), angle=np.rad2deg(np.arctan2(arc_HF_angle_r[1], arc_HF_angle_r[0])), 
                                     theta1=0, theta2=np.rad2deg(self.n_HF), edgecolor='red', linewidth=linewidth, label='Upper_Rim_r')
        arc_FG_r_outer = patches.Arc(self.vec_lower_rim_r, 2 * (self.R + self.r), 2 *(self.R + self.r), angle=np.rad2deg(np.arctan2(arc_FG_angle_r[1], arc_FG_angle_r[0])), 
                                     theta1=0, theta2=np.rad2deg(np.pi-self.n_HF), edgecolor='black', linewidth=linewidth, label='Lower_Rim_r')
        arc_HF_l_outer = patches.Arc(self.vec_upper_rim_l, 2 * (self.R + self.r), 2 * (self.R + self.r), angle=np.rad2deg(np.arctan2(arc_HF_angle_l[1], arc_HF_angle_l[0])), 
                                     theta1=-np.rad2deg(self.n_HF), theta2=0, edgecolor='red', linewidth=linewidth, label='Upper_Rim_l')
        arc_FG_l_outer = patches.Arc(self.vec_lower_rim_l, 2 * (self.R + self.r), 2 * (self.R + self.r), angle=np.rad2deg(np.arctan2(arc_FG_angle_l[1], arc_FG_angle_l[0])), 
                                     theta1=-np.rad2deg(np.pi-self.n_HF), theta2=0, edgecolor='black', linewidth=linewidth, label='Lower_Rim_l')
        
        # arc_G_outer = patches.Arc(self.vec_OG, 2 * self.r, 2 * self.r, angle=np.rad2deg(np.arctan2(self.OG.pos[1], self.OG.pos[0])), 
        #                              theta1=-np.rad2deg(self.contact_angle_G_max), theta2=np.rad2deg(self.contact_angle_G_max), edgecolor='red', linewidth=linewidth, label='Arc G')
        
        # create end point G
        circle_G = patches.Circle(self.vec_OG, self.r, edgecolor='red', linewidth=linewidth, facecolor='none', label='G')
        
        # circle_observe = patches.Circle(self.observe_point.pos, 0.005, edgecolor='darkblue', linewidth=linewidth, facecolor='none', label='G')
        
        # create velocity arrow
        # ax = self.draw_vel(ax)
        
        
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
            
        ax.add_patch(arc_HF_r)
        ax.add_patch(arc_FG_r)
        ax.add_patch(arc_HF_l)
        ax.add_patch(arc_FG_l)
        
        ax.add_patch(arc_HF_r_outer)
        ax.add_patch(arc_FG_r_outer)
        ax.add_patch(arc_HF_l_outer)
        ax.add_patch(arc_FG_l_outer)
        
        ax.add_patch(circle_G)
        # ax.add_patch(arc_G_outer)
        # ax.add_patch(circle_observe)
        
        
        # Setup the ax
        ax.set_aspect('equal')
        ax.set_xlim(-0.35, 0.35)
        ax.set_ylim(-0.35, 0.35)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title('Leg Kinematics')
        
        
    def plot_once(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        self.draw(ax)
        plt.show()
        
        
    def plot_animation(self, tb_list=None):
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
    beta = 30
    
    theta = deg2rad(theta) + leg.theta_0
    beta = deg2rad(beta) + leg.beta_0
    
    leg.set_tb(theta=theta, beta=beta)
    
    leg.plot_once()
    
    
    '''
    tb_list = []
    for i in range(360):
        theta = 30 * (np.sin(i/30*np.pi)+1)
        beta = 240 * (np.sin(i/180*np.pi))
        
        theta = np.deg2rad(theta) + leg.theta_0 
        beta = np.deg2rad(beta) + leg.beta_0
        
        tb_list.append((theta, beta))
    
    leg.plot_animation(tb_list=tb_list)
    '''