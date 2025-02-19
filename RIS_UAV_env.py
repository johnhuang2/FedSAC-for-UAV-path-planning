#-- coding:UTF-8 --
"""
Reinforcement learning RIS-assisted UAV system.
Basical setting:
AoI:                1000m*1000m;
Height_levels, maximal height, minimula height:
                    50, 200m, 100m;

This script is the environment part of the RIS-assisted UAV system.
The RL is in RL_brain.py.

View more on my information see paper: "3D-Trajectory Design and Phase-Shift for RIS-Assisted UAV Communication using Deep Reinforcement Learning"
by Haibo Mei, Kun Yang, Qiang Liu, Kezhi Wang;
"""
import numpy as np
import random as rd
import time
import math as mt
import sys
import copy
from scipy.interpolate import splrep
from mpl_toolkits.mplot3d import Axes3D
import matplotlib


# if sys.version_info.major == 2:
#     import Tkinter as tk
# else:
#     import tkinter as tk
# import matplotlib.pyplot as plt

UNIT =  1              # pixels
IOT_H = 500         # grid height
IOT_W = 500         # grid width
Max_Hight = 200        # maximum level of height
Min_Hight = 0        # minimum level of height


#gradient of the horizontal and vertical locations of the UAV
D_k = 1024
t_min = 1
t_max = 3
# Initialize the wireless environement and some other verables.
B = 2000  # overall Bandwith is 2Gb;
#N_0 = mt.pow(10, (-169  / 10))*(0.1 * mt.pow(10, 3))/B  # Noise power spectrum density is -169dBm/Hz;
N_0 =mt.pow(10, ((-169 / 3) / 10))
Xi =  mt.pow(10, (3/10)) #the path loss at the reference distance D0 = 1m, 3dB;
a = 9.61
b = 0.16  # and paper [Optimal LAP Altitude for Maximum Coverage, IEEE WIRELESS COMMUNICATIONS LETTERS, VOL. 3, NO. 6, DECEMBER 2014]
eta_los = 0.01  # Loss corresponding to the LoS connections defined in (2) of the paper;
eta_nlos = 0.2  # Loss corresponding to the NLoS connections defined in (2) of the paper;
A = eta_los - eta_nlos  # A varable defined in (2) of the paper;
C = 0.2 * np.log10(
    4 * np.pi * 9 / 3) + eta_nlos  # C varable defined in (2)of the paper, where carrier frequncy is 900Mhz=900*10^6, and light speed is c=3*1
Power = 0.5 * mt.pow(10, 3)  # maximum uplink transimission power of one GT is 500mW;
#RIS setting
W_R = [250,250]  #horizontal location of RIS
Z_R = 100     #height of RIS
M = 100  #phase element
data = 10

class RIS_UAV(object):
    def __init__(self):
        super(RIS_UAV, self).__init__()
        self.N_slot = 600  # number of time slots in one episode
        self.GTs = 20
        self.l_o_v = 200  # initial vertical location
        self.l_f_v = 0  # final vertical location
        self.l_o_h = [0, 0]  # initial horizontal location
        self.l_f_h = [500, 500]  # final horizontal location
        self.eps = 60   # number of episodes
        self.beta_0 = 1  # Channel power gain at reference distance 1m
        self.c = 3e8  # Speed of light
        self.f_c = 2e9  # Carrier frequency
        self.Xi = 1  # Some constant
        self.B = 2000  # Bandwidth
        self.N_0 = 1e-9  # Noise power spectral density
        self.M_r = 10  # Number of reflecting elements in row
        self.M_c = 10  # Number of reflecting elements in column
        self.alpha_k_RG = 2  # Path loss exponent for RIS-to-user link
        self.kappa_k_RG = 10  # Rician factor for RIS-to-user link
        self.alpha_k_UG = 2  # Path loss exponent for UAV-to-user link
        self.kappa_k_UG = 10  # Rician factor for UAV-to-user link
        
        self.Z_R = 50  # Height of RIS
        self.W_R = [250, 250]  # Location of RIS
        self.Power = 1  # Transmission power
        self.A = 1  # Path loss factor
        self.C = 1  # Path loss factor
        self.delta_t = 1  # Time slot duration
        self.p_TX = 1  # UAV transmit power
        self.sigma = np.sqrt(3.98e-6)  # 或者根据具体的带宽计算出来的噪声功率
        self.N_f = 64  # Number of sub-carriers
        self.P = 0.3 #0.3
        
        self.cumulative_d_s = np.zeros(self.GTs, dtype=float)  # 每个 GT 的累计数据量
        self.energy = None  # 在 _build_ris_uav 或 reset 方法中初始化
        self.cumulative_energy = 0.0
        self.previous_distance_to_target = None
        self.previous_distance_to_charging = None
        self.finish = False
        # north, south, east, west, hover
        self.action_space_uav_horizontal = ['n', 's', 'e', 'w', 'h']
        # ascend, descend, slf
        self.action_space_uav_vertical = ['a', 'd', 's']

        # overall_action_space
        self.n_actions = len(self.action_space_uav_horizontal) * len(self.action_space_uav_vertical) * self.GTs * (int(t_max/0.1)-int(t_min/0.1)+1)
        self.n_features = 4   # horizontal, vertical trajectory of the UAV

        # generate action table
        self.actions = np.zeros((int(self.n_actions), 1+4), dtype=int)
        index = 0
        for h in range(len(self.action_space_uav_horizontal)):
            for v in range(len(self.action_space_uav_vertical)):
                for s in range(self.GTs):
                    for t in range(int(t_min/0.5), int((t_max)/0.5)+1):
                        self.actions[index, :] = [index, h, v, s, t]
                        index = index + 1
        self._build_ris_uav()


    def _build_ris_uav(self):
            self.d_s = np.zeros((self.N_slot, self.GTs), dtype=float)
            self.energy = np.zeros(self.N_slot, dtype=float)
            self.w_k = np.zeros((self.GTs, 2), dtype=float)
            self.u_k = np.zeros((self.GTs, 1), dtype=float)
            
            # gt_coordinates = np.array([
            #     [100, 400],
            #     [100, 100],
            #     [400, 400],
            #     [400, 100],
            # ])

            gt_coordinates = np.array([[100, 400],
                                        [200, 300],
                                        [400, 400],
                                        [400, 100],
                                        [400, 250],
                                        [250, 150],
                                        [ 70, 440],
                                        [200, 200],
                                        [100, 300],
                                        [300, 300],
                                        [300, 100],
                                        [300, 250],
                                        [250, 100],
                                        [300, 450],
                                        [300, 100],
                                        [300, 250],
                                        [200, 100],
                                        [400,450],
                                        [400,100],
                                        [100,450],]
                                        )
            
            if len(gt_coordinates) != self.GTs:
                raise ValueError("Mismatch in the number of GTs and predefined coordinates.")

            self.w_k = gt_coordinates

            for g in range(self.GTs):
                self.u_k[g, 0] = D_k / 2 + (D_k / 2) * rd.random()

    def reset(self):
        self.cumulative_d_s = np.zeros(self.GTs, dtype=float)
        self.cumulative_energy = 0.0
        self.previous_distance_to_charging = None
        self.d_s = np.zeros((self.N_slot, self.GTs), dtype=float)
        self.energy = np.zeros(self.N_slot, dtype=float)
        self.h_n = 200
        self.l_n = [0, 0]
        self.finish = False
        self.slot = 0
        self.OtPoI = 0

        start_position = np.array([self.l_o_h[0], self.l_o_h[1], self.l_o_v])
        target_position = np.array([self.l_f_h[0], self.l_f_h[1], self.l_f_v])
        distance_to_target = np.linalg.norm(target_position - start_position)

        return np.array([self.l_n[0], self.l_n[1], self.h_n, distance_to_target])

    def step(self, action, slot):
        if slot >= self.N_slot:
            self.finish = True
            return self.reset(), 0

        h = action[1]
        v = action[2]
        c_n = action[3]
        t_n = action[4]

        pre_l_n = self.l_n.copy()
        pre_h_n = self.h_n

        # Update height of the UAV
        self.OtPoI = 0
        if v == 0:  # ascending
            self.h_n = self.h_n + 2
            if self.h_n > Max_Hight:
                self.h_n = self.h_n - 2
                self.OtPoI = 1
        elif v == 1:  # descending
            self.h_n = self.h_n - 2
            if self.h_n < Min_Hight:
                self.h_n = self.h_n + 2               
                self.OtPoI = 1
        elif v == 2:  # SLF
            self.h_n = self.h_n

        # Update horizontal location of the UAV
        if h == 0:  # north
            self.l_n[1] = self.l_n[1] + 6
            if self.l_n[1] > IOT_H:
                self.l_n[1] = self.l_n[1] - 10
                self.OtPoI = 1
        elif h == 1:  # south
            self.l_n[1] = self.l_n[1] - 6
            if self.l_n[1] < 0:
                self.l_n[1] = self.l_n[1] + 10
                self.OtPoI = 1
        elif h == 2:  # east
            self.l_n[0] = self.l_n[0] + 6
            if self.l_n[0] > IOT_W:
                self.l_n[0] = self.l_n[0] - 10
                self.OtPoI = 1
        elif h == 3:  # west
            self.l_n[0] = self.l_n[0] - 6
            if self.l_n[0] < 0:
                self.l_n[0] = self.l_n[0] + 10
                self.OtPoI = 1
        elif h == 4:  # hover
            self.l_n[0] = self.l_n[0]
            self.l_n[1] = self.l_n[1]

        # Energy consumption calculation
        is_ascending = v == 0
        energy_slot = self.flight_energy_slot(pre_l_n, self.l_n, pre_h_n, self.h_n, 2, is_ascending)
        self.energy[slot] = energy_slot

        # Calculate throughput
        current_throughput = 0
        for g in range(self.GTs):
            c_ki = 1 if g == c_n else 0
            if c_ki == 1:
                link_rate = self.link_rate(g, c_ki)
                current_throughput += link_rate * (t_n / 10)

        # Calculate energy efficiency for this step
        ee_reward = 0
        if energy_slot > 0:
            ee_reward = (current_throughput / energy_slot) # Energy efficiency reward

        # Original distance reward calculation
        uav_position = np.array([self.l_n[0], self.l_n[1], self.h_n])
        target_position = np.array([self.l_f_h[0], self.l_f_h[1], self.l_f_v])
        current_distance = np.linalg.norm(uav_position - target_position)

        if self.previous_distance_to_charging is None:
            self.previous_distance_to_charging = current_distance

        distance_delta = self.previous_distance_to_charging - current_distance

        # Combine original rewards with energy efficiency
        reward = distance_delta  # Original distance reward
        reward -= 1  # Original time penalty
        reward -= current_distance * 0.01  # Original distance penalty
        reward += ee_reward  # Add energy efficiency reward

        if current_distance <= 30:
            reward += 1000
            self.finish = True
        else:
            self.finish = False

        if self.OtPoI == 1:
            reward -= 100

        self.previous_distance_to_charging = current_distance

        _state = np.array([self.l_n[0], self.l_n[1], self.h_n, current_distance])

        return _state, reward




    def link_rate(self, gt, c_ki):
        h = self.h_n
        x = self.l_n[0]
        y = self.l_n[1]
        w_k_x = self.w_k[gt, 0]
        w_k_y = self.w_k[gt, 1]
        W_R_x = self.W_R[0]
        W_R_y = self.W_R[1]
        Z_R = self.Z_R

        d_ug = np.sqrt(h ** 2 + (x - w_k_x) ** 2 + (y - w_k_y) ** 2)
        d_ur = np.sqrt((h - Z_R) ** 2 + (W_R_x - x) ** 2 + (W_R_y - y) ** 2)
        d_rg = np.sqrt(Z_R ** 2 + (W_R_x - w_k_x) ** 2 + (W_R_y - w_k_y) ** 2)

        # Calculate angles
        theta_UR = np.arctan2(h - Z_R, np.sqrt((x - W_R_x) ** 2 + (y - W_R_y) ** 2))
        xi_UR = np.arctan2(x - W_R_x, y - W_R_y)
        theta_RG = np.arctan2(Z_R, np.sqrt((W_R_x - w_k_x) ** 2 + (W_R_y - w_k_y) ** 2))
        xi_RG = np.arctan2(W_R_x - w_k_x, W_R_y - w_k_y)

        # Define distances between reflecting elements
        d_r = 0.5  # Distance between reflecting elements in row
        d_c = 0.5  # Distance between reflecting elements in column

        # Calculate LoS components
        h_LoS_UR = np.kron(
            np.exp(-1j * 2 * np.pi * self.f_c * np.arange(self.M_r) * d_r * np.sin(theta_UR) * np.sin(xi_UR) / self.c),
            np.exp(-1j * 2 * np.pi * self.f_c * np.arange(self.M_c) * d_c * np.sin(theta_UR) * np.sin(xi_UR) / self.c)
        )

        # Adjust h_LoS_UR to match N_f shape
        h_LoS_UR = h_LoS_UR[:self.N_f]  # Assuming M_r * M_c >= N_f

        h_i_UR = np.sqrt(self.beta_0 / (d_ur ** 2)) * np.exp(-1j * 2 * np.pi * np.arange(self.N_f) * (d_ur / self.c))[:self.N_f] * h_LoS_UR

        h_LoS_RG = np.kron(
            np.exp(-1j * 2 * np.pi * self.f_c * np.arange(self.M_r) * d_r * np.sin(theta_RG) * np.sin(xi_RG) / self.c),
            np.exp(-1j * 2 * np.pi * self.f_c * np.arange(self.M_c) * d_c * np.sin(theta_RG) * np.sin(xi_RG) / self.c)
        )

        # Adjust h_LoS_RG to match N_f shape
        h_LoS_RG = h_LoS_RG[:self.N_f]  # Assuming M_r * M_c >= N_f

        h_k_i_RG = np.sqrt(self.beta_0 / (d_rg ** self.alpha_k_RG)) * (
            np.sqrt(self.kappa_k_RG / (1 + self.kappa_k_RG)) * np.exp(-1j * 2 * np.pi * np.arange(self.N_f) * (d_rg / self.c))[:self.N_f] * h_LoS_RG +
            np.sqrt(1 / (1 + self.kappa_k_RG)) * np.random.normal(size=self.N_f)  # Adjust size to N_f
        )

        # UAV-to-user link
        h_k_i_UG = np.sqrt(self.beta_0 / (d_ug ** self.alpha_k_UG)) * (
            np.sqrt(self.kappa_k_UG / (1 + self.kappa_k_UG)) * np.exp(-1j * 2 * np.pi * np.arange(self.N_f) * (d_ug / self.c))[:self.N_f] +
            np.sqrt(1 / (1 + self.kappa_k_UG)) * np.random.normal(size=self.N_f)
        )

        # RIS reflection matrix
        Phi = np.diag(np.exp(1j * np.random.uniform(-np.pi, np.pi, self.N_f)))  # Adjust size to N_f

        # Composite channel gain
        h_k_i_URG = self.Xi * (h_k_i_RG.T @ Phi @ h_i_UR)

        # Estimation of composite channel gain
        h_prime = h_k_i_UG + h_k_i_URG
        tilde_h = np.random.normal(size=h_prime.shape)
        g_k_i_UG = np.sqrt(1 - self.P) * h_prime + np.sqrt(self.P) * tilde_h

        # Calculate data rate
        R_k_i_n = c_ki * self.B * np.log2(1 + self.p_TX * np.abs(g_k_i_UG).mean() ** 2 / self.sigma**2)  # Take the mean to return a scalar
        
        return R_k_i_n / 1e3


    

    def find_action(self, index):
        return self.actions[index,:]

    def flight_energy_slot(self, pre_l_n, l_n, pre_h, h, t_n, is_ascending):
        epsilon = 1e-6  # 用于避免零值的常数
        t_n = max(t_n, epsilon)  # 确保 t_n 不为零

        d_o = 0.6  # 阻力比
        SF = 0.5   # 机身等效平板面积
        rho = 1.225  # 空气密度，单位kg/m³
        s = 0.05  # 旋翼实度
        G = 0.503  # 旋翼盘面积，单位平方米
        U_tip = 120  # 旋翼叶片尖端速度，单位米/秒
        v_o = 4.3  # 悬停时平均旋翼诱导速度，单位米/秒
        omega = 300  # 旋翼叶片角速度，单位弧度/秒
        R = 0.4  # 旋翼半径，单位米
        delta = 0.012  # 阻力系数
        k = 0.1  # 诱导功率增量修正系数
        W = 20  # 总重
        P0 = 4 * ((delta / 8) * rho * s * G * (pow(omega, 3)) * (pow(R, 3)) +
                (1 + k) * (pow((W / 4), (3 / 2)) / np.sqrt(2 * rho * G)))
        P1 = (1 + k) * (pow((W / 4), (3 / 2)) / np.sqrt(2 * rho * G))
        PB = (delta / 8) * rho * s * G * (pow(omega, 3)) * (pow(R, 3))
        P2 = 11.46

        x_pre = pre_l_n[0]
        y_pre = pre_l_n[1]
        z_pre = pre_h
        x = l_n[0]
        y = l_n[1]
        z = h

        d = np.sqrt((x_pre - x) ** 2 + (y_pre - y) ** 2)
        h_diff = np.abs(z_pre - z)
        v_h = d / t_n
        v_v = h_diff / t_n

        if v_h == 0 and v_v == 0:  # 悬停状态
            Energy_uav = t_n * P0
        else:
            if is_ascending is None:  # 水平飞行
                Energy_uav = t_n * ((4 * PB * (1 + (3 * v_h ** 2) / (omega ** 2 * R ** 2))) +
                                    4 * P1 * (np.sqrt(1 + pow(v_h, 4) / (4 * pow(v_o, 4))) - pow(v_h, 2) / (2 * pow(v_o, 2))) +
                                    2 * d_o * rho * s * G * pow(v_h, 3) +
                                    P0 + (W * v_v) / 2 - SF * pow(v_v, 3) +
                                    (W / 2 - SF * rho * pow(v_v, 2)) * (np.sqrt((1 - SF / G) * pow(v_v, 2)) + W / (2 * rho * G)))
            elif is_ascending:  # 上升
                Energy_uav = t_n * (P0 + (W * v_v) / 2 + SF * rho * pow(v_v, 3) +
                                    (W / 2 + SF * rho * pow(v_v, 2)) * (np.sqrt((1 + SF / G) * pow(v_v, 2)) + W / (2 * rho * G)))
            else:  # 下降
                Energy_uav = t_n * (P0 + (W * v_v) / 2 - SF * rho * pow(v_v, 3) +
                                    (W / 2 - SF * rho * pow(v_v, 2)) * (np.sqrt((1 - SF / G) * pow(v_v, 2)) + W / (2 * rho * G)))

        return Energy_uav




    def flight_energy(self,UAV_trajectory,UAV_flight_time,EP,slot):
            d_o = 0.6  # 阻力比
            SF = 0.5   #机身等效平板面积
            rho = 1.225  # 空气密度，单位kg/m³
            s = 0.05  # 旋翼实度
            G = 0.503  # 旋翼盘面积，单位平方米
            U_tip = 120  # 旋翼叶片尖端速度，单位米/秒
            v_o = 4.3  # 悬停时平均旋翼诱导速度，单位米/秒
            omega = 300  # 旋翼叶片角速度，单位弧度/秒
            R = 0.4  # 旋翼半径，单位米
            delta = 0.012  # 阻力系数
            k = 0.1  # 诱导功率增量修正系数
            W = 20  #总重
            P0 = 4*((delta / 8) * rho * s * G * (pow(omega, 3)) * (pow(R, 3)) + (1 + k) * (pow( (W/4), (3 / 2)) / np.sqrt(2 * rho * G)))
            P1 = (1 + k) * (pow((W/4), (3 / 2)) / np.sqrt(2 * rho * G))
            PB = (delta / 8) * rho * s * G * (pow(omega, 3)) * (pow(R, 3))
            Energy_uav = np.zeros((EP, self.N_slot), dtype=float)
            P2 =11.46
            count =0
            for ep in range(self.eps-EP,self.eps):
                horizontal = UAV_trajectory[ep,:, [0, 1]]
                vertical = UAV_trajectory[ep,:, -1]
                t_n=UAV_flight_time[ep,:]
                t_n= t_n/10

                for i in range(slot[0,ep]):
                    if (i==0):
                        d = np.sqrt((horizontal[0,i] - self.l_o_h[0])**2 + (horizontal[1,i] - self.l_o_h[1])**2)
                        h = np.abs(vertical[i]-vertical[0])
                    else:
                        d = np.sqrt((horizontal[0,i] - horizontal[0,i-1])**2 + (horizontal[1,i] - horizontal[1,i-1])**2)
                        h = np.abs(vertical[i] - vertical[i - 1])

                    v_h = d/t_n[i]
                    v_v = h/t_n[i]
                    Energy_uav[count, i] = t_n[i] * ((4 * PB *(1 + (3 * v_h ** 2)/(omega ** 2 * R ** 2))) \
                        + 4 * P1 * (np.sqrt(1 + pow(v_h, 4)/(4 * pow(v_o, 4))) - pow(v_h, 2)/(2 * pow(v_o, 2))) \
                        + 2 * d_o * rho * s * G * pow(v_h, 3) \
                        + P0 + (W * v_v)/2 - SF * pow(v_v, 3) \
                        + (W/2 - SF * rho * pow(v_v, 2)) * (np.sqrt((1 - SF/G) * pow(v_v, 2)) + W/(2 * rho *G)))                
                                    
                count=count+1
                
            return Energy_uav


    def UAV_FLY(self, UAV_trajectory, Slot):
        for slot in range(Slot):
            UAV_trajectory[slot, 0] = UAV_trajectory[slot, 0]
            UAV_trajectory[slot, 1] = UAV_trajectory[slot, 1]
            UAV_trajectory[slot, 2] = UAV_trajectory[slot, 2]

        for slot in range(2, Slot):
            diff = np.abs(UAV_trajectory[slot, 0] - UAV_trajectory[slot - 2, 0]) + np.abs(UAV_trajectory[slot, 1] - UAV_trajectory[slot - 2, 1])
            if diff > 10:
                UAV_trajectory[slot - 1, 0] = (UAV_trajectory[slot - 2, 0] + UAV_trajectory[slot, 0]) / 2
                UAV_trajectory[slot - 1, 1] = (UAV_trajectory[slot - 2, 1] + UAV_trajectory[slot, 1]) / 2
        return UAV_trajectory


   