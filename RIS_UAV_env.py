"""
RIS-assisted UAV Communication System Environment

This module implements a simulation environment for a RIS (Reconfigurable Intelligent Surface)
assisted UAV (Unmanned Aerial Vehicle) communication system. The environment models the
3D trajectory optimization problem considering both communication quality and energy efficiency.

Environment Parameters:
- Area of Interest (AoI): 500m x 500m
- UAV Height Range: 0m - 200m
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

# Environment Constants
UNIT = 1              # Grid unit size (pixels)
IOT_H = 500          # Grid height (m)
IOT_W = 500          # Grid width (m)
Max_Hight = 200      # Maximum UAV height (m)
Min_Hight = 0        # Minimum UAV height (m)

# UAV Movement Parameters
D_k = 1024           # Data size for transmission
t_min = 1            # Minimum time slot duration
t_max = 3            # Maximum time slot duration

# Wireless Environment Parameters
B = 2000             # Overall bandwidth (MHz)
N_0 = mt.pow(10, ((-169 / 3) / 10))  # Noise power spectral density
Xi = mt.pow(10, (3/10))  # Path loss at reference distance (D0 = 1m, 3dB)
a = 9.61             # Environment-dependent parameter
b = 0.16             # Environment-dependent parameter

# Channel Model Parameters
eta_los = 0.01       # Loss for Line-of-Sight (LoS) connections
eta_nlos = 0.2       # Loss for Non-Line-of-Sight (NLoS) connections
A = eta_los - eta_nlos  # Loss difference parameter
C = 0.2 * np.log10(4 * np.pi * 9 / 3) + eta_nlos  # Path loss parameter (carrier frequency: 900MHz)
Power = 0.5 * mt.pow(10, 3)  # Maximum uplink transmission power (500mW)

# RIS Configuration Parameters
W_R = [250,250]      # RIS horizontal location [x,y] (m)
Z_R = 100            # RIS height (m)
M = 100              # Number of RIS phase elements
data = 10            # Base data rate

class RIS_UAV(object):
    def __init__(self):
        super(RIS_UAV, self).__init__()
        # Episode Parameters
        self.N_slot = 600      # Number of time slots per episode
        self.GTs = 20          # Number of ground terminals
        self.l_o_v = 200       # Initial vertical location (m)
        self.l_f_v = 0         # Final vertical location (m)
        self.l_o_h = [0, 0]    # Initial horizontal location [x,y] (m)
        self.l_f_h = [500, 500]  # Final horizontal location [x,y] (m)
        self.eps = 60          # Number of episodes
        
        # Channel Parameters
        self.beta_0 = 1        # Channel power gain at reference distance
        self.c = 3e8           # Speed of light (m/s)
        self.f_c = 2e9         # Carrier frequency (Hz)
        self.Xi = 1            # Path loss constant
        self.B = 2000          # Bandwidth (MHz)
        self.N_0 = 1e-9        # Noise power spectral density
        self.M_r = 10          # Number of reflecting elements in row
        self.M_c = 10          # Number of reflecting elements in column
        self.alpha_k_RG = 2    # Path loss exponent for RIS-to-user link
        self.kappa_k_RG = 10   # Rician factor for RIS-to-user link
        self.alpha_k_UG = 2    # Path loss exponent for UAV-to-user link
        self.kappa_k_UG = 10   # Rician factor for UAV-to-user link
        
        # System Parameters
        self.Z_R = 50          # RIS height (m)
        self.W_R = [250, 250]  # RIS location coordinates [x,y] (m)
        self.Power = 1         # Transmission power
        self.A = 1             # Path loss factor
        self.C = 1             # Path loss factor
        self.delta_t = 1       # Time slot duration
        self.p_TX = 1          # UAV transmit power
        self.sigma = np.sqrt(3.98e-6)  # Noise power
        self.N_f = 64          # Number of sub-carriers
        self.P = 0.3           # Power allocation factor
        
        # State tracking parameters
        self.cumulative_d_s = np.zeros(self.GTs, dtype=float)  # Cumulative data per GT
        self.energy = None          # Energy consumption
        self.cumulative_energy = 0.0
        self.previous_distance_to_target = None
        self.previous_distance_to_charging = None
        self.finish = False

        # Action space definition
        self.action_space_uav_horizontal = ['n', 's', 'e', 'w', 'h']  # North, South, East, West, Hover
        self.action_space_uav_vertical = ['a', 'd', 's']  # Ascend, Descend, Stay

        # Action and state space dimensions
        self.n_actions = len(self.action_space_uav_horizontal) * len(self.action_space_uav_vertical) * \
                        self.GTs * (int(t_max/0.1)-int(t_min/0.1)+1)
        self.n_features = 4    # State space dimension: [x, y, z, distance_to_target]

        # Generate action table
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
        """Initialize environment with ground terminals and their parameters"""
        self.d_s = np.zeros((self.N_slot, self.GTs), dtype=float)
        self.energy = np.zeros(self.N_slot, dtype=float)
        self.w_k = np.zeros((self.GTs, 2), dtype=float)
        self.u_k = np.zeros((self.GTs, 1), dtype=float)

        # Define ground terminal coordinates [x,y] in meters
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
                                [100,450]])
        
        if len(gt_coordinates) != self.GTs:
            raise ValueError("Mismatch in the number of GTs and predefined coordinates.")

        self.w_k = gt_coordinates

        # Initialize data requirements for each GT
        for g in range(self.GTs):
            self.u_k[g, 0] = D_k / 2 + (D_k / 2) * rd.random()

    def reset(self):
        """Reset environment to initial state"""
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
        """
        Execute one time step in the environment
        
        Args:
            action: Array containing [index, horizontal_move, vertical_move, GT_index, time_slot]
            slot: Current time slot
            
        Returns:
            tuple: (next_state, reward)
        """
        if slot >= self.N_slot:
            self.finish = True
            return self.reset(), 0

        h = action[1]  # Horizontal movement
        v = action[2]  # Vertical movement
        c_n = action[3]  # Selected ground terminal
        t_n = action[4]  # Time slot duration

        pre_l_n = self.l_n.copy()
        pre_h_n = self.h_n

        # Update UAV height
        self.OtPoI = 0
        if v == 0:  # Ascending
            self.h_n = self.h_n + 2
            if self.h_n > Max_Hight:
                self.h_n = self.h_n - 2
                self.OtPoI = 1
        elif v == 1:  # Descending
            self.h_n = self.h_n - 2
            if self.h_n < Min_Hight:
                self.h_n = self.h_n + 2               
                self.OtPoI = 1
        elif v == 2:  # Stay at current height
            self.h_n = self.h_n

        # Update UAV horizontal position
        if h == 0:  # Move North
            self.l_n[1] = self.l_n[1] + 6
            if self.l_n[1] > IOT_H:
                self.l_n[1] = self.l_n[1] - 10
                self.OtPoI = 1
        elif h == 1:  # Move South
            self.l_n[1] = self.l_n[1] - 6
            if self.l_n[1] < 0:
                self.l_n[1] = self.l_n[1] + 10
                self.OtPoI = 1
        elif h == 2:  # Move East
            self.l_n[0] = self.l_n[0] + 6
            if self.l_n[0] > IOT_W:
                self.l_n[0] = self.l_n[0] - 10
                self.OtPoI = 1
        elif h == 3:  # Move West
            self.l_n[0] = self.l_n[0] - 6
            if self.l_n[0] < 0:
                self.l_n[0] = self.l_n[0] + 10
                self.OtPoI = 1
        elif h == 4:  # Hover
            self.l_n[0] = self.l_n[0]
            self.l_n[1] = self.l_n[1]

        # Calculate energy consumption for current time slot
        is_ascending = v == 0
        energy_slot = self.flight_energy_slot(pre_l_n, self.l_n, pre_h_n, self.h_n, 2, is_ascending)
        self.energy[slot] = energy_slot

        # Calculate communication throughput
        current_throughput = 0
        for g in range(self.GTs):
            c_ki = 1 if g == c_n else 0
            if c_ki == 1:
                link_rate = self.link_rate(g, c_ki)
                current_throughput += link_rate * (t_n / 10)

        # Calculate energy efficiency reward
        ee_reward = 0
        if energy_slot > 0:
            ee_reward = (current_throughput / energy_slot)

        # Calculate distance-based reward
        uav_position = np.array([self.l_n[0], self.l_n[1], self.h_n])
        target_position = np.array([self.l_f_h[0], self.l_f_h[1], self.l_f_v])
        current_distance = np.linalg.norm(uav_position - target_position)

        if self.previous_distance_to_charging is None:
            self.previous_distance_to_charging = current_distance

        distance_delta = self.previous_distance_to_charging - current_distance

        # Compute combined reward
        reward = distance_delta  # Distance improvement reward
        reward -= 1  # Time penalty
        reward -= current_distance * 0.01  # Distance penalty
        reward += ee_reward  # Energy efficiency reward

        # Check if target is reached
        if current_distance <= 30:
            reward += 1000  # Bonus for reaching target
            self.finish = True
        else:
            self.finish = False

        # Penalty for out of bounds movement
        if self.OtPoI == 1:
            reward -= 100

        self.previous_distance_to_charging = current_distance
        _state = np.array([self.l_n[0], self.l_n[1], self.h_n, current_distance])

        return _state, reward

    def link_rate(self, gt, c_ki):
        """
        Calculate the communication link rate between UAV and ground terminal
        
        Args:
            gt: Ground terminal index
            c_ki: Communication indicator (1 if communicating, 0 otherwise)
            
        Returns:
            float: Link rate in kbps
        """
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
        epsilon = 1e-6  # Constant to avoid zero values
        t_n = max(t_n, epsilon)  # Ensure t_n is not zero

        d_o = 0.6  # Drag ratio
        SF = 0.5   # Equivalent fuselage area
        rho = 1.225  # Air density, kg/m³
        s = 0.05  # Rotor solidity
        G = 0.503  # Rotor disk area, m²
        U_tip = 120  # Rotor blade tip speed, m/s
        v_o = 4.3  # Average induced velocity in hover, m/s
        omega = 300  # Rotor blade angular velocity, rad/s
        R = 0.4  # Rotor radius, m
        delta = 0.012  # Drag coefficient
        k = 0.1  # Induced power correction factor
        W = 20  # Total weight
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

        if v_h == 0 and v_v == 0:  # Hovering state
            Energy_uav = t_n * P0
        else:
            if is_ascending is None:  # Horizontal flight
                Energy_uav = t_n * ((4 * PB * (1 + (3 * v_h ** 2) / (omega ** 2 * R ** 2))) +
                                    4 * P1 * (np.sqrt(1 + pow(v_h, 4) / (4 * pow(v_o, 4))) - pow(v_h, 2) / (2 * pow(v_o, 2))) +
                                    2 * d_o * rho * s * G * pow(v_h, 3) +
                                    P0 + (W * v_v) / 2 - SF * pow(v_v, 3) +
                                    (W / 2 - SF * rho * pow(v_v, 2)) * (np.sqrt((1 - SF / G) * pow(v_v, 2)) + W / (2 * rho * G)))
            elif is_ascending:  # Ascending
                Energy_uav = t_n * (P0 + (W * v_v) / 2 + SF * rho * pow(v_v, 3) +
                                    (W / 2 + SF * rho * pow(v_v, 2)) * (np.sqrt((1 + SF / G) * pow(v_v, 2)) + W / (2 * rho * G)))
            else:  # Descending
                Energy_uav = t_n * (P0 + (W * v_v) / 2 - SF * rho * pow(v_v, 3) +
                                    (W / 2 - SF * rho * pow(v_v, 2)) * (np.sqrt((1 - SF / G) * pow(v_v, 2)) + W / (2 * rho * G)))

        return Energy_uav




    def flight_energy(self,UAV_trajectory,UAV_flight_time,EP,slot):
            d_o = 0.6  # Drag ratio
            SF = 0.5   # Equivalent fuselage area
            rho = 1.225  # Air density, kg/m³
            s = 0.05  # Rotor solidity
            G = 0.503  # Rotor disk area, m²
            U_tip = 120  # Rotor blade tip speed, m/s
            v_o = 4.3  # Average induced velocity in hover, m/s
            omega = 300  # Rotor blade angular velocity, rad/s
            R = 0.4  # Rotor radius, m
            delta = 0.012  # Drag coefficient
            k = 0.1  # Induced power correction factor
            W = 20  # Total weight
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



    def _build_ris_uav(self):
            self.d_s = np.zeros((self.N_slot, self.GTs), dtype=float)
            self.energy = np.zeros(self.N_slot, dtype=float)
            self.w_k = np.zeros((self.GTs, 2), dtype=float)
            self.u_k = np.zeros((self.GTs, 1), dtype=float)

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


   