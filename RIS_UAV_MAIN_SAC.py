# -- coding:UTF-8 --
import torch
import time
import os
from RIS_UAV_env import RIS_UAV
from RL_BRAIN_SAC import SAC
import numpy as np
from RES_PLOT import Res_plot
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import csv

def smooth(data, window_size=51, poly_order=3):
    return savgol_filter(data, window_size, poly_order)

def record_reward(id, ep, reward, is_global=False):
    """记录奖励到CSV文件"""
    if is_global:
        with open('global_reward.csv', 'a+', newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerow([ep, reward])
    else:
        with open(f'sac_train_reward_{id}.csv', 'a+', newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerow([id, ep, reward])

def train(RL, id, Episodes, env, UAV_trajectory, GT_schedule, UAV_flight_time, slots):
    total = 0
    total_rewards = []

    for ep in range(Episodes):
        observation = env.reset()
        slot = 0
        total_reward = 0
        
        while not env.finish:
            if slot >= env.N_slot:
                break
                
            action = RL.choose_action(observation)
            action_actual = env.find_action(action)
            observation_, reward = env.step(action_actual, slot)
            
            RL.store_transition(observation, action, reward, observation_, env.finish)
            total_reward += reward
            
            observation = observation_
            slot += 1
            total += 1
            slots[0, ep] += 1

        RL.learn()
        
        # 记录reward到两个文件
        record_reward(id, ep, total_reward)
        record_reward(id, ep, total_reward, is_global=True)
        
        total_rewards.append(total_reward)
        print(f"Agent {id} - Finish episode {ep} with reward {total_reward}")

        UAV_trajectory[ep, :] = env.UAV_FLY(UAV_trajectory[ep, :], slots[0, ep])
    
    return UAV_trajectory, GT_schedule, UAV_flight_time, slots

def main_SAC(id, LOCAL, episodes):
    # 清除旧的reward记录
    if not LOCAL:
        try:
            os.remove(f'sac_train_reward_{id}.csv')
        except:
            pass

    start_time = time.time()
    env = RIS_UAV()
    MEMORY_SIZE = 100000
    env.eps = episodes

    print(f"Start RIS UAV MAIN SAC - Agent {id}, LOCAL={LOCAL}")
    
    UAV_trajectory = np.zeros((episodes, env.N_slot, 3), dtype=np.float64)
    GT_schedule = np.zeros((episodes, env.N_slot), dtype=np.float64)
    UAV_flight_time = np.zeros((episodes, env.N_slot), dtype=np.float64)
    slots = np.zeros((1, episodes), dtype=np.int64)

    sac = SAC(
        n_actions=env.n_actions,
        n_features=env.n_features,
        memory_size=MEMORY_SIZE,
        batch_size=256
    )

    if LOCAL:
        print(f"Agent {id} attempting to load global model weights...")
        try:
            weight1 = torch.load('global_model_actor_weight2.pth')
            weight2 = torch.load('global_model_critic1_weight2.pth')
            weight3 = torch.load('global_model_critic2_weight2.pth')
            GWeights_np = [weight1, weight2, weight3]
            sac.set_weights(GWeights_np)
            print(f"Agent {id} loaded global weights successfully")
        except Exception as e:
            print(f"Error loading global weights for Agent {id}: {e}")
            return

    UAV_trajectory, GT_schedule, UAV_flight_time, slots = train(
        sac, id, episodes, env, UAV_trajectory, GT_schedule, UAV_flight_time, slots
    )

    print(f"Agent {id} completed training, saving weights...")
    weights = sac.get_weights()
    if weights:
        if LOCAL:
            torch.save(weights[0], f'local_model_{id}_actor_weight2.pth')
            torch.save(weights[1], f'local_model_{id}_critic1_weight2.pth')
            torch.save(weights[2], f'local_model_{id}_critic2_weight2.pth')
            print(f"Agent {id} saved local weights successfully")
        else:
            torch.save(weights[0], 'global_model_actor_weight2.pth')
            torch.save(weights[1], 'global_model_critic1_weight2.pth')
            torch.save(weights[2], 'global_model_critic2_weight2.pth')
            print("Initial global weights saved successfully")
    else:
        print(f"Warning: Agent {id} weights are empty!")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Agent {id} spent: {elapsed_time:.4f} seconds")
    return elapsed_time

if __name__ == "__main__":
    import sys
    model_id = int(sys.argv[1])
    local_mode = sys.argv[2].lower() == 'true' if len(sys.argv) > 2 else False
    episodes = 1000
    elapsed_time = main_SAC(model_id, local_mode, episodes)
    
# python RIS_UAV_MAIN_SAC.py 0 false