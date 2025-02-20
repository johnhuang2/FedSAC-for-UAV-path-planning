"""
Federated Soft Actor-Critic (FedSAC) Implementation for RIS-assisted UAV Communication Systems

This module implements a federated learning framework using the Soft Actor-Critic (SAC) algorithm
for optimizing RIS-assisted UAV communication systems. It supports both traditional federated
averaging (FedAvg) and best model selection strategies.

Key Features:
- Multi-agent federated learning with SAC
- Model aggregation using FedAvg or best model selection
- Performance tracking and logging
- Distributed training support
"""

import torch
from RL_BRAIN_SAC import SAC
import numpy as np
from RIS_UAV_MAIN_SAC import main_SAC
from multiprocessing import Process
import subprocess
import time
import copy
import csv
import os
import argparse

def train_model(model_id):
    """
    Train a local model with the specified ID.
    
    Args:
        model_id (int): Identifier for the local model/agent
    """
    subprocess.run(['python', 'RIS_UAV_MAIN_SAC.py', str(model_id), 'True'])

def average_weights(w):
    """
    Compute the average of model weights.
    
    Args:
        w (list): List of model state dictionaries
        
    Returns:
        dict: Averaged model weights
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def fed_avg(weights_list):
    """
    Implement traditional federated averaging method.
    
    Args:
        weights_list (list): List of weights from all local models
        
    Returns:
        list: Averaged weights for actor and critic networks
    """
    if weights_list is None:
        print("Error: No valid weights list provided")
        return None

    actor_state_dicts = []
    critic1_state_dicts = []
    critic2_state_dicts = []
    
    for weights in weights_list:
        actor_state_dicts.append(weights[0])
        critic1_state_dicts.append(weights[1])
        critic2_state_dicts.append(weights[2])

    avg_actor = average_weights(actor_state_dicts)
    avg_critic1 = average_weights(critic1_state_dicts)
    avg_critic2 = average_weights(critic2_state_dicts)

    return [avg_actor, avg_critic1, avg_critic2]

def get_best_model(weights_list, M, round_num):
    """
    Select the best performing model based on average reward.
    
    Args:
        weights_list (list): List of weights from all local models
        M (int): Number of agents/models
        round_num (int): Current federation round number
        
    Returns:
        list: Weights of the best performing model
    """
    if weights_list is None:
        print("Error: No valid weights list provided")
        return None
        
    best_reward = float('-inf')
    best_model_idx = 0
    
    print(f"\nEvaluating models for round {round_num}:")
    
    # Record rewards for all processes in global_reward.csv
    for i in range(M):
        try:
            with open(f'sac_train_reward_{i}.csv', 'r') as f:
                reader = csv.reader(f)
                rewards = list(reader)
                # Record all rewards
                for row in rewards:
                    with open('global_reward.csv', 'a+', newline='') as global_f:
                        writer = csv.writer(global_f)
                        writer.writerow([int(row[1]) + round_num * 20, float(row[2])])
                
                # Calculate average reward for best model selection
                if len(rewards) >= 20:
                    recent_rewards = [float(row[2]) for row in rewards[-20:]]
                    avg_reward = sum(recent_rewards) / 20
                    print(f"Agent {i} average reward over last 20 episodes: {avg_reward}")
                    
                    if avg_reward > best_reward:
                        best_reward = avg_reward
                        best_model_idx = i
        except Exception as e:
            print(f"Error processing rewards for agent {i}: {e}")
            continue
    
    print(f"\nSelected agent {best_model_idx} as best model with average reward {best_reward}")
    
    # Cleanup temporary files
    for i in range(M):
        try:
            os.remove(f'sac_train_reward_{i}.csv')
        except:
            pass
            
    return weights_list[best_model_idx]

def save_local_models(M):
    """
    Load and save weights from all local models.
    
    Args:
        M (int): Number of agents/models
        
    Returns:
        list: List of weights from all local models
    """
    print("Loading local models...")
    weights_list = []
    for i in range(M):
        try:
            weight1 = torch.load(f'local_model_{i}_actor_weight2.pth')
            weight2 = torch.load(f'local_model_{i}_critic1_weight2.pth')
            weight3 = torch.load(f'local_model_{i}_critic2_weight2.pth')
            GWeights_np = [weight1, weight2, weight3]
            weights_list.append(GWeights_np)
            print(f"Successfully loaded weights for agent {i}")
        except Exception as e:
            print(f"Error loading weights for agent {i}: {e}")
            return None
    return weights_list

def start_local_processes(M):
    """
    Start training processes for all local models.
    
    Args:
        M (int): Number of agents/models to train
    """
    processes = []
    for i in range(M):
        p = Process(target=train_model, args=(i,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def fedrateLearning(FedNum, FedRound, start_time, strategy='best_model'):
    """
    Main federated learning loop implementation.
    
    Args:
        FedNum (int): Number of federated learning agents
        FedRound (int): Number of federation rounds
        start_time (float): Start time of training
        strategy (str): Federation strategy ('fed_avg' or 'best_model')
    """
    print(f"\nInitializing federated learning with strategy: {strategy}")
    
    # Initialize/clear global reward tracking file
    with open('global_reward.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Reward'])
    
    # Initialize global model
    print("Training initial model...")
    main_SAC(0, False, 20)
        
    for i in range(FedRound):
        print(f"\nStarting Fed Learning Round: {i}")
        
        # Start all local training processes
        start_local_processes(FedNum)
        
        # Get weights from all local models
        weights_list = save_local_models(FedNum)
        if weights_list is None:
            print(f"Error in round {i}: Could not load local models")
            continue
            
        # Select global model based on strategy
        if strategy == 'fed_avg':
            print("Using traditional federated averaging strategy")
            GlobalWeight = fed_avg(weights_list)
        else:
            print("Using best model selection strategy")
            GlobalWeight = get_best_model(weights_list, FedNum, i)
            
        if GlobalWeight is None:
            print(f"Error in round {i}: Could not update global model")
            continue

        # Save global model weights for next round
        print(f"\nSaving global model for round {i}...")
        torch.save(GlobalWeight[0], 'global_model_actor_weight2.pth')
        torch.save(GlobalWeight[1], 'global_model_critic1_weight2.pth')
        torch.save(GlobalWeight[2], 'global_model_critic2_weight2.pth')

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time for Round {i}: {elapsed_time:.4f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Federated Learning for SAC')
    parser.add_argument('--strategy', type=str, choices=['fed_avg', 'best_model'], 
                       default='best_model', help='Strategy for federal learning')
    parser.add_argument('--fed_num', type=int, default=5, help='Number of federal learning agents')
    parser.add_argument('--fed_round', type=int, default=10, help='Number of federal learning rounds')
    parser.add_argument('--episode', type=int, default=20, help='Number of episodes per round')
    
    args = parser.parse_args()
    
    print(f"Start FED_SAC with strategy: {args.strategy}")
    start_time = time.time()
    
    if os.path.exists('global_reward.csv'):
        os.remove('global_reward.csv')
        
    fedrateLearning(args.fed_num, args.fed_round, start_time, args.strategy)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total training time: {elapsed_time:.4f} seconds")
    
    with open('fedsacTime.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([args.fed_num * args.fed_round * args.episode, elapsed_time])

# Usage examples:
# 1. Run with best model selection strategy (default):
#    python fedsac.py --strategy best_model

# 2. Run with traditional federated averaging:
#    python fedsac.py --strategy fed_avg

# 3. Run with custom parameters:
#    python fedsac.py --strategy fed_avg --fed_num 8 --fed_round 15 --episode 30