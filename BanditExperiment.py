#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bandit environment
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
2021
By Thomas Moerland
"""
import numpy as np
from BanditEnvironment import BanditEnvironment
from BanditPolicies import EgreedyPolicy, OIPolicy, UCBPolicy
from Helper import LearningCurvePlot, ComparisonPlot, smooth
 
def run_repetitions(n_actions, n_timesteps, n_repetitions, policy_id, eps = 0, initial_value = 0, c_value = 0):
    # Runs the policy (policy_id) for (n_timesteps) with (n_repetitions) for a certain hyperparameter
    rewards = np.zeros((n_repetitions, n_timesteps))
    
    for rep in range(n_repetitions):
        # Initialize environment 
        env = BanditEnvironment(n_actions = n_actions)
        # Create policy 
        if policy_id == 0:
            policy = EgreedyPolicy(n_actions = n_actions) 
        if policy_id == 1:
            policy = OIPolicy(n_actions = n_actions, initial_value = initial_value)
        if policy_id == 2:
            policy = UCBPolicy(n_actions = n_actions)
        # Select action
        for t in range(n_timesteps):
            if policy_id == 0:
                a = policy.select_action(epsilon = eps) 
            if policy_id == 1:
                a = policy.select_action()
            if policy_id == 2:
                a = policy.select_action(c_value, t)
            # Sample reward
            r = env.act(a)
            # Update policy
            policy.update(a,r)
            # Store reward
            rewards[rep][t] += r 
    return rewards

def experiment(n_actions, n_timesteps, n_repetitions, smoothing_window):
  # --------------------- Assignment 1: Optimistic init --------------------- #
    epsilons = [0.01, 0.05, 0.1, 0.25]
    y_e_greedy = np.empty((len(epsilons), n_timesteps))
    
    for i in range(len(epsilons)):
        rewards = run_repetitions(n_actions, n_timesteps, n_repetitions, policy_id = 0, eps = epsilons[i])
        y_e_greedy[i] = rewards.mean(axis = 0)
    
    plot_e_greedy = LearningCurvePlot() 
    for i in range(len(epsilons)): 
        plot_e_greedy.add_curve(smooth(y = y_e_greedy[i], window = smoothing_window), label = 'e = ' + str(epsilons[i]))
    
    plot_e_greedy.save(name = 'e_greedy.png')
    
    # --------------------- Assignment 2: Optimistic init --------------------- #
    initial_values = [0.1, 0.5, 1.0, 2.0]
    y_OI = np.empty((len(initial_values), n_timesteps))
    
    for i in range(len(initial_values)):
        rewards = run_repetitions(n_actions, n_timesteps, n_repetitions, policy_id = 1, initial_value = initial_values[i])
        y_OI[i] = rewards.mean(axis = 0)
    
    plot_OI = LearningCurvePlot() 
    for i in range(len(initial_values)):
        plot_OI.add_curve(smooth(y = y_OI[i], window = smoothing_window), label = 'Ψ = ' + str(initial_values[i]))
    
    plot_OI.save(name = 'OI.png')
   
   # --------------------- Assignment 3: UCB --------------------- #
    c = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    y_UCB = np.empty((len(c), n_timesteps))
    
    for i in range(len(c)):
        rewards = run_repetitions(n_actions, n_timesteps, n_repetitions, policy_id = 2, c_value = c[i])
        y_UCB[i] = rewards.mean(axis = 0)
        
    plot_UCB = LearningCurvePlot() 
    for i in range(len(c)):
        plot_UCB.add_curve(smooth(y = y_UCB[i], window = smoothing_window), label = 'c = ' + str(c[i]))
    
    plot_UCB.save(name = 'UCB.png')
    
    # --------------------- Assignment 4 (a): Comparison --------------------- #
    plot_comparison = ComparisonPlot()
    
    avg_rewards_e_greedy = np.empty(len(epsilons))
    for i in range(len(epsilons)):
        avg_rewards_e_greedy[i] = y_e_greedy[i].mean()
    plot_comparison.add_curve(epsilons, avg_rewards_e_greedy, label = 'e-greedy')
    
    avg_rewards_OI = np.empty(len(initial_values))
    for i in range(len(initial_values)):
        avg_rewards_OI[i] = y_OI[i].mean()
    plot_comparison.add_curve(initial_values, avg_rewards_OI, label = 'OI')
   
    avg_rewards_UCB = np.empty(len(c))
    for i in range(len(c)):
        avg_rewards_UCB[i] = y_UCB[i].mean()
    plot_comparison.add_curve(c, avg_rewards_UCB, label = 'UCB')
    
    plot_comparison.save(name = 'Comparison')
    
    # --------------------- Assignment 4 (b): Comparison --------------------- #
    epsilon_optimal = 0.1
    initial_value_optimal = 0.5
    c_optimal = 0.25
    
    y_e_greedy_optimal = run_repetitions(n_actions, n_timesteps, n_repetitions, policy_id = 0, eps = epsilon_optimal).mean(axis = 0)
    y_OI_optimal = run_repetitions(n_actions, n_timesteps, n_repetitions, policy_id = 1, initial_value = initial_value_optimal).mean(axis = 0)
    y_UCB_optimal = run_repetitions(n_actions, n_timesteps, n_repetitions, policy_id = 2, initial_value = c_optimal).mean(axis = 0)
    
    plot_optimal = LearningCurvePlot() 
    plot_optimal.add_curve(smooth(y = y_e_greedy_optimal, window = smoothing_window), label = 'Optimal e-greedy (0.1)')
    plot_optimal.add_curve(smooth(y = y_OI_optimal, window = smoothing_window), label = 'Optimal OI (0.5)')
    plot_optimal.add_curve(smooth(y = y_UCB_optimal, window = smoothing_window), label = 'Optimal UCB (0.25)')
    
    plot_optimal.save(name = 'Optimal.png')

    pass

if __name__ == '__main__':
    # experiment settings
    n_actions = 10
    n_repetitions = 500
    n_timesteps = 1000
    smoothing_window = 31
    
    experiment(n_actions=n_actions,n_timesteps=n_timesteps,
               n_repetitions=n_repetitions,smoothing_window=smoothing_window)