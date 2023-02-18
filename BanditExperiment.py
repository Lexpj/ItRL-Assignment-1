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
 

def run_repetitions_egreedy():
    n_timesteps = 1000
    n_rep = 500
    smoothing_window = 31
    
    
    plot = LearningCurvePlot("Comparison of different epsilon values")
    
    for test in [0.01,0.05,0.1,0.25]:
        
        rewards = np.zeros((n_rep,n_timesteps))

        for episode in range(n_rep):
            env = BanditEnvironment(n_actions)
            pi = EgreedyPolicy(n_actions=n_actions) # Initialize policy

            for timestep in range(n_timesteps):
                a = pi.select_action(epsilon=test) # select action
                r = env.act(a) # sample reward
                pi.update(a,r) # update policy
                rewards[episode][timestep] = r
        
        x = np.arange(n_timesteps)
        y = np.mean(rewards,axis=0)
        plot.add_curve(smooth(y,window=smoothing_window),f"e={test}")
        std = np.std(y,axis=0)
        lowy, highy = smooth(y-std,smoothing_window), smooth(y+std,smoothing_window)
        plot.ax.fill_between(x, lowy, highy, alpha = 0.25)

    plot.save("epsilon")

def run_repetitions_ucb():
    n_timesteps = 1000
    n_rep = 500
    smoothing_window = 31
    
    
    plot = LearningCurvePlot("Comparison of different c values")
    
    for test in [0.01,0.05,0.1,0.25,0.5,1.0]:
        
        rewards = np.zeros((n_rep,n_timesteps))

        for episode in range(n_rep):
            env = BanditEnvironment(n_actions)
            pi = UCBPolicy(n_actions=n_actions) # Initialize policy

            for timestep in range(n_timesteps):
                a = pi.select_action(c=test,t=timestep) # select action
                r = env.act(a) # sample reward
                pi.update(a,r) # update policy
                rewards[episode][timestep] = r
        print("done with",test)
        x = np.arange(n_timesteps)
        y = np.mean(rewards,axis=0)
        plot.add_curve(smooth(y,window=smoothing_window),label=f"c={test}")
        std = np.std(y,axis=0)
        lowy, highy = smooth(y-std,smoothing_window), smooth(y+std,smoothing_window)
        plot.ax.fill_between(x, lowy, highy, alpha = 0.25)

    plot.save("UCB")
        
        
        
    
    



        

def experiment(n_actions, n_timesteps, n_repetitions, smoothing_window):
    #To Do: Write all your experiment code here
    
    # Assignment 1: e-greedy
    #run_repetitions_egreedy()
    # Assignment 2: Optimistic init
    
    # Assignment 3: UCB
    run_repetitions_ucb()
    # Assignment 4: Comparison
    
    pass

if __name__ == '__main__':
    # experiment settings
    n_actions = 10
    n_repetitions = 500
    n_timesteps = 1000
    smoothing_window = 31
    
    experiment(n_actions=n_actions,n_timesteps=n_timesteps,
               n_repetitions=n_repetitions,smoothing_window=smoothing_window)