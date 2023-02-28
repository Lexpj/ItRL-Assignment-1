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

class EgreedyPolicy:

    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        self.Q = np.zeros(n_actions)
        self.n = np.zeros(n_actions)
        
    def select_action(self, epsilon):
        e_gredy = np.zeros(self.n_actions)
        e_gredy.fill(epsilon / (self.n_actions - 1))
        e_gredy[np.argmax(self.Q)] = 1 - epsilon
        
        return np.random.choice(self.n_actions, p = e_gredy)
        
        
    def update(self,a,r):
        self.n[a] += 1
        self.Q[a] += (1 / self.n[a]) * (r - self.Q[a])

class OIPolicy:

    def __init__(self, n_actions=10, initial_value=0.0, learning_rate=0.1):
        self.n_actions = n_actions
        self.Q = np.zeros(n_actions)
        self.Q.fill(initial_value) 
        self.lr = learning_rate
        
    def select_action(self):
        return np.argmax(self.Q)
        
    def update(self,a,r):
        self.Q[a] +=  self.lr * (r - self.Q[a]) 

class UCBPolicy:

    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        self.Q = np.zeros(n_actions)
        self.n = np.zeros(n_actions)
    
    def select_action(self, c, t):
        
        UCB = [np.inf if self.n[a] == 0 else (self.Q[a] + c * np.sqrt(np.log(t) / self.n[a])) for a in range(self.n_actions)]  
        return np.argmax(UCB)
        
    def update(self,a,r):
        self.n[a] += 1
        self.Q[a] += (1 / self.n[a]) * (r - self.Q[a])
    
def test():
    n_actions = 10
    env = BanditEnvironment(n_actions=n_actions) # Initialize environment    
    
    pi = EgreedyPolicy(n_actions=n_actions) # Initialize policy
    a = pi.select_action(epsilon=0.5) # select action
    r = env.act(a) # sample reward
    pi.update(a,r) # update policy
    print("Test e-greedy policy with action {}, received reward {}".format(a,r))
    
    pi = OIPolicy(n_actions=n_actions,initial_value=1.0) # Initialize policy
    a = pi.select_action() # select action
    r = env.act(a) # sample reward
    pi.update(a,r) # update policy
    print("Test greedy optimistic initialization policy with action {}, received reward {}".format(a,r))
    
    pi = UCBPolicy(n_actions=n_actions) # Initialize policy
    a = pi.select_action(c=1.0,t=1) # select action
    r = env.act(a) # sample reward
    pi.update(a,r) # update policy
    print("Test UCB policy with action {}, received reward {}".format(a,r))
    
if __name__ == '__main__':
    test()
