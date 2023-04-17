# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 19:00:30 2023

@author: naresh
"""

import gym
from custom_grid import CustomGrid
import numpy as np
import matplotlib.pyplot as plt

class MonteAgent(object):
    def __init__(self, env, policy):
        self.q_table = {}
        self.returns = {}
        self.policy= policy
        
        for i in range(env.n_states):
                for a in env.action_space:
                    self.q_table[(i, a)] = 0
                    self.returns[(i, a)] = []
                
    def generate_episode(self, state, max_step = 10, ep = 0):
        
        env.agent_pos=state
        done = False
        episode = []
        step = 0
        
        while not done:
            
            if np.random.rand() <= ep:
                action = np.random.choice(self.policy[state])                
            else:
        
                action = np.random.choice(env.action_space)
                
            next_state, r, done = env.step(state, action)
            
            episode.append(((state, action), r))
            state = next_state
            
            if step>= max_step:
                break
            
            step+=1
            
        _ =env.reset()
            
                        
        return episode
    
    def train(self, num_episodes = 200):
        
        all_episodes_rew = []
        all_episodes = []
        
        for i in range(num_episodes):
            state = np.random.randint(env.n_states)
            
            episode = self.generate_episode(state)                
            all_episodes.append(episode)
            all_episodes_rew.append(np.sum([epi[-1] for epi in episode]))
            
            g = 0
            
            for t in range(len(episode)-1, -1, -1):
                g  = env.gamma*g + episode[t][-1]
                
                if episode[t][0] not in [epi[0] for epi in episode[:t]]:
                    self.returns[tuple(episode[t][0])].append(g)
                    self.q_table[tuple(episode[t][0])] = np.mean(self.returns[tuple(episode[t][0])])
                    
                    self.new = {}
                    for key, value in enumerate(self.q_table):
                        self.new[value[0]] = [self.q_table[(value[0], i)] for i in range(env.n_actions)]
            
                    max_q = np.max(self.new[episode[t][0][0]])
                    self.policy[episode[t][0][0]] = list(np.where(self.new[episode[t][0][0]]==max_q)[0])
                    
        return self.policy, all_episodes, all_episodes_rew
    
def create_random_episode(plot= False):
    epi_rew = 0
    state = np.random.randint(16)
    done = False
    episode = []
    # if done:
      # print("State is :{}, Action was {}".format(state, env.act_dict[action]))
    while not done:
        action = agent.policy[state][0]
        n, r, done = env.step(state, action)
        epi_rew+=r
        # print("State->{}, Action->{}, N->{}, done-> {}".format(state, env.act_dict[action], n, done))
        episode.append((state, env.act_dict[action], n, done))
        if plot:
            clear_output(wait=True)
            env.render(plot=True)
        state = n
    return epi_rew

if __name__ == "__main__":
    env = CustomGrid()
    # env = gym.make("FrozenLake-v1", is_slippery = False)
    
    policy = {}
    for i in range(env.n_states):
        policy[i] = np.random.randint(env.n_actions)
    
    agent = MonteAgent(env, policy)
    _, all_episodes, all_rew = agent.train(450)    

    plt.plot(all_rew)
    plt.show()
          
    total_rew = []
    print("###Testing the Learnt Monte Carlo Agent###")
    for i in range(1000):
        epi_rew = create_random_episode()
        total_rew.append(epi_rew)
          
    for i in agent.policy.keys():
        if type(agent.policy[i])==list:
            agent.policy[i] = [env.act_dict[j] for j in agent.policy[i]]
        else:
            agent.policy[i] = env.act_dict[agent.policy[i]]
            
    print(agent.policy)
    plt.plot(total_rew)
    plt.show()
        