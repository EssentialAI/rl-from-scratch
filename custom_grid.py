# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 13:39:47 2023

@author: naresh
"""

# Working code below

import cv2
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from IPython.display import clear_output
from collections import defaultdict
import gymnasium as gym

class CustomGrid(object):
    def __init__(self, env_type = "Deterministic"):
        self.start = 0
        self.goal= 15
        self.act_dict = {0:"left", 1:"down", 2:"right", 3:"up", 4:"stay"}
        self.env_type = env_type
        # self.tot_rew = 0

        self.observation_space = gym.spaces.Discrete(16)
        self.n_states = 16
        self.n_actions = 5
        self.environment_height = 4
        self.environment_width = 4
  
        self.rew_dist = np.array([0, 0, 0, -0.3, 0, 0, 0, 0, 0, -0.3, 0, 0, 0, 0, 0, 2])
        self.holes = [3, 9]
        self.agent_pos = 0
        self.action_space = [0,1,2,3,4]
        self.gamma = 0.9

        # self.agent_pos = np.asarray([0,0])
        self.gold_pos = np.asarray([3, 3])
        self.gold_quantity = 1
        self.pit_pos = np.asarray([[3, 0], [1,2]])
        self.run_pos = np.asarray([[0,1], [2, 0], [4, 0], [3,1]])
        self.wumpus_pos = np.asarray([2, 4])
        
    def reset(self):
        self.agent_pos = 0
        # self.value = np.zeros((3,3))
        return self.agent_pos
        
    def step(self, p_state, action):

        d_b = [0,1,2,3]
        u_b = [12,13,14,15]
        l_b = [0, 4, 8, 12]
        r_b = [3, 7, 11, 15]
      
        self.done = False
        # print(p_state)
        self.agent_pos = p_state

        if action == 1:
          if p_state in d_b:
            self.agent_pos = self.agent_pos
          
          else:
            self.agent_pos -= 4 # down
    
        if action == 3:
          if p_state in u_b:
            self.agent_pos = self.agent_pos
          else:
            self.agent_pos += 4 # up
            
        if action == 2:
          if p_state in r_b:
            self.agent_pos = self.agent_pos

          else:
            self.agent_pos += 1 # right

        if action == 0:
          if p_state in l_b:
            self.agent_pos = self.agent_pos
          else:
            self.agent_pos -= 1 # left
            
        if action ==4:
            self.agent_pos = self.agent_pos
          
        self.agent_pos = np.clip(self.agent_pos, 0, 15)
        
        if action == 4: # if stay
            rew = 0 + self.rew_dist[self.agent_pos]
            
        else:
            
            if (self.agent_pos==p_state).all():
                rew = -0.3 + self.rew_dist[self.agent_pos] # negative reward for hitting boundaries
            else:
                rew = self.rew_dist[self.agent_pos]
        
        if self.env_type == "Stochastic":
            self.agent_pos = np.random.randint(env.n_states)
                          
        if (self.agent_pos == self.goal):
            # print("Agent pos:", self.agent_pos)
            # print("Won")
            self.done = True
            
        elif self.decode(self.agent_pos) in [list(i) for i in self.pit_pos]:
            # print("Agent pos:", self.agent_pos)
            # print("Dead")
            self.done = True
            
        return self.agent_pos, float(rew), self.done

    def decode(self, state):
        l = []   
        for i in range(4):
            for j in range(4):
                l.append([j,i])

        p = {}        
        for i in range(16):
            p[i] = l[i]

        return p[state]
          

    def render(self, mode='human', plot=False):
        """This method renders the environment.

        :param str mode: 'human' renders to the current display or terminal and returns nothing.
        :param bool plot: Boolean indicating whether we show a plot or not. If False, the method returns a resized NumPy
                     array representation of the environment to be used as the state. If True it plots the environment.

        :returns arr preprocessed_image: Grayscale NumPy array representation of the environment."""

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)

        def plot_image(plot_pos):
            """This is a helper function to render the environment. It checks which objects are in a particular
            position on the grid and renders the appropriate image.

            :param arr plot_pos: Co-ordinates of the grid position which needs to be rendered."""

            # Initially setting every object to not be plotted.
            plot_agent, plot_gold, plot_pit, plot_wumpus, plot_run = \
                False, False, False, False, False

            # Checking which objects need to be plotted by comparing their positions.
            if np.array_equal(self.decode(self.agent_pos), plot_pos) and any(np.array_equal(self.run_pos[i], plot_pos) for i in range(len(self.run_pos))):
                plot_run = True
            
            if np.array_equal(self.decode(self.agent_pos), plot_pos) and not any(np.array_equal(self.run_pos[i], plot_pos) for i in range(len(self.run_pos))):
                plot_agent = True
                
            if self.gold_quantity > 0:  # Gold isn't plotted if it has already been picked by one of the agents.
                if np.array_equal(plot_pos, self.gold_pos):
                    plot_gold = True
                    
            if any(np.array_equal(self.pit_pos[i], plot_pos) for i in range(len(self.pit_pos))):
                plot_pit = True
                
            if np.array_equal(plot_pos, self.wumpus_pos):
                plot_wumpus = True

            # Plot for Agent.
            if plot_agent and \
                    all(not item for item in
                        [plot_gold, plot_pit, plot_wumpus, plot_run]):
                agent = AnnotationBbox(OffsetImage(plt.imread('./images/custom_grid_images/walk.png'), zoom=0.2),
                                       np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(agent)
            
            if plot_run and \
                    all(not item for item in
                        [plot_gold, plot_pit, plot_wumpus]):
                agent = AnnotationBbox(OffsetImage(plt.imread('./images/custom_grid_images/worried.png'), zoom=0.15),
                                       np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(agent)

            # Plot for Gold.
            elif plot_gold and \
                    all(not item for item in
                        [plot_agent, plot_pit, plot_wumpus]):
                gold = AnnotationBbox(OffsetImage(plt.imread('./images/custom_grid_images/cheese.jpg'), zoom=0.06),
                                      np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(gold)

            # Plot for Pit.
            elif plot_pit and \
                    all(not item for item in
                        [plot_agent, plot_gold, plot_wumpus]):
                pit = AnnotationBbox(OffsetImage(plt.imread('./images/custom_grid_images/tom.png'), zoom=0.035),
                                     np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(pit)

            # Plot for Wumpus.
            elif plot_wumpus and \
                    all(not item for item in
                        [plot_agent, plot_gold, plot_pit]):
                wumpus = AnnotationBbox(OffsetImage(plt.imread('./images/custom_grid_images/hole.jpg'), zoom=0.2),
                                        np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(wumpus)

            # Plot for Agent and Breeze.
            elif all(item for item in [plot_agent, plot_wumpus]) and \
                    all(not item for item in
                        [plot_gold, plot_pit]):
                agent_breeze = AnnotationBbox(OffsetImage(plt.imread('./images/custom_grid_images/dead.png'), zoom=0.48),
                                              np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(agent_breeze)

            # Plot for Agent and Pit.
            elif all(item for item in [plot_agent, plot_pit]) and \
                    all(not item for item in
                        [plot_gold, plot_wumpus]):
                agent_pit = AnnotationBbox(OffsetImage(plt.imread('./images/custom_grid_images/run.jpg'), zoom=0.08),
                                           np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(agent_pit)
            
            # Plot for Agent and Pit.
            elif all(item for item in [plot_agent, plot_gold]) and \
                    all(not item for item in
                        [plot_pit, plot_wumpus]):
                agent_pit = AnnotationBbox(OffsetImage(plt.imread('./images/custom_grid_images/small_rew.png'), zoom=0.2),
                                           np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(agent_pit)

            

        coordinates_state_mapping_2 = {}
        for j in range(self.environment_height * self.environment_width):
            coordinates_state_mapping_2[j] = np.asarray(
                [j % self.environment_width, int(np.floor(j / self.environment_width))])

        # Rendering the images for all states.
        for position in coordinates_state_mapping_2:
            plot_image(coordinates_state_mapping_2[position])

        plt.xticks([0, 1, 2, 3])
        plt.yticks([0, 1, 2, 3])
        plt.grid()

        if plot:  # Displaying the plot.
            # plt.cla()
            plt.show()
        else:  # Returning the preprocessed image representation of the environment.
            fig.canvas.draw()
            img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
            width = 4
            height = 4
            dim = (width, height)
            # noinspection PyUnresolvedReferences
            preprocessed_image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            plt.show()
            return preprocessed_image

env = CustomGrid()
env.reset()

def create_random_episode(plot= False):
    
  state = np.random.randint(16)
  
  done = False
  episode = []
  if done:
    print("State is :{}, Action was {}".format(state, env.act_dict[action]))
  while not done:
      action = np.random.randint(4)
      n, r, done = env.step(state, action)
      print("State->{}, Action->{}, N->{}, done-> {}".format(state, env.act_dict[action], n, done))
      episode.append((state, env.act_dict[action], n, done))
      if plot:
          clear_output(wait=True)
          env.render(plot=True)
      state = n
      
# create_random_episode()