import numpy as np
import random
from random import randint
import collections
from collections import Counter 
import matplotlib.pyplot as plt
from tabulate import tabulate

#self made files
from explanations import *

class mountaincar():
    def __init__(self,tot_bins,min_vel,max_vel,min_position,max_postion):

        self.gravity = 9.8
        self.mass = 0.2
        self.friction = 0.05
        self.delta_t = 0.1
        self.nS = tot_bins*tot_bins
        self.actions = 3
        self.goallist = []
        self.tot_bins = tot_bins

        self.Pl = np.zeros((self.nS,self.actions,self.nS))
        self.Rl = np.zeros((self.nS,self.actions))

        self.position_state_array = np.linspace(min_position, max_postion, num=tot_bins, endpoint=True)
        self.velocity_state_array = np.linspace(min_vel, max_vel, num=tot_bins, endpoint=True)
        policy_matrix = np.random.randint(low=0,high=3,size=(tot_bins,tot_bins)).astype(np.float32)

        #print(self.velocity_state_array)
        #print(self.position_state_array)
        #print(policy_matrix)

    def step(self, action, position_t,velocity_t):
        done = False
        reward = -0.01
        action_list = [-0.2, 0, +0.2]
        action_t = action_list[action]
        velocity_t1 = velocity_t + (-self.gravity * self.mass * np.cos(3*position_t) + (action_t/self.mass) - (self.friction*velocity_t)) * self.delta_t
        position_t1 = position_t + (velocity_t1 * self.delta_t)
        # Check the limit condition (car outside frame)
        if position_t1 < -1.2:
            position_t1 = -1.2
            velocity_t1 = 0
        # self.position_list.append(position_t1)
        # Reward when the car reaches the goal
        if position_t >= 0.5:
            reward = +5.0
            position_t1 = 0.5
            velocity_t1 = 0
            done = True
        # Return state_t1, reward
        return [position_t1, velocity_t1], reward, done

    def calculate_matrix(self):
        state = 0
        for pos in self.position_state_array:
            for vel in self.velocity_state_array:
                for action in range(self.actions):
                    obs, reward,done = self.step(action,pos,vel)
                    nextstate = (np.digitize(obs[0], self.position_state_array) - 1) * len(self.velocity_state_array)  + (np.digitize(obs[1], self.velocity_state_array) - 1)
                    self.Pl[state,action,nextstate] = 1
                    self.Rl[state,action] = reward 
                    if done:
                        if state not in self.goallist:
                            self.goallist.append(state)         
                state += 1

    def get_pos_vel(self,state):
        pos_i = state // self.tot_bins
        vel_i = state % self.tot_bins
        pos = self.position_state_array[pos_i]
        vel = self.velocity_state_array[vel_i]
        return pos, vel 

    def get_state(self,pos,vel):
        state = (np.digitize(pos, self.position_state_array) - 1) * len(self.velocity_state_array)  + (np.digitize(vel, self.velocity_state_array) - 1)
        return state




if __name__ == "__main__":
    mountcar = mountaincar(30,-1.5,+1.5,-1.2,+0.5)
    mountcar.calculate_matrix()
    expl = Explanations(mountcar.nS,mountcar.Pl,mountcar.Rl,3,0.90,mountcar.goallist)
    # for state in range(len(mountcar.Pl)):
    #     for action in range(len(mountcar.Pl[state])):
    #         for nextstate in range(len(mountcar.Pl[state,action])):
    #             if mountcar.Pl[state,action,nextstate] != 0 and state == 374:
    #                 print(str(state) + ' , ' +str(action) + ' -> ' + str(nextstate))
    # print(mountcar.goallist)
    # print(mountcar.step(2,-1.2,0))
    # print(expl.Q[29])

    #print(expl.Pol)
    #print(expl.Q[374])
    #print(mountcar.get_pos_vel(374))
    #print(mountcar.get_state(-0.5,0))

    # for i in range(len(expl.Pol)):
    #     if expl.Pol[i] != 0:
    #         print(i)

    #mountcar.test_traj(-1.2, -0.0862068965517242)

    #TESTS
    # expl.single_get_state_explanation_distribution(374,labels=['left','no opt','right'])
    # expl.single_get_state_explanation_distribution(374,change_Pl=False,labels=['left','no opt','right'])
    # pth = expl.get_optimal_path_from_state(374,only_policy_path=True)
    # print(pth
    # expl.c51_state_distribution(374,-1,1,labels=['left','no opt','right'])
    path = [374,342,311,280,249,219,189,159,130,102,104,106,108,140,172,204,266,328,389,449,509,568,627,686,715,744,773,802,831,860,890]
    for el in path:
        print(mountcar.get_pos_vel(el)[0])
    # pathactions = [0,0,0,0,0,0,0,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2]
    # expl.get_path_distribution(path,pathactions)
    # expl.get_path_security_distribution(path,pathactions,epsilon=0.1,always_return_path=False,return_path_if_same_or_new=True,Never_return_path=False)
