import numpy as np
import random
from random import randint
import collections
from collections import Counter 
import matplotlib.pyplot as plt

class singlepath_explanation():
    # nS - number of States
    # Pl - transation matrix (S,A,S')
    # Rl - Reward matrix
    # Pol - Optimal Policy
    # V - Optimal Values for the Policy
    # actions - number of actions
    # gamma - gamma value (discount)
    def __init__(self,nS,Pl,Rl,actions,gamma,goallist,Pol=np.array([]),V=np.array([]),Q=np.array([])):
        self.nS = nS
        self.Pl = Pl
        self.Rl = Rl
        self.actions = actions
        self.gamma = gamma
        self.goallist=goallist
        if Pol.size==0:
            self.Q = np.zeros((self.nS,self.actions))
            self.V = np.zeros((self.nS))
            self.value_iteration()
        else:
            self.Pol = Pol
            self.V = V
            self.Q = Q

    def value_iteration(self):
        nQ = np.zeros((self.nS,self.actions))
        while True:
            self.V = np.max(self.Q,axis=1)
            for a in range(0,self.actions):
                nQ[:,a] = self.Rl[:,a] + self.gamma * np.dot(self.Pl[:,a,:],self.V)
            err = np.linalg.norm(self.Q-nQ)
            self.Q = np.copy(nQ)
            if err<1e-20:
                break
            
        #update policy
        self.V = np.max(self.Q,axis=1) 
        #correct for 2 equal actions
        self.Pol = np.argmax(self.Q, axis=1)
            
        return self.Pol

    # identify bifurcations in a path 
    # only identifies bifurcation that make it jump outside of the path 
    # it will return a list of bifurcation where each element coresponds to the bifurcation of the element in the path following the action in the pathactions
    #excluding the last element
    def path_bifurcations(self,path,pathactions):
        bifurcations = []
        for i in range(len(path)-1):
            pos = path[i]
            action = pathactions[i]
            bifurcations.append([])
            for state in range(self.nS):
                if self.Pl[pos,action,state] > 0 and state not in path:
                    bifurcations[i].append(state)
        return bifurcations

    # return the possible next state(s) to all the elements in the path, even if there is only one possible next state
    def all_bifurcations(self,path,pathactions):
        bifurcations = []
        for i in range(len(path)-1):
            pos = path[i]
            action = pathactions[i]
            bifurcations.append([])
            for state in range(self.nS):
                if self.Pl[pos,action,state] > 0:
                    bifurcations[i].append(state)
        return bifurcations

    # return the possible next state(s) to a single pair state action
    def state_bifurcations(self,state,action):
        bifurcations = []
        for nextstate in range(self.nS):
            if self.Pl[state,action,nextstate] > 0:
                bifurcations.append(nextstate)
        return bifurcations


    # run a path a single time (needs corrections based on possible jumps) +-
    # it can jump as long as it is inside the path (assuming it doesn't repeat states)
    # if it jumps outside the path, the path is incomplete and it fails
    def run_path(self,path,pathactions):
        pos = path[0]
        i = 1
        action = pathactions[i-1]
        steps = 0
        reward = self.Rl[pos,action] * self.gamma ** steps
        while i != len(path): 
            nextpos = path[i] 
            action = pathactions[i-1] 

            prob = random.random()
            while prob == 0:
                prob = random.random()
            addprob = 0

            for state in range(self.nS):
                if prob <= self.Pl[pos,action,state] + addprob:
                    if state == nextpos:
                        i = i + 1
                        pos = nextpos
                        break
                    elif state == pos:
                        break
                    else:
                        if state in path:
                            pos = state
                            i = path.index(pos) + 1
                        else:
                            return False, steps, reward # it failed to do the path
                    break
                addprob += self.Pl[pos,action,state]
 
            steps += 1
            reward += self.Rl[pos,action] * self.gamma ** steps
        return True, steps, reward

    #run the path n time to get the average of steps and reward and its distribution
    def run_n(self,path,pathactions,n=100000):
        totalsteps = 0
        totalreward = 0
        run_reward_dict = {}
        run_steps_dict = {}
        n_incomplete = 0
        for i in range(n):
            flag, steps,reward = self.run_path(path,pathactions)
            if flag:
                if steps in run_steps_dict:
                    run_steps_dict[steps] = run_steps_dict[steps] + 1
                else:
                    run_steps_dict[steps] = 1
                if reward in run_reward_dict:
                    run_reward_dict[reward] = run_reward_dict[reward] + 1
                else:
                    run_reward_dict[reward] = 1
                totalsteps += steps
                totalreward += reward
            else:
                n_incomplete += 1
        order_run_steps_dict = collections.OrderedDict(sorted(run_steps_dict.items()))
        order_run_reward_dict = collections.OrderedDict(sorted(run_reward_dict.items()))
        totalsteps = totalsteps / n
        totalreward = totalreward / n
        return order_run_steps_dict, order_run_reward_dict, totalsteps, totalreward, n_incomplete

    # returns true if it's goal, false otherwise
    def is_goal(self,state):
        if state in self.goallist:
            return True
        else:
            return False

    def run_from_point(self,Pol,start_pos):
        pos = start_pos
        action = Pol[pos]
        steps = 0
        reward = self.Rl[pos,action] * self.gamma ** steps
        while not self.is_goal(pos): 
            #generate next pos
            prob = random.random()
            while prob == 0:
                prob = random.random()
            addprob = 0

            for state in range(self.nS):
                if prob <= self.Pl[pos,action,state] + addprob:
                    pos = state
                    break
                addprob += self.Pl[pos,action,state]
            
            action = Pol[pos]
         
            steps += 1
            reward += self.Rl[pos,action] * self.gamma ** steps
        return steps, reward

    # generate the distribution of starting in a certain pos and following a certain policy
    def generate_single_state_distribution(self,Pol,start_pos,n=100000):  
        totalsteps = 0
        totalreward = 0
        run_reward_dict = {}
        run_steps_dict = {}
        for i in range(n):
            steps,reward = self.run_from_point(Pol,start_pos)
            if steps in run_steps_dict:
                run_steps_dict[steps] = run_steps_dict[steps] + 1
            else:
                run_steps_dict[steps] = 1
            if reward in run_reward_dict:
                run_reward_dict[reward] = run_reward_dict[reward] + 1
            else:
                run_reward_dict[reward] = 1
            totalsteps += steps
            totalreward += reward
        order_run_steps_dict = collections.OrderedDict(sorted(run_steps_dict.items()))
        order_run_reward_dict = collections.OrderedDict(sorted(run_reward_dict.items()))
        totalsteps = totalsteps / n
        totalreward = totalreward / n
        return order_run_steps_dict, order_run_reward_dict, totalsteps, totalreward

    def get_original_Pl(self,pos):
        return self.Pl[pos].copy()

    def set_Pl(self,pos,listPl):
        self.Pl[pos] = listPl.copy()
            

    def change_Pl(self,pos,action):
        Pl_state_zero = np.zeros(self.nS)
        newPl = []
        for a in range(self.actions):
            if a == action:
                newPl.append(self.Pl[pos,action])
            else:
                newPl.append(Pl_state_zero)
        self.set_Pl(pos,newPl)


    #proc_actions : actions to be processed it can be all or a certain action
    def state_explain(self,singlestate,change_Pl=False,proc_actions='all',n=100000):
        auxpol = self.Pol.copy()
        distribution_steps = {}
        distribution_reward = {}
        #save original Pl
        if change_Pl: #in the choosen state it can only do a certain action
            originalPl = self.get_original_Pl(singlestate)
        if proc_actions == 'all':
            for action in range(0,self.actions):
                distribution_steps[action] = {}
                distribution_reward[action] = {}
                if change_Pl:
                    flag = self.change_Pl(singlestate,action)
                    auxpol = self.value_iteration()
                    bifurcations = self.state_bifurcations(singlestate,action)
                    if len(bifurcations) == 0 or (len(bifurcations) == 1 and bifurcations[0] == singlestate):
                        self.set_Pl(singlestate,originalPl)
                        continue
                    start_pos = singlestate
                    run_steps_dict, run_reward_dict, totalsteps, totalreward = self.generate_single_state_distribution(auxpol,start_pos,n)
                    distribution_steps[action] = run_steps_dict
                    distribution_reward[action] = run_reward_dict
                    self.set_Pl(singlestate,originalPl)
                else:
                    bifurcations = self.state_bifurcations(singlestate,action)
                    this_run_steps_dict = {}
                    this_run_reward_dict = {}
                    for nextstate in bifurcations:
                        run_steps_dict, run_reward_dict, totalsteps, totalreward = self.generate_single_state_distribution(auxpol,nextstate,n)
                        prob = self.Pl[singlestate,action,nextstate]
                        for key in run_steps_dict:
                            run_steps_dict[key] = run_steps_dict[key] * prob
                            if key in this_run_steps_dict:
                                this_run_steps_dict[key] = this_run_steps_dict[key] + run_steps_dict[key]
                            else:
                                this_run_steps_dict[key] = run_steps_dict[key]
                        for key in run_reward_dict:
                            run_reward_dict[key] = run_reward_dict[key] * prob
                            if key in this_run_reward_dict:
                                this_run_reward_dict[key] = this_run_reward_dict[key] + run_reward_dict[key]
                            else:
                                this_run_reward_dict[key] = run_reward_dict[key]
                    order_run_steps_dict = collections.OrderedDict(sorted(this_run_steps_dict.items()))
                    order_run_reward_dict = collections.OrderedDict(sorted(this_run_reward_dict.items()))
                    distribution_steps[action] = order_run_steps_dict
                    distribution_reward[action] = order_run_reward_dict
        elif len(proc_actions) >= 0 and len(proc_actions) < self.actions:
            for action in proc_actions:
                distribution_steps[action] = {}
                distribution_reward[action] = {}
                if change_Pl:
                        flag = self.change_Pl(singlestate,action)
                        auxpol = self.value_iteration()
                        bifurcations = self.state_bifurcations(singlestate,action)
                        if len(bifurcations) == 0 or (len(bifurcations) == 1 and bifurcations[0] == singlestate):
                            self.set_Pl(singlestate,originalPl)
                            continue
                        start_pos = singlestate
                        run_steps_dict, run_reward_dict, totalsteps, totalreward = self.generate_single_state_distribution(auxpol,start_pos,n)
                        distribution_steps[action] = run_steps_dict
                        distribution_reward[action] = run_reward_dict
                        self.set_Pl(start_pos,originalPl)
                else:
                    bifurcations = self.state_bifurcations(singlestate,action)
                    this_run_steps_dict = {}
                    this_run_reward_dict = {}
                    for nextstate in bifurcations:
                        run_steps_dict, run_reward_dict, totalsteps, totalreward = self.generate_single_state_distribution(auxpol,nextstate,n)
                        prob = self.Pl[singlestate,action,nextstate]
                        for key in run_steps_dict:
                            run_steps_dict[key] = run_steps_dict[key] * prob
                            if key in this_run_steps_dict:
                                this_run_steps_dict[key] = this_run_steps_dict[key] + run_steps_dict[key]
                            else:
                                this_run_steps_dict[key] = run_steps_dict[key]
                        for key in run_reward_dict:
                            run_reward_dict[key] = run_reward_dict[key] * prob
                            if key in this_run_reward_dict:
                                this_run_reward_dict[key] = this_run_reward_dict[key] + run_reward_dict[key]
                            else:
                                this_run_reward_dict[key] = run_reward_dict[key]
                    order_run_steps_dict = collections.OrderedDict(sorted(this_run_steps_dict.items()))
                    order_run_reward_dict = collections.OrderedDict(sorted(this_run_reward_dict.items()))
                    distribution_steps[action] = order_run_steps_dict
                    distribution_reward[action] = order_run_reward_dict
        else:
            return
        self.value_iteration()
        for action in distribution_steps:
            for key in distribution_steps[action]:
                distribution_steps[action][key] = distribution_steps[action][key] / n
        for action in distribution_reward:
            for key in distribution_reward[action]:
                distribution_reward[action][key] = distribution_reward[action][key] / n
        return distribution_steps, distribution_reward

    #gets the optimal path(s) depending on certain conditions ending at the closest goal (following the policy),
    #it return a list where the first element is the position and the second is the list of possible states   
    #start_pos: the starting position of the path
    #remove_path_cicles: removes transitions to an already point on the path, if False it will appear as a possible transation but it will not be explored
    #remove_self_trans: removes transitions from a state to itself, if False it will appear as a possible transation but it will not be explored (can only be False if remove_oath_cicles is False)
    def get_optimal_paths(self,start_pos,remove_path_cicles=False,remove_self_trans=False,explored_path=[]):
        explored_path.append(start_pos)
        if self.is_goal(start_pos):
            return [start_pos]
        else:
            state_path = [start_pos, []]
            for action in range(self.actions):
                if self.Q[start_pos,action] == self.V[start_pos]:
                    for state in range(self.nS):
                        if self.Pl[start_pos,action,state] > 0:
                            #explored
                            if state in explored_path:
                                if not remove_path_cicles:
                                    if state == start_pos and not remove_self_trans:
                                        state_path[1].append([state])
                                    elif state != start_pos:
                                        state_path[1].append([state])
                            #not explored
                            else:
                                state_path[1].append(self.get_optimal_paths(state,remove_path_cicles,remove_self_trans,explored_path.copy()))
        return state_path

    # it receives a list of states (path) and a list of actions between each state (pathactions)
    def path_explain(self,path,pathactions,n=100000):
        run_steps_dict, run_reward_dict, totalsteps, totalreward, n_incomplete = self.run_n(path,pathactions,n)
        for key in run_steps_dict:
            run_steps_dict[key] = run_steps_dict[key] / n
        for key in run_reward_dict:
            run_reward_dict[key] = run_reward_dict[key] / n
        return run_steps_dict, run_reward_dict, n_incomplete

