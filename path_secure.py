import numpy as np
import random
from random import randint
import collections
from collections import Counter 
import matplotlib.pyplot as plt

class path_secure():
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
        self.list_actions = []
        for i in range(self.actions):
            self.list_actions.append(i)
        if Pol.size == 0:
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

    def is_goal(self,state):
        if state in self.goallist:
            return True
        else:
            return False

    #returns an action following an epsilon-greedy aproach where we acte greedly but take a random action epsilon percentage of the time
    def greedy_action(self,x,epsilon):
        prob = random.random()
        if prob > epsilon:
            actions = []
            maxQ = max(self.Q[x])
            i = 0
            for el in self.Q[x]:
                if el == maxQ:
                    actions.append(i)
                i+=1
            action = np.random.choice(actions)
        else:
            action = np.random.choice(np.array(self.list_actions))
        return action


    #checks if an action can return to the path, and it checks what is the best position for this action and what is the worst
    def action_returns_to_path(self,pos,action,path):
        nextstates=[]
        flag = False
        for state in range(self.nS):
            if self.Pl[pos,action,state] > 0:
                if state in path:
                    nextstates.append(state)
                    flag = True
        return flag, nextstates

    #it always tries to return to the path eve if the prosition is the beginning of the path
    def always_return_path(self,pos,path,path_explored,epsilon):
        pathaction = False
        notexploredactions = []
        exploredactions = []
        for action in range(self.actions):
            flag, nextstates = self.action_returns_to_path(pos,action,path)
            if flag:
                pathaction = True
                explored = True
                for state in nextstates:
                    if state not in path_explored:
                        explored = False
                if not explored:
                    notexploredactions.append(action)
                else:
                    exploredactions.append(action)
        
        prob = random.random()
        if prob > epsilon:
            if not pathaction:
                actions = []
                maxQ = max(self.Q[pos])
                i = 0
                for el in self.Q[pos]:
                    if el == maxQ:
                        actions.append(i)
                    i+=1
                action = np.random.choice(actions)
            else:
                if notexploredactions:
                    action = np.random.choice(notexploredactions)
                else:
                    action = np.random.choice(exploredactions)
        else:
            action = np.random.choice(np.array(self.list_actions))

        return action 

    #only tries to return if the position is the previous position or a new position further in the path
    def return_path_if_same_or_new(self,pos,path,path_explored,epsilon):
        pathaction = False
        notexploredactions = []
        for action in range(self.actions):
            flag, nextstates = self.action_returns_to_path(pos,action,path)
            if flag:
                pathaction = True
                explored = True
                for state in nextstates:
                    if state not in path_explored:
                        explored = False
                    elif path_explored and state == path_explored[-1]:
                        explored = False
                if not explored:
                    notexploredactions.append(action)

        prob = random.random()
        if prob > epsilon:
            if notexploredactions:
                action = np.random.choice(notexploredactions)
            else:
                actions = []
                maxQ = max(self.Q[pos])
                i = 0
                for el in self.Q[pos]:
                    if el == maxQ:
                        actions.append(i)
                    i+=1
                action = np.random.choice(actions)
        else:
            action = np.random.choice(np.array(self.list_actions))

        return action 


    #by default is uses the highest possible state
    #path needs to end at a goal state 
    def get_secure_value(self,path,pathactions,epsilon=0.3,always_return_path=False,return_path_if_same_or_new=True,Never_return_path=False):
        pos = path[0]
        i = 1
        steps = 0
        path_explored = []
        action = -1
        reward = 0
        while not self.is_goal(pos):

            if pos in path:
                prob = random.random()
                if prob >= epsilon:
                    action = pathactions[i-1]
                else:
                    action = self.greedy_action(pos,epsilon)
                if pos not in path_explored:
                    path_explored.append(pos)
            #the next 3 if are in case it gets out of the path
            elif always_return_path:
                action = self.always_return_path(pos,path,path_explored,epsilon)
            elif return_path_if_same_or_new:
                action = self.return_path_if_same_or_new(pos,path,path_explored,epsilon)
            #in this case if it fails the path it will do a greedy action
            elif Never_return_path:
                action = self.greedy_action(pos,epsilon)

            reward += self.Rl[pos,action] * self.gamma ** steps
            #nextpos = path[i] 

            prob = random.random()

            #make sure prob is not 0
            while prob == 0:
                prob = random.random()

            addprob = 0

            for state in range(self.nS):
                if prob <= self.Pl[pos,action,state] + addprob:
                    pos = state
                    if state in path:
                        i = path.index(pos) + 1
                    break
                addprob += self.Pl[pos,action,state]

 
            steps += 1

        reward += self.Rl[pos,action] * self.gamma ** steps
        return steps, reward
    
    def path_secure_distribution(self,path,pathactions,epsilon=0.3,always_return_path=False,return_path_if_same_or_new=True,Never_return_path=False,n=100000):
        totalsteps = 0
        totalreward = 0
        run_reward_dict = {}
        run_steps_dict = {}
        for i in range(n):
            steps,reward = self.get_secure_value(path,pathactions,epsilon,always_return_path,return_path_if_same_or_new,Never_return_path)
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
        for key in order_run_steps_dict:
            order_run_steps_dict[key] = order_run_steps_dict[key] / n
        for key in order_run_reward_dict:
            order_run_reward_dict[key] = order_run_reward_dict[key] / n
        return order_run_steps_dict, order_run_reward_dict




