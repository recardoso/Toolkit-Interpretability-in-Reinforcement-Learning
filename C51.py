from tkinter import *
import time
import numpy as np
import random
from random import randint
from PIL import Image,ImageTk
import pickle 
from tkinter import messagebox
import colorsys
import matplotlib.pyplot as plt
import collections
import time
import math

class C51():
    def __init__(self,nS,actions,gamma,vmax,vmin,Pl,Rl,atoms = 51):
        self.actions = actions

        # we are clipping the values between Vmax and Vmin
        self.Vmax = vmax
        self.Vmin = vmin
        self.Pl = Pl
        self.Rl = Rl
        self.gamma = gamma
        self.actions = actions
        # number of states
        self.nS = nS
        # list of actions
        self.list_actions = []
        for i in range(self.actions):
            self.list_actions.append(i)

        # number of atoms, 51 because it was the recommended on the paper
        self.atoms = atoms

        # Define delta z
        self.delta_z = float((self.Vmax - self.Vmin)) / (self.atoms - 1)

        # Define zis
        self.zis = []
        for i in range(self.atoms):
            zi = self.Vmin + i * self.delta_z
            self.zis.append(zi)

        # Initialize Q values
        self.Q = np.zeros((self.nS,self.actions))

        #Initialize pis
        self.pis = np.zeros((self.nS,self.actions,len(self.zis)))

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

    def normalize(self,x):
        # Normalize values for each value in pis
        return x / x.sum(axis=0)


    def Categorial_Algorithm(self,xt,at,rt,nextx):
        #initializations
        self.Q[nextx] = np.dot(self.pis[nextx],self.zis)
        a_star = np.argmax(self.Q[nextx],axis=0)
        m = np.zeros(self.atoms)

        #main cicle
        for j in range(self.atoms):
            Tz = min(self.Vmax, max(self.Vmin, rt + self.gamma * self.zis[j]))
            b = (Tz - self.Vmin) / self.delta_z
            #it might happen due to rounding problems
            if b > (self.atoms - 1):
                b = self.atoms - 1
            l = math.floor(b)
            u = math.ceil(b)

            if u == b and b == l:
                m[l] = m[l] + self.pis[nextx,a_star,j]
            else:
                m[l] = m[l] + self.pis[nextx,a_star,j] * (u - b)
                m[u] = m[u] + self.pis[nextx,a_star,j] * (b - l)
        return m

    def initialize(self,state,action):
        reward = self.Rl[state,action]

        m = np.zeros(self.atoms)

        #The distribution will be a single point
        Tz = min(self.Vmax, max(self.Vmin, reward))
        b = (Tz - self.Vmin) / self.delta_z
        l = math.floor(b)
        u = math.ceil(b)

        if b > (self.atoms - 1):
            b = self.atoms - 1

        if u == b and b == l:
            m[l] = 1
        else:
            m[l] = (u - b)
            m[u] = (b - l)

        return m 

    # goal states are the objective, normally the actions with the highest reward
    def exploration(self):

        #initialize the probability at 0
        for state in range(self.nS):
            for a in range(self.actions):
                self.pis[state,a] = self.initialize(state,a)


        err = 1

        while(err > 1e-8):
            npis= np.copy(self.pis)

            for state in range(self.nS):
                for action in range(self.actions):
                    new_p = np.zeros(self.atoms)
                    reward = self.Rl[state,action]

                    #should not happen if the transitions follow the rule that all probabilities add to 1
                    flagnotransition = True 

                    for nextstate in range(self.nS):
                        if self.Pl[state,action,nextstate] > 0:
                            flagnotransition = False
                            m = self.Categorial_Algorithm(state,action,reward,nextstate)
                            new_p = np.add(new_p, self.Pl[state,action,nextstate] * m)
                    
                    if flagnotransition:
                        new_p = self.initialize(state,action)

                    npis[state,action] = new_p

            err = np.linalg.norm(self.pis-npis)

            self.pis = np.copy(npis)
            