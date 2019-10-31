import numpy as np
import random
from random import randint
import collections
from collections import Counter 
import matplotlib.pyplot as plt
from tabulate import tabulate

#self made files
from singleandpath import *
from path_secure import *
from C51 import *
from plot_distributions import *


class Explanations():
    def __init__(self,nS,Pl,Rl,actions,gamma,goallist,Pol=None,V=None,Q=None):
        self.nS = nS
        self.Pl = Pl
        self.Rl = Rl
        self.actions = actions
        self.gamma = gamma
        self.goallist = goallist
        if Pol == None or V == None or Q == None:
            self.Q = np.zeros((self.nS,self.actions))
            self.V = np.zeros((self.nS))
            self.value_iteration()
        else:
            self.Pol = Pol
            self.V = V
            self.Q=Q

        self.create_singleandpath()
        self.create_path_secure()

        #c51 values
        self.vmax = None
        self.vmin = None
        self.c51 = None 

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


    def create_singleandpath(self):
        self.single =  singlepath_explanation(self.nS,self.Pl,self.Rl,self.actions,self.gamma,self.goallist,self.Pol,self.V,self.Q)

    def create_C51(self,vmax,vmin,atoms=51):
        self.vmax = vmax
        self.vmin = vmin
        self.atoms = atoms
        self.c51 = C51(self.nS,self.actions,self.gamma,vmax,vmin,self.Pl,self.Rl,atoms)
        self.c51.exploration()

    def create_path_secure(self):
        self.path_s =  path_secure(self.nS,self.Pl,self.Rl,self.actions,self.gamma,self.goallist,self.Pol,self.V,self.Q)

    #singleandpath functions

    def single_get_state_explanation_distribution(self,state,n=100000,max_steps=100000,change_Pl=True,proc_actions='all',showplot=True,savefigure=True,namefigure='',allaction1plot=True,nbins=21,colors=None,labels=None):
        distribution_steps, distribution_reward = self.single.state_explain(state,change_Pl,proc_actions,n=n,max_steps=max_steps)
        if labels == None:
            labels = []
        if colors == None:
            colors = []
        if showplot or savefigure:
            if proc_actions == 'all':
                if allaction1plot:
                    listactions = []
                    for action in range(self.actions):
                        listactions.append(action)
                    all_actions_histigram_plot(distribution_reward,listactions,show=showplot,save=savefigure,savename=namefigure,nbins=nbins,colors=colors,labels=labels)
                else:
                    for i in range(self.actions):
                        one_actions_histigram_plot(distribution_reward[i],show=showplot,save=savefigure,savename=namefigure,nbins=nbins)
            else:
                if allaction1plot:
                        all_actions_histigram_plot(distribution_reward,proc_actions,show=showplot,save=savefigure,savename=namefigure,nbins=nbins,colors=colors,labels=labels)
                else:
                    for el in proc_actions:
                        one_action_histigram_plot(distribution_reward[el],show=showplot,save=savefigure,savename=namefigure,nbins=nbins)
        return distribution_steps, distribution_reward

    def get_path_distribution(self,path,pathactions,n=100000,showplot=True,savefigure=True,namefigure='',nbins=21):
        distribution_steps, distribution_reward, n_incomplete = self.single.path_explain(path,pathactions,n=n)
        if n_incomplete > 0:
            print('There were ' + str(n_incomplete) + ' paths due to bifurcations on the path')
        if showplot or savefigure:
            one_action_histigram_plot(distribution_reward,show=showplot,save=savefigure,savename=namefigure,nbins=nbins)
        return distribution_steps, distribution_reward

    #show optimal paths

    def get_optimal_path_from_state(self,start_pos,remove_path_cicles=False,remove_self_trans=False,only_policy_path=False):
        state_path = self.single.get_optimal_paths(start_pos,remove_path_cicles,remove_self_trans,only_policy_path)
        return state_path

    #bifurcations information

    def get_state_bifurcations(self,state,action,printbifurcations=True):
        bifurcations = self.single.state_bifurcations(state,action)
        if printbifurcations:
            print('The bifurcations of state ' + str(state) + ' following action ' + str(action) + ' are: ' + str(bifurcations))
            print('Use c51_state_distribution to get more information of a bifurcation')
        return bifurcations

    def get_path_bifurcations_outside_path(self,path,pathactions,printbifurcations=True):
        bifurcations = self.single.path_bifurcations(path,pathactions)
        if printbifurcations:
            for i in range(len(path)-1):
                print('The bifurcations of state ' + str(path[i]) + ' following action ' + str(pathactions[i]) + ' are: ' + str(bifurcations[i]))
            print('Use c51_state_distribution to get more information of a bifurcation')
        return bifurcations

    def get_path_all_bifurcations(self,path,pathactions,printbifurcations=True):
        bifurcations = self.single.all_bifurcations(path,pathactions)
        if printbifurcations:
            for i in range(len(path)-1):
                print('The bifurcations of state ' + str(path[i]) + ' following action ' + str(pathactions[i]) + ' are: ' + str(bifurcations[i]))
            print('Use c51_state_distribution to get more information of a bifurcation')
        return bifurcations

    #c51 functions
    #if procactions only has 1 element and showbifurcations is true it will instead show the plot organized by possible states by a single action
    def c51_state_distribution(self,state,vmin,vmax,proc_actions='all',showbifurcations=True,showplot=True,namefigure='',savefigure=True,allaction1plot=True,atoms=51,colors=None,labels=None):
        if vmin != self.vmin or vmax != self.vmax or self.c51 == None:
            self.create_C51(vmax,vmin,atoms)

        if labels is None:
            labels = []
        if colors is None:
            colors = []

        if showplot or savefigure:
            distribution_reward_keys = self.c51.zis
            distribution_reward = {}
            if proc_actions == 'all':
                for action in range(self.actions):
                    distribution_reward_values = self.c51.pis[state,action]
                    distribution_reward[action] = {}
                    for i in range(len(distribution_reward_keys)):
                        distribution_reward[action][distribution_reward_keys[i]] = distribution_reward_values[i]
                if allaction1plot:
                    listactions = []
                    for action in range(self.actions):
                        listactions.append(action)
                    all_actions_histigram_plot(distribution_reward,listactions,density=False,show=showplot,save=savefigure,savename=namefigure,nbins=atoms,colors=colors,labels=labels)
                else:
                    for i in range(self.actions):
                        one_actions_histigram_plot(distribution_reward[i],density=False,show=showplot,save=savefigure,savename=namefigure,nbins=atoms)
            else:
                if len(proc_actions) == 1 and showbifurcations: 
                    action = proc_actions[0]
                    numberstates = 0
                    for nstate in range(self.nS):
                        if self.Pl[state,action,nstate] > 0:
                            numberstates += 1
                            distribution_reward_values = self.c51.pis[nstate,self.Pol[nstate]]
                            distribution_reward[nstate] = {}
                            for i in range(len(distribution_reward_keys)):
                                distribution_reward[nstate][distribution_reward_keys[i]] = distribution_reward_values[i]
                    action_biforcation_histogram_plot(distribution_reward,numberstates,density=False,show=showplot,save=savefigure,savename=namefigure,nbins=atoms,colors=colors,labels=labels)
                else:
                    for action in proc_actions:
                        distribution_reward_values = self.c51.pis[state,action]
                        distribution_reward[action] = {}
                        for i in range(len(distribution_reward_keys)):
                            distribution_reward[action][distribution_reward_keys[i]] = distribution_reward_values[i]
                    if allaction1plot:
                        all_actions_histigram_plot(distribution_reward,proc_actions,density=False,show=showplot,save=savefigure,savename=namefigure,nbins=atoms,colors=colors,labels=labels)
                    else:
                        for el in proc_actions:
                            one_action_histigram_plot(distribution_reward[el],density=False,show=showplot,save=savefigure,savename=namefigure,nbins=atoms)

    #path security functions

    def get_path_security_distribution(self,path,pathactions,epsilon=0.3,always_return_path=False,return_path_if_same_or_new=True,Never_return_path=False,n=100000,max_steps=100000,showplot=True,savefigure=True,namefigure='',nbins=21):
        distribution_steps, distribution_reward = self.path_s.path_secure_distribution(path,pathactions,epsilon,always_return_path,return_path_if_same_or_new,Never_return_path,n,max_steps=max_steps)
        if showplot or savefigure:
            one_action_histigram_plot(distribution_reward,show=showplot,save=savefigure,savename=namefigure,nbins=nbins)
        return distribution_steps, distribution_reward



# if __name__ == "__main__":
    # #example 1
    # Pl = np.zeros((4,1,4))
    # Pl[0,0,1] = 0.5
    # Pl[0,0,2] = 0.5
    # Pl[1,0,3] = 1
    # Pl[2,0,2] = 0.5
    # Pl[2,0,3] = 0.5


    # Rl = np.zeros((4,1))
    # Rl[0,0] = 0
    # Rl[1,0] = -500
    # Rl[2,0] = 0
    # Rl[3,0] = 1000

    # path = [0,2,3]
    # pathactions = [0,0]
    # expl = Explanations(4,Pl,Rl,1,0.90,[3])

    # #example 2
    # #cliffwalk
    # # Pl = np.zeros((15,4,15))
    # # Pl[0,0,5] = 1
    # # Pl[0,3,1] = 1
    # # Pl[1,0,6] = 1
    # # Pl[1,2,0] = 1
    # # Pl[1,3,2] = 1
    # # Pl[2,0,7] = 1
    # # Pl[2,2,1] = 1
    # # Pl[2,3,3] = 1
    # # Pl[3,0,8] = 1
    # # Pl[3,2,2] = 1
    # # Pl[3,3,4] = 1
    # # Pl[4,0,9] = 1
    # # Pl[4,2,3] = 1
    # # Pl[5,0,10] = 1
    # # Pl[5,1,0] = 1
    # # Pl[5,3,6] = 1
    # # Pl[6,0,11] = 1
    # # Pl[6,1,1] = 1
    # # Pl[6,2,5] = 1
    # # Pl[6,3,7] = 1
    # # Pl[7,0,12] = 1
    # # Pl[7,1,2] = 1
    # # Pl[7,2,6] = 1
    # # Pl[7,3,8] = 1
    # # Pl[8,0,13] = 1
    # # Pl[8,1,3] = 1
    # # Pl[8,2,7] = 1
    # # Pl[8,3,9] = 1
    # # Pl[9,0,14] = 1
    # # Pl[9,1,4] = 1
    # # Pl[9,2,8] = 1
    # # Pl[10,1,5] = 1
    # # Pl[10,3,11] = 1
    # # Pl[11,1,6] = 1
    # # Pl[11,2,10] = 1
    # # Pl[11,3,12] = 1
    # # Pl[12,1,7] = 1
    # # Pl[12,2,11] = 1
    # # Pl[12,3,13] = 1
    # # Pl[13,1,8] = 1
    # # Pl[13,2,12] = 1
    # # Pl[13,3,14] = 1
    # # Pl[14,1,9] = 1
    # # Pl[14,2,13] = 1

    # # Rl = np.zeros((15,4))
    # # Rl[1,0] = -100
    # # Rl[1,1] = -100
    # # Rl[1,2] = -100
    # # Rl[1,3] = -100
    # # Rl[2,0] = -100
    # # Rl[2,1] = -100
    # # Rl[2,2] = -100
    # # Rl[2,3] = -100
    # # Rl[3,0] = -100
    # # Rl[3,1] = -100
    # # Rl[3,2] = -100
    # # Rl[3,3] = -100
    # # Rl[4,0] = 100
    # # Rl[4,1] = 100
    # # Rl[4,2] = 100
    # # Rl[4,3] = 100

    # # path = [0,5,6,7,8,9,4]
    # # pathactions = [0,3,3,3,3,1]

    # # expl = Explanations(15,Pl,Rl,4,0.90,[4])

    # #expl.single_get_state_explanation_distribution(0)

    # #expl.c51_state_distribution(0,-400,400)#,proc_actions=[0])

    # expl.c51_state_distribution(0,-400,400, proc_actions=[0])

    # #expl.get_path_distribution(path,pathactions)

    # #bif = expl.get_path_bifurcations_outside_path(path,pathactions)

    # #bif = expl.get_state_bifurcations(0,0)

    # #pth = expl.get_optimal_path_from_state(0)

    # #print(pth)

    # #expl.get_path_security_distribution(path,pathactions)
