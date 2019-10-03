import numpy as np
import matplotlib.pyplot as plt
import random

colorpallet = ['lime','blue','green','red','orange','purple','yellow','grey','magenta','cyan']

def one_action_histigram_plot(distribution,show=True,save=True,savename='',nbins=21,density=False,histtype='bar',color='blue'):
    plt.hist(list(distribution.keys()) , nbins,weights=list(distribution.values()),density=density, histtype=histtype, color=color)
    plt.xlabel('Reward Values', fontsize=15)
    plt.ylabel('Probability', fontsize=15)
    
    if save == True:
        if savename:
            plt.savefig(savename)
        else:
            plt.savefig('plot.png')
    if show == True:
        plt.show()
        

def all_actions_histigram_plot(distribution,list_actions,show=True,save=True,savename='',nbins=21,density=False,histtype='bar',colors=None,labels=None,stacked=False):
    if colors == [] or colors == []:
        if len(list_actions) <= len(colorpallet):
            colors = colorpallet[:len(list_actions)]
        else:
            colors = colorpallet + ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(list_actions)-len(colorpallet))]
    keys = []
    weights = []
    defaultlabels = False
    if labels == [] or labels == []:
        defaultlabels = True
    for action in list_actions:
        keys.append(list(distribution[action].keys()))
        weights.append(list(distribution[action].values()))
        if defaultlabels:
            labels.append('Action ' + str(action))

    plt.hist(keys , nbins,weights=weights,density=density, histtype=histtype, color=colors,label=labels,stacked=stacked)
    plt.xlabel('Reward Values', fontsize=15)
    plt.ylabel('Probability', fontsize=15)
    plt.legend(prop={'size': 15})
    
    if save == True:
        if savename:
            plt.savefig(savename)
        else:
            plt.savefig('plot.png')
    if show == True:
        plt.show()


def action_biforcation_histogram_plot(distribution,numberstates,show=True,save=True,savename='',nbins=51,density=False,histtype='bar',colors=None,labels = None,stacked=False):
    if colors == None or colors == []:
        if numberstates <= len(colorpallet):
            colors = colorpallet[:numberstates]
        else:
            colors = colorpallet + ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(numberstates-len(colorpallet))]
    keys = []
    weights = []
    defaultlabels = False
    if labels == None or labels == []:
        defaultlabels = True
    for state in list(distribution.keys()):
        keys.append(list(distribution[state].keys()))
        weights.append(list(distribution[state].values()))
        if defaultlabels:
            labels.append('State ' + str(state))

    plt.hist(keys , nbins,weights=weights,density=density, histtype=histtype, color=colors,label=labels,stacked=stacked)
    plt.xlabel('Reward Values', fontsize=15)
    plt.ylabel('Probability', fontsize=15)
    plt.legend(prop={'size': 15})

    if save == True:
        if savename:
            plt.savefig(savename)
        else:
            plt.savefig('plot.png')
    if show == True:
        plt.show()