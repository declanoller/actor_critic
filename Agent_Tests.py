from Agent import Agent
from PuckworldAgent import PuckworldAgent
import matplotlib.pyplot as plt
from statistics import mean,stdev
import FileSystemTools as fst
from time import time
import numpy as np


def multipleEpisodesNewAgent(**kwargs):
    st = fst.getCurTimeObj()
    params = {}
    params['N_eps'] = kwargs.get('N_eps',5)
    params['N_steps'] = kwargs.get('N_steps',10**3)
    show_plot = kwargs.get('show_plot',False)

    params_str = fst.paramDictToFnameStr(params)
    date_time = fst.getDateString()
    dir = fst.makeLabelDateDir(params_str)
    ext = '.png'
    fname = params_str + date_time + ext

    R_tots = []
    for i in range(params['N_eps']):
        ag = Agent(**kwargs,agent_class=PuckworldAgent,dir=dir)
        r_tot = ag.episode(show_plot=False,save_plot=True)
        R_tots.append(r_tot)

    fig = plt.figure(figsize=(8,8))
    ax = fig.subplots(1,1)
    ax.plot(R_tots)
    plt.savefig(fst.combineDirAndFile(dir,fname))
    if show_plot:
        plt.show()
    plt.close()
    print('\n\ntook {} to execute'.format(fst.getTimeDiffStr(st)))




def varyParam(**kwargs):

    st = fst.getCurTimeObj()

    date_time = fst.getDateString()
    notes = kwargs.get('notes','')
    N_runs = kwargs.get('N_runs',3)
    show_plot = kwargs.get('show_plot',False)

    exclude_list = ['notes','N_runs','show_plot']
    vary_params,vary_param_dict_list,vary_param_tups = fst.parseSingleAndListParams(kwargs,exclude_list)

    label = 'vary_' + fst.listToFname(vary_params) + '_' + notes
    dir = fst.makeLabelDateDir(label)
    ext = '.png'
    base_name = fst.combineDirAndFile(dir,label + date_time)
    fname = base_name + ext

    R_tots = []
    SD = []
    for i,kws in enumerate(vary_param_dict_list):

        print('\n{}\n'.format(vary_param_tups[i]))
        results = []
        for j in range(N_runs):
            print('run ',j)

            ag = Agent(**kws,agent_class=PuckworldAgent,dir=dir)
            r_tot = ag.episode(show_plot=False,save_plot=True)
            results.append(r_tot)

        R_tots.append(mean(results))
        if N_runs>1:
            SD.append(stdev(results))


    plt.close('all')
    fig,axes = plt.subplots(1,1,figsize=(6,9))

    if N_runs>1:
        plt.errorbar(list(range(len(R_tots))),R_tots,yerr=SD,fmt='ro-')
    else:
        plt.plot(R_tots,'ro-')

    axes.set_xticks(list(range(len(R_tots))))
    labels = ['\n'.join(['{}={}'.format(k,v) for k,v in param.items()]) for param in vary_param_tups]
    axes.set_xticklabels(labels, rotation='vertical')
    axes.set_ylabel('Total reward')
    plt.tight_layout()
    plt.savefig(fname)

    if N_runs == 1:
        f = open(base_name + '_values.txt','w+')
        for label, val in zip(labels,R_tots):
            f.write('{}\t{}\n'.format(label.replace('\n',','),val))
        f.close()
    else:
        f = open(base_name + '_values.txt','w+')
        for label, val, sd in zip(labels,R_tots,SD):
            f.write('{}\t{}\t{}\n'.format(label.replace('\n',','),val,sd))
        f.close()



    print('\n\ntook {} to execute'.format(fst.getTimeDiffStr(st)))
    if show_plot:
        plt.show()




'''
def varyParam():


    alphas = [10**-i for i in list(range(1,7))]
    R_tots = []
    SD = []
    for alpha in alphas:
        print('alpha: ',alpha)
        runs = 5
        err = []
        for run in range(runs):
            ag = Agent(agent_class=PuckworldAgent,features='DQL',ep_time=10**3,alpha=alpha)
            err.append(ag.DQLepisode(show_plot=False))

        R_tots.append(mean(err))
        SD.append(stdev(err))


    plt.close('all')
    fig,axes = plt.subplots(1,1,figsize=(8,8))
    #plt.plot(R_tots)
    plt.errorbar(list(range(len(alphas))),R_tots,yerr=SD,fmt='ro-')
    plt.show()
    exit(0)'''
