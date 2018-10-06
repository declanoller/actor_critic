from Agent import Agent
from PuckworldAgent import PuckworldAgent
import matplotlib.pyplot as plt
from statistics import mean,stdev
import FileSystemTools as fst
from time import time
import numpy as np
import os


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




def gifFromModel(model_fname,N_steps):

    #This will definitely shit the bed if you change the way the files are named.
    log_fname = model_fname.replace('model_','log_').replace('.model','.txt')
    param_dict = fst.readFileToDict(log_fname)


    path = fst.dirFromFullPath(log_fname) + 'gifs'

    if not os.path.isdir(path):
        os.mkdir(path)

    ds = fst.getDateString()
    path = fst.combineDirAndFile(path, ds)

    if not os.path.isdir(path):
        os.mkdir(path)


    ag = Agent(agent_class=PuckworldAgent, **param_dict, dir=path)
    ag.loadModelPlay(model_fname, show_plot=False, save_plot=True, make_gif=True, N_steps=20)

    fst.gifFromImages(path, ds + '.gif')














#
