from Agent import Agent
from PuckworldAgent import PuckworldAgent
import matplotlib.pyplot as plt
from statistics import mean,stdev
import FileSystemTools as fst
from time import time
import numpy as np
import os
import glob


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

            if kwargs.get('agent_class', None) is None:
                ag = Agent(**kws, agent_class=PuckworldAgent, dir=dir)
            else:
                ag = Agent(**kws, dir=dir)

            r_tot = ag.episode(show_plot=False, save_plot=True)
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
    labels = ['\n'.join(fst.dictToStringList(param)) for param in vary_param_tups]
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

    plotRewardCurvesByVaryParam(dir)

    if show_plot:
        plt.show()




def gifFromModel(model_fname, N_steps):

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


    ag = Agent(agent_class=PuckworldAgent, **param_dict, dir=path, figsize=(6, 4))
    ag.loadModelPlay(model_fname, show_plot=False, save_plot=True, make_gif=True, N_steps=N_steps)

    fst.gifFromImages(path, ds + '.gif')




def plotRewardCurvesIndividually(fnames, labels, colors=None):

    percent_range = 1.0

    for i, (fname, label) in enumerate(zip(fnames,labels)):

        dat = np.loadtxt(fname)
        if colors is None:
            plt.plot(dat[:int(percent_range*len(dat))], label=label)
        else:
            plt.plot(dat[:int(percent_range*len(dat))], label=label, color=colors[i])


    plt.legend()
    plt.xlabel('time steps')
    plt.ylabel('R_tot/(# time steps)')

    plt.savefig('__'.join(labels) + '.png')

    plt.show()



def plotRewardCurvesByVaryParam(dir, **kwargs):

    #Use the "values" file from now on to get the vary_param values

    val_file_list = glob.glob(fst.addTrailingSlashIfNeeded(dir) + 'vary_' + '*' + '.txt')

    assert len(val_file_list)==1, 'there needs to be exactly one values.txt file.'

    vals_file = val_file_list[0]

    with open(vals_file, 'r') as f:
        vary_param_vals = f.read().split('\n')

    vary_param_vals = [x.split('\t')[0] for x in vary_param_vals if x!='']
    print('vary param vals:', vary_param_vals)
    vary_param_files = [glob.glob(fst.addTrailingSlashIfNeeded(dir) + 'reward_' + '*' + val + '*' + '.txt') for val in vary_param_vals]

    fig, ax = plt.subplots(1, 1, figsize=(10,8))

    line_cols = ['darkred', 'mediumblue', 'darkgreen', 'goldenrod', 'purple', 'darkorange', 'black']
    shade_cols = ['tomato', 'dodgerblue', 'lightgreen', 'khaki', 'plum', 'peachpuff', 'lightgray']
    max_total = -1000
    min_total = 1000
    N_stds = 2
    N_skip = 300

    #This is a really hacky way of lining up curves that are shifted. You pass it a list of
    #how each curve (the avg) will be scaled and how each will be offset. If you don't pass it
    #anything, it won't do anything differently.
    scale_factors = kwargs.get('scale_factors', np.ones(len(vary_param_vals)))
    offsets = kwargs.get('offsets', np.zeros(len(vary_param_vals)))

    for i, (val, file_group) in enumerate(zip(vary_param_vals, vary_param_files)):
        print('val:',val)
        dat_array = np.array([np.loadtxt(fname) for fname in file_group])
        avg = np.mean(dat_array, axis=0)*scale_factors[i] + offsets[i]
        std = np.std(dat_array, axis=0)*scale_factors[i]
        if max((avg + N_stds*std)[N_skip:]) > max_total:
            max_total = max((avg + N_stds*std)[N_skip:])
        if min((avg - N_stds*std)[N_skip:]) < min_total:
            min_total = min((avg - N_stds*std)[N_skip:])
        plt.plot(avg, color=line_cols[i], label=val)
        plt.fill_between(np.array(range(len(avg))), avg - std, avg + std, facecolor=shade_cols[i], alpha=0.5)

    print(max_total, min_total)
    plt.legend()
    plt.xlabel('time steps')
    plt.ylabel('R_tot/(# time steps)')
    plt.ylim((min_total,max_total))

    plt.savefig(fst.addTrailingSlashIfNeeded(dir) + 'allrewards_' + '__'.join(vary_param_vals) + '__' + fst.getDateString() + '.png')

    #plt.show()


#
