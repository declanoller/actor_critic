from Environment import Environment
from Agent import Agent

from PuckworldAgent import PuckworldAgent

import matplotlib.pyplot as plt
from statistics import mean,stdev


env = Environment()

for i in range(5):
    ag = Agent(env,agent_class=PuckworldAgent,features='DQN')
    #ag.createFigure()
    ag.episode(show_plot=False,N_steps=10**4)

exit(0)


alphas = [10**-i for i in list(range(1,7))]
R_tots = []
SD = []
for alpha in alphas:
    print('alpha: ',alpha)
    runs = 5
    err = []
    for run in range(runs):
        ag = Agent(env,agent_class=PuckworldAgent,features='DQL',ep_time=10**3,alpha=alpha)
        err.append(ag.DQLepisode(show_plot=False))

    R_tots.append(mean(err))
    SD.append(stdev(err))


plt.close('all')
fig,axes = plt.subplots(1,1,figsize=(8,8))
#plt.plot(R_tots)
plt.errorbar(list(range(len(alphas))),R_tots,yerr=SD,fmt='ro-')
plt.show()
exit(0)































#
