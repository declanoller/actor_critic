import Agent_Tests as at
from Agent import Agent
from PuckworldAgent import PuckworldAgent


at.varyParam(N_steps=[10**3,10**4,10**5],N_runs=3,show_plot=True,features='AC')

exit(0)
#at.multipleEpisodesNewAgent(show_plot=True)

ag = Agent(agent_class=PuckworldAgent,features='AC')
ag.ACepisode(N_steps=10**3,show_plot=True)




























#
