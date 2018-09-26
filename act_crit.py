from Environment import Environment
from Agent import Agent

from PuckworldAgent import PuckworldAgent


env = Environment()

ag = Agent(env,agent_class=PuckworldAgent)



ag.createFigure()
ag.drawState()
ag.showFig()

ag.episode()

exit(0)































#
