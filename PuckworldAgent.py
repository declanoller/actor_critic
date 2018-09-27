import matplotlib.pyplot as plt
import numpy as np
from math import sqrt



class PuckworldAgent:


    def __init__(self,**kwargs):

        (self.xlim,self.ylim) = (1.0,1.0)
        self.lims = np.array((self.xlim,self.ylim))
        self.max_dist = sqrt(np.sum(self.lims**2))
        self.a = 0.2
        self.time_step = kwargs.get('dt',10**-1)

        self.N_actions = 4


        self.circ_rad = self.xlim/20.0
        self.addTarget()

        self.pos0 = np.array([self.xlim/2.0,self.ylim/2.0])
        self.v0 = np.array([0.0,0.0])
        self.resetStateValues()
        self.accel_array = np.array([[0,1],[0,-1],[-1,0],[1,0]])



        self.N_state_terms = len(self.getStateVec())



    def addToHist(self):
        self.pos_hist = np.concatenate((self.pos_hist,[self.pos]))
        self.v_hist = np.concatenate((self.v_hist,[self.v]))
        self.t.append(self.t[-1] + self.time_step)
        self.r_hist.append(self.reward())


    def addTarget(self):

        self.target = self.circ_rad + np.random.random((2,))*(1-2*self.circ_rad)


    def iterateEuler(self,action):

        #this uses the Euler-Cromer method to move.

        #Right now I'm just gonna make it sit against a wall if it goes to the
        #boundary, but it might be cool to make periodic bry conds, to see if it would
        #learn to zoom around it.

        a = self.actionToAccel(action) - .3*self.v

        v_next = self.v + a*self.time_step
        pos_next = self.pos + v_next*self.time_step

        #To handle the walls
        for i in [0,1]:
            if pos_next[i] < (0 + self.circ_rad):
                pos_next[i] = 0 + self.circ_rad
                v_next[i] = -v_next[i]

            if pos_next[i] > (self.lims[i] - self.circ_rad):
                pos_next[i] = self.lims[i] - self.circ_rad
                v_next[i] = -v_next[i]

        self.pos = pos_next
        self.v = v_next
        self.addToHist()


    def actionToAccel(self,action):
        self.a_hist.append(action)
        return(self.a*self.accel_array[action])





    def getFeatureVec(self,state,action):
        #Here, state is a 4-array of [x,y,vx,vy]
        #So it returns a 4x4 matrix where each column is for a different action.
        fv = np.zeros((self.N_state_terms,self.N_actions))
        fv[:,action] = state
        return(fv)


    def getStateVec(self):
        assert self.target is not None, 'Need target to get state vec'
        #return(np.concatenate((self.pos,self.v,self.target,(self.pos-self.target)**2)))
        #return(np.concatenate((self.pos,self.v,self.target,[(self.pos[0]-self.target[0])],[(self.pos[1]-self.target[1])])))
        #return(np.array([(self.pos[0]-self.target[0]),(self.pos[1]-self.target[1])]))
        return(np.concatenate((self.pos,self.v,self.target)))


    def reward(self):

        assert self.target is not None, 'Need a target'
        #Currently just gonna do a limited inverse from the pos of the target.
        max_R = 10
        #return(max_R*(.5*self.max_dist-self.puckTargetDist()) - 1)
        #return(1/(1/max_R + sqrt(np.sum((self.pos-self.target)**2)) ) )
        #return(1.0/(self.puckTargetDist()**2 + 1/max_R))
        return(-max_R*self.puckTargetDist())


    def puckTargetDist(self):
        return(sqrt(np.sum((self.pos-self.target)**2)))


    def initEpisode(self):
        self.resetStateValues()
        #self.addTarget()


    def iterate(self,action):
        self.iterateEuler(action)


    def resetStateValues(self):

        self.pos = self.pos0
        self.v = self.v0

        self.pos_hist = np.array([self.pos])
        self.v_hist = np.array([self.v])
        self.action_hist = [0]
        self.t = [0]
        self.a_hist = [0]
        self.r_hist = []


    def drawState(self,ax):

        ax.clear()
        ax.set_xlim((0,self.xlim))
        ax.set_ylim((0,self.ylim))

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

        puck = plt.Circle(tuple(self.pos), self.circ_rad, color='tomato')
        ax.add_artist(puck)

        if self.target is not None:
            target = plt.Circle(tuple(self.target), self.circ_rad, color='seagreen')
            ax.add_artist(target)


    def plotStateParams(self,axes):

        ax1 = axes[0]
        ax2 = axes[1]
        ax3 = axes[2]

        ax1.clear()
        ax1.plot(self.pos_hist[:,0],label='x')
        ax1.plot(self.pos_hist[:,1],label='y')
        ax1.legend()

        ax2.clear()
        ax2.plot(self.r_hist,label='R')
        ax2.legend()


        ax3.clear()
        ax3.plot(self.a_hist,label='a')
        ax3.legend()






#
