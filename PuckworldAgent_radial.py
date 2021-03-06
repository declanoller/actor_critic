import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, pi, cos, sin



class PuckworldAgent_radial:


    def __init__(self,**kwargs):

        self.xlims = np.array([-0.5,0.5])
        self.ylims = np.array([-0.5,0.5])
        self.lims = np.array((self.xlims,self.ylims))
        self.max_dist = sqrt(np.ptp(self.xlims)**2 + np.ptp(self.ylims)**2)
        self.a = kwargs.get('a',1.0)
        self.drag = 0.5
        self.time_step = kwargs.get('dt',10**-1)
        self.reward_type = kwargs.get('reward','sparse')

        self.passed_params = {}
        check_params = ['a', 'dt', 'reward']
        for param in check_params:
            if kwargs.get(param, None) is not None:
                self.passed_params[param] = kwargs.get(param, None)

        #Here, the 4 actions will be, in order: forward (F), backward (B), CCW, clockwise turn (CW)
        self.N_actions = 4
        #This is how many different angles it can have. I think 32 should be good.
        self.d_theta = 2*pi/4.0


        self.circ_rad = np.ptp(self.xlims)/10.0
        self.target_rad = 1*self.circ_rad
        self.resetTarget()

        #For this, the state/position now has an (x,y) coord AND an angle from the horizontal right (0 degrees).
        self.pos0 = np.array([self.xlims.mean()/2.0,self.ylims.mean()/2.0])
        self.angle0 = 0
        self.v0 = np.array([0.0,0.0])
        self.resetStateValues()
        self.accel_array = np.array([[0,1],[0,-1],[-1,0],[1,0]])

        self.N_state_terms = len(self.getStateVec())


    def getPassedParams(self):
        #This returns a dict of params that were passed to the agent, that apply to the agent.
        #So if you pass it a param for 'reward', it will return that, but it won't return the
        #default val if you didn't pass it.

        return(self.passed_params)


    def addToHist(self):
        self.pos_hist = np.concatenate((self.pos_hist,[self.pos]))
        self.v_hist = np.concatenate((self.v_hist,[self.v]))
        self.angle_hist = np.concatenate((self.angle_hist,[self.angle]))
        self.t.append(self.t[-1] + self.time_step)
        self.r_hist.append(self.reward())


    def resetTarget(self):

        self.target = self.target_rad + self.lims[:,0] + np.random.random((2,))*(np.ptp(self.lims,axis=1)-2*self.target_rad)


    def iterateEuler(self,action):

        #this uses the Euler-Cromer method to move.

        #Right now I'm just gonna make it sit against a wall if it goes to the
        #boundary, but it might be cool to make periodic bry conds, to see if it would
        #learn to zoom around it.
        self.a_hist.append(action)

        if action in [0,1]:

            a = self.actionToAccel(action) - self.drag*self.v

            v_next = self.v + a*self.time_step
            pos_next = self.pos + v_next*self.time_step

            #To handle the walls
            for i in [0,1]:
                if pos_next[i] < (self.lims[i,0] + self.circ_rad):
                    pos_next[i] = self.lims[i,0] + self.circ_rad
                    v_next[i] = -v_next[i]

                if pos_next[i] > (self.lims[i,1] - self.circ_rad):
                    pos_next[i] = self.lims[i,1] - self.circ_rad
                    v_next[i] = -v_next[i]

            self.pos = pos_next
            self.v = v_next
            self.addToHist()

        else:
            #These should try to enforce it to be between -pi and +pi.
            if action==2:
                self.angle = (self.angle + self.d_theta)
                if self.angle > pi:
                    self.angle -= 2*pi
            if action==3:
                self.angle = (self.angle - self.d_theta)
                if self.angle < -pi:
                    self.angle += 2*pi


    def actionToAccel(self,action):
        #This maps (0,1) to (1, -1)
        a = -self.a*(2*(action - 0.5))
        return(np.array([a*cos(self.angle), a*sin(self.angle)]))


    def getStateVec(self):
        assert self.target is not None, 'Need target to get state vec'
        #return(np.concatenate((self.pos,self.v,self.target,(self.pos-self.target)**2)))
        #return(np.concatenate((self.pos,self.v,self.target,[(self.pos[0]-self.target[0])],[(self.pos[1]-self.target[1])])))
        #return(np.array([(self.pos[0]-self.target[0]),(self.pos[1]-self.target[1])]))
        return(np.concatenate((self.pos, [self.angle/pi], self.v, self.target)))


    def reward(self):

        assert self.target is not None, 'Need a target'

        max_R = 1

        if self.reward_type == 'sparse':
            if self.puckTargetDist() <= (self.target_rad + self.circ_rad):
                return(max_R)
            else:
                return(-0.01)

        if self.reward_type == 'shaped':
            #return(max_R*(self.max_dist/2.0 - self.puckTargetDist()))
            #These numbers will probably have to change if a, dt, or the dimensions change.
            return(-1*self.puckTargetDist() + 0.45)


    def puckTargetDist(self):
        return(sqrt(np.sum((self.pos-self.target)**2)))


    def initEpisode(self):
        self.resetStateValues()
        self.resetTarget()


    def iterate(self,action):
        self.iterateEuler(action)

        r = self.reward()
        if r > 0:
            self.resetTarget()

        return(r,self.getStateVec())


    def resetStateValues(self):

        self.pos = self.pos0
        self.angle = self.angle0
        self.v = self.v0

        self.pos_hist = np.array([self.pos])
        self.v_hist = np.array([self.v])
        self.action_hist = [0]
        self.angle_hist = [self.angle]
        self.t = [0]
        self.a_hist = [0]
        self.r_hist = []


    def drawState(self,ax):

        ax.clear()
        ax.set_xlim(tuple(self.xlims))
        ax.set_ylim(tuple(self.ylims))

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

        puck = plt.Circle(tuple(self.pos), self.circ_rad, color='tomato')
        ax.add_artist(puck)

        tail = self.pos + np.array([self.circ_rad*cos(self.angle + pi), self.circ_rad*sin(self.angle + pi)])
        tweak = 0.4
        w = 0.01
        ax.arrow(tail[0], tail[1], tweak*2*self.circ_rad*cos(self.angle), tweak*2*self.circ_rad*sin(self.angle), width=w, head_width=8*w, color='black')

        if self.target is not None:
            target = plt.Circle(tuple(self.target), self.target_rad, color='seagreen')
            ax.add_artist(target)


    def plotStateParams(self,axes):

        ax1 = axes[0]
        ax2 = axes[1]
        ax3 = axes[2]
        ax4 = axes[3]

        ax1.clear()
        ax1.plot(self.pos_hist[:,0][-1000:],label='x')
        ax1.plot(self.pos_hist[:,1][-1000:],label='y')
        ax1.plot(self.angle_hist[-1000:],label='theta')
        ax1.legend()

        ax2.clear()
        ax2.plot(self.a_hist[-1000:],label='a')
        ax2.set_yticks([0,1,2,3])
        ax2.set_yticklabels(['F', 'B', 'CCW', 'CW'])
        ax2.legend()


        ax3.clear()
        ax3.plot(self.r_hist[-1000:],label='R')
        ax3.legend()


        ax4.clear()
        ax4.plot(self.v_hist[:,0][-1000:],label='vx')
        ax4.plot(self.v_hist[:,1][-1000:],label='vy')
        ax4.legend()




#
