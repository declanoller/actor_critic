import matplotlib.pyplot as plt
from random import randint,random,sample
import numpy as np
from math import atan,sin,cos,sqrt,ceil,floor,log
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DQN(nn.Module):

    def __init__(self,D_in,H,D_out):
        super(DQN, self).__init__()
        self.lin1 = nn.Linear(D_in,H)
        self.lin2 = nn.Linear(H,D_out)

    def forward(self, x):
        out = self.lin2(torch.tanh(self.lin1(x)))
        return(out)


class Pw_Agent:

    def __init__(self,**kwargs):

        self.agent = PuckworldAgent()

        self.gamma = kwargs.get('gamma',1.0)
        self.epsilon = kwargs.get('epsilon',0.0)

        self.N_batch = 20

        self.initLearningParams()


    def initLearningParams(self):

        self.dtype = torch.float64
        self.device = torch.device("cpu")

        torch.set_default_dtype(self.dtype)
        torch.set_default_tensor_type(torch.DoubleTensor)

        D_in, H, D_out = self.agent.N_state_terms, 100, self.agent.N_actions
        self.policy_NN = DQN(D_in,H,D_out)
        self.target_NN = DQN(D_in,H,D_out)
        self.target_NN.load_state_dict(self.policy_NN.state_dict())
        self.target_NN.eval()
        self.optimizer = optim.RMSprop(self.policy_NN.parameters())
        self.samples_Q = []


    def updateTargetNetwork(self):
        self.target_NN.load_state_dict(self.policy_NN.state_dict())

    def resetStateValues(self):
        self.agent.resetStateValues()

    def forwardPassQ(self,state_vec):
        Q_s_a = self.policy_NN(state_vec)
        return(Q_s_a)

    def forwardPassQFrozen(self,state_vec):
        Q_s_a = self.target_NN(state_vec)
        return(Q_s_a)


    def singleStateForwardPassQ(self,state_vec):
        qsa = torch.squeeze(self.forwardPassQ(torch.unsqueeze(torch.Tensor(state_vec),dim=0)))
        return(qsa)


    def greedyAction(self,state_vec):
        qsa = self.singleStateForwardPassQ(state_vec)
        return(torch.argmax(qsa))


    def epsGreedyAction(self,state_vec):
        if random()>self.epsilon:
            return(self.greedyAction(state_vec))
        else:
            return(self.getRandomAction())

    def getRandomAction(self):
        return(randint(0,self.agent.N_actions-1))


    def DQNepisode(self,show_plot=True,save_plot=False,N_steps=10**3):

        if show_plot:
            self.showFig()

        R_tot = 0
        self.agent.initEpisode()

        s = self.agent.getStateVec()
        a = self.epsGreedyAction(s)
        #Iterate automatically puts it in the next state.
        self.agent.iterate(a)
        r = self.agent.reward()

        for i in range(N_steps):

            if i%11==0 and i>self.N_batch:
                self.updateTargetNetwork()

            self.epsilon *= .99
            R_tot += r
            s_next = self.agent.getStateVec()
            a_next = self.epsGreedyAction(s_next)

            experience = (s,a,r,s_next)
            self.samples_Q.append(experience)

            if len(self.samples_Q)>=2*self.N_batch:

                #Get random batch
                batch_Q_samples = sample(self.samples_Q,self.N_batch)
                states = torch.Tensor(np.array([samp[0] for samp in batch_Q_samples]))
                actions = [samp[1] for samp in batch_Q_samples]
                rewards = torch.Tensor([samp[2] for samp in batch_Q_samples])
                states_next = torch.Tensor([samp[3] for samp in batch_Q_samples])

                #Get current Q value and target value
                Q_cur = self.forwardPassQ(states)[list(range(len(actions))),actions]
                Q_next = torch.max(self.forwardPassQFrozen(states_next),dim=1)[0]

                TD0_error = F.smooth_l1_loss(Q_cur,(rewards + self.gamma*Q_next).detach())

                self.optimizer.zero_grad()
                TD0_error.backward()
                for param in self.policy_NN.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.optimizer.step()

            s = s_next
            a = a_next
            self.agent.iterate(a)
            r = self.agent.reward()

            if show_plot:
                self.plotAll()
                self.fig.canvas.draw()


        print('puck-target dist: {:.2f}, R_tot/N_steps: {:.2f}'.format(self.agent.puckTargetDist(),R_tot/N_steps))
        return(R_tot)


    def plotAll(self):
        self.drawState()
        self.plotStateParams()


    def drawState(self):
        self.agent.drawState(self.ax_state)


    def plotStateParams(self):
        self.agent.plotStateParams([self.ax_state_params1,self.ax_state_params2,self.ax_state_params3])


    def createFigure(self):

        self.fig, self.axes = plt.subplots(3,3,figsize=(12,8))
        self.ax_state = self.axes[0,0]
        self.ax_state_params1 = self.axes[0,1]
        self.ax_state_params2 = self.axes[0,2]
        self.ax_state_params3 = self.axes[1,2]
        #self.plotAll()

        self.ax_wQ = self.axes[2,1]
        self.ax_theta_pi = self.axes[1,1]

        self.ax_loc_vals = self.axes[2,0]


    def showFig(self):
        plt.show(block=False)


###########################


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
        #Added a bit of friction here, doesn't make a difference though.
        a = self.actionToAccel(action) - .3*self.v

        v_next = self.v + a*self.time_step
        pos_next = self.pos + v_next*self.time_step

        #To handle the walls: move it back inside, and reverse its velocity
        #i is the x and y dimensions.
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


    def getStateVec(self):
        assert self.target is not None, 'Need target to get state vec'
        return(np.concatenate((self.pos,self.v,self.target)))


    def reward(self):
        assert self.target is not None, 'Need a target'
        max_R = 10
        return(-max_R*self.puckTargetDist())


    def puckTargetDist(self):
        return(sqrt(np.sum((self.pos-self.target)**2)))


    def initEpisode(self):
        self.resetStateValues()


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
