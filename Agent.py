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
        x = self.lin1(x)
        #x = F.relu(x)
        x = torch.tanh(x)
        x = self.lin2(x)
        return(x)


class Agent:

    def __init__(self,env,**kwargs):

        self.env = env
        self.agent_class = kwargs.get('agent_class',None)
        self.agent_class_name = self.agent_class.__name__
        self.agent = self.agent_class()

        self.time_step = kwargs.get('dt',10**-1)
        self.ep_time = kwargs.get('ep_time',100)
        self.steps = int(self.ep_time/self.time_step)

        self.gamma = kwargs.get('gamma',1.0)
        self.alpha = kwargs.get('alpha',10**-1)
        self.beta = kwargs.get('beta',.2)
        self.epsilon = kwargs.get('epsilon',0.0)

        self.N_batch = 20
        self.N_hidden_layer_nodes = 100
        self.features = kwargs.get('features','DQN')
        self.initLearningParams()

        #self.createFigure()


    def initLearningParams(self):

        self.dtype = torch.float64
        self.device = torch.device("cpu")

        torch.set_default_dtype(self.dtype)
        torch.set_default_tensor_type(torch.DoubleTensor)


        if self.features == 'linear':

            self.fv_shape = self.agent.getFeatureVec(self.agent.getStateVec(),0).shape

            #init_smallnum = 0.01
            self.w_Q = torch.zeros(self.fv_shape, device=self.device, dtype=self.dtype, requires_grad=False)
            self.theta_pi = torch.zeros(self.fv_shape, device=self.device, dtype=self.dtype, requires_grad=True)
            '''self.w_Q = torch.randn(self.fv_shape, device=self.device, dtype=self.dtype, requires_grad=False)
            self.theta_pi = torch.randn(self.fv_shape, device=self.device, dtype=self.dtype, requires_grad=True)'''

        if self.features == 'DQN':

            D_in, H, D_out = self.agent.N_state_terms, self.N_hidden_layer_nodes, self.agent.N_actions
            self.policy_NN = DQN(D_in,H,D_out)
            self.target_NN = DQN(D_in,H,D_out)
            self.target_NN.load_state_dict(self.policy_NN.state_dict())
            self.target_NN.eval()
            self.optimizer = optim.RMSprop(self.policy_NN.parameters())
            self.samples_Q = []




    def updateFrozenQ(self):
        self.target_NN.load_state_dict(self.policy_NN.state_dict())

    def resetStateValues(self):
        self.agent.resetStateValues()


    def softmaxAction(self,state_vec):

        if self.features == 'linear':
            #state_vec = self.agent.getStateVec()
            #returns the softmax action, as well as the value of pi for (s,a).
            exp_lin_combos = [torch.exp(torch.sum(torch.tensor(self.agent.getFeatureVec(state_vec,a))*self.theta_pi)) for a in range(self.agent.N_actions)]

            norm = sum(exp_lin_combos)
            p_actions = [t/norm for t in exp_lin_combos]
            action = np.random.choice(list(range(self.agent.N_actions)),p=p_actions)
            return(action)

        if self.features == 'NN':
            #print(torch.tensor(self.agent.getFeatureVec(state_vec,a)).shape)

            pi_vals = torch.squeeze(torch.tensor(np.expand_dims(state_vec,axis=0)).mm(self.theta_pi_1).clamp(min=0).mm(self.theta_pi_2).softmax(dim=1))
            #pi_vals = [torch.tensor(np.expand_dims(state_vec,axis=0)).mm(self.theta_pi_1).clamp(min=0).mm(self.theta_pi_2) for a in range(self.agent.N_actions)]
            #norm = torch.sum(pi_vals)
            #print(pi_vals)
            #p_actions = [t/norm for t in pi_vals]
            #print(p_actions)
            action = np.random.choice(list(range(self.agent.N_actions)),p=pi_vals.detach().numpy())
            return(action)

    def policyVal(self,state_vec,action):

        #state_vec = self.agent.getStateVec()
        #returns the softmax action, as well as the value of pi for (s,a).
        if self.features == 'linear':
            exp_lin_combos = [torch.exp(torch.sum(torch.tensor(self.agent.getFeatureVec(state_vec,a))*self.theta_pi)) for a in range(self.agent.N_actions)]

            norm = sum(exp_lin_combos)
            p_actions = [t/norm for t in exp_lin_combos]
            return(p_actions[action])

        if self.features == 'NN':
            pi_vals = torch.squeeze(torch.tensor(np.expand_dims(state_vec,axis=0)).mm(self.theta_pi_1).clamp(min=0).mm(self.theta_pi_2).softmax(dim=1))

            return(pi_vals[action])


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


    def episode(self,show_plot=True,save_plot=False,N_steps=10**3):

        R_tot = 0

        if show_plot:
            self.showFig()

        self.agent.initEpisode()

        s = self.agent.getStateVec()
        a = self.epsGreedyAction(s)

        for i in range(N_steps):
            self.epsilon *= .99

            if i%11==0 and i>self.N_batch:
                self.updateFrozenQ()

            self.agent.iterate(a)
            r = self.agent.reward()
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

                Q_cur = self.forwardPassQ(states)[list(range(len(actions))),actions]
                Q_next = torch.max(self.forwardPassQFrozen(states_next),dim=1)[0]

                #TD0_error = (rewards + self.gamma*Q_next - Q_cur).pow(2).sum()#.clamp(max=1)
                TD0_error = F.smooth_l1_loss(Q_cur,(rewards + self.gamma*Q_next).detach())

                self.optimizer.zero_grad()
                TD0_error.backward()
                for param in self.policy_NN.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.optimizer.step()

            s = s_next
            a = a_next

            if show_plot:
                self.plotAll()
                self.fig.canvas.draw()

        print('puck-target dist: {:.2f}, R_tot/N_steps: {:.2f}'.format(self.agent.puckTargetDist(),R_tot/N_steps))
        return(R_tot)


    def plotAll(self):
        self.drawState()
        self.plotStateParams()
        #self.plotWeights()


    def drawState(self):
        self.agent.drawState(self.ax_state)


    def plotWeights(self):
        self.ax_wQ.clear()
        #print(self.w_Q.view(1,-1).numpy())
        #self.ax_wQ.plot(self.w_Q.view(1,-1).numpy().flatten(),label='w_Q weights')
        for i in range(self.w_Q.shape[0]):
            self.ax_wQ.plot(self.w_Q.detach().numpy()[i,:],label='w_Q '+str(i))
        self.ax_wQ.legend()

        self.ax_theta_pi.clear()
        #print(self.w_Q.view(1,-1).numpy())
        #self.ax_theta_pi.plot(self.theta_pi.view(1,-1).detach().numpy().flatten(),label='theta_pi weights')
        '''for i in range(self.theta_pi.shape[0]):
            self.ax_theta_pi.plot(self.theta_pi.detach().numpy()[i,:],label='theta_pi '+str(i))
        self.ax_theta_pi.legend()'''


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







#
