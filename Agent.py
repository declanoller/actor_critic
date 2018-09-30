import matplotlib.pyplot as plt
from random import randint,random,sample
import numpy as np
from math import atan,sin,cos,sqrt,ceil,floor,log
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import FileSystemTools as fst




class DQN(nn.Module):

    def __init__(self,D_in,H,D_out,NL_fn=torch.tanh):
        super(DQN, self).__init__()

        self.lin1 = nn.Linear(D_in,H)
        self.lin2 = nn.Linear(H,D_out)
        self.NL_fn = NL_fn

    def forward(self, x):
        x = self.lin1(x)
        #x = F.relu(x)
        #x = torch.tanh(x)
        x = self.NL_fn(x)
        x = self.lin2(x)
        return(x)


class Agent:

    def __init__(self,**kwargs):

        self.agent_class = kwargs.get('agent_class',None)
        self.agent_class_name = self.agent_class.__name__
        self.agent = self.agent_class()

        self.params = {}
        self.params['gamma'] = kwargs.get('gamma',1.0)
        self.params['alpha'] = kwargs.get('alpha',10**-1)
        self.params['beta'] = kwargs.get('beta',None)
        self.params['epsilon'] = kwargs.get('epsilon',0.8)

        self.params['N_steps'] = kwargs.get('N_steps',10**3)
        self.params['N_batch'] = kwargs.get('N_batch',20)
        self.params['N_hidden_layer_nodes'] = kwargs.get('N_hidden_layer_nodes',100)

        self.features = kwargs.get('features','DQN')
        self.params['NL_fn'] = kwargs.get('NL_fn','tanh')
        self.params['loss_method'] = kwargs.get('loss','smoothL1')
        self.params['clamp_grad'] = kwargs.get('clamp_grad',True)

        self.dir = kwargs.get('dir','.')
        self.date_time = kwargs.get('date_time',fst.getDateString())
        self.base_fname = fst.paramDictToFnameStr(self.params) + '_' + self.date_time + '.png'
        self.fname = fst.combineDirAndFile(self.dir,self.base_fname)


        self.initLearningParams()

        self.createFigure()


    def initLearningParams(self):

        self.dtype = torch.float64
        #self.device = torch.device("cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('\nusing device:',self.device)

        torch.set_default_dtype(self.dtype)
        if self.device == 'cpu':
            torch.set_default_tensor_type(torch.DoubleTensor)
        if self.device == 'cuda':
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)

        print('current cuda device:',torch.cuda.current_device())
        #torch.cuda.device()
        #exit(0)
        if self.features == 'linear':

            self.fv_shape = self.agent.getFeatureVec(self.agent.getStateVec(),0).shape

            #init_smallnum = 0.01
            self.w_Q = torch.zeros(self.fv_shape, device=self.device, dtype=self.dtype, requires_grad=False)
            self.theta_pi = torch.zeros(self.fv_shape, device=self.device, dtype=self.dtype, requires_grad=True)
            '''self.w_Q = torch.randn(self.fv_shape, device=self.device, dtype=self.dtype, requires_grad=False)
            self.theta_pi = torch.randn(self.fv_shape, device=self.device, dtype=self.dtype, requires_grad=True)'''

        if self.features == 'DQN':

            D_in, H, D_out = self.agent.N_state_terms, self.params['N_hidden_layer_nodes'], self.agent.N_actions
            if self.params['NL_fn'] == 'tanh':
                NL_fn = torch.tanh
            if self.params['NL_fn'] == 'relu':
                NL_fn = F.relu
            if self.params['NL_fn'] == 'sigmoid':
                NL_fn = F.sigmoid

            self.policy_NN = DQN(D_in,H,D_out,NL_fn=NL_fn)
            self.target_NN = DQN(D_in,H,D_out,NL_fn=NL_fn)
            self.target_NN.load_state_dict(self.policy_NN.state_dict())
            self.target_NN.eval()
            self.optimizer = optim.RMSprop(self.policy_NN.parameters())
            self.samples_Q = []

            self.episode = self.DQNERepisode

        if self.features == 'AC':

            D_in, H, D_out = self.agent.N_state_terms, self.params['N_hidden_layer_nodes'], self.agent.N_actions
            if self.params['NL_fn'] == 'tanh':
                NL_fn = torch.tanh
            if self.params['NL_fn'] == 'relu':
                NL_fn = F.relu
            if self.params['NL_fn'] == 'sigmoid':
                NL_fn = F.sigmoid

            #The "actor" is the policy network, the "critic" is the Q network.
            self.actor_NN = DQN(D_in,H,D_out,NL_fn=NL_fn).to(self.device)
            self.critic_NN = DQN(D_in,H,D_out,NL_fn=NL_fn).to(self.device)
            self.target_critic_NN = DQN(D_in,H,D_out,NL_fn=NL_fn).to(self.device)
            self.target_critic_NN.load_state_dict(self.critic_NN.state_dict())
            self.target_critic_NN.eval()
            self.actor_optimizer = optim.RMSprop(self.actor_NN.parameters())
            self.critic_optimizer = optim.RMSprop(self.critic_NN.parameters())
            self.samples_Q = []

            self.episode = self.ACERepisode


    def updateFrozenQ(self):
        if self.features == 'DQN':
            self.target_NN.load_state_dict(self.policy_NN.state_dict())
        if self.features == 'AC':
            self.target_critic_NN.load_state_dict(self.critic_NN.state_dict())


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
        if random()>self.params['epsilon']:
            return(self.greedyAction(state_vec))
        else:
            return(self.getRandomAction())


    def getRandomAction(self):
        return(randint(0,self.agent.N_actions-1))


    def DQNERepisode(self,show_plot=True,save_plot=False):

        R_tot = 0

        if show_plot:
            self.showFig()

        self.agent.initEpisode()

        s = self.agent.getStateVec()
        a = self.epsGreedyAction(s)

        for i in range(self.params['N_steps']):
            self.params['epsilon'] *= .99

            if i%11==0 and i>self.params['N_batch']:
                self.updateFrozenQ()

            self.agent.iterate(a)
            r = self.agent.reward()
            R_tot += r

            s_next = self.agent.getStateVec()
            a_next = self.epsGreedyAction(s_next)

            experience = (s,a,r,s_next)
            self.samples_Q.append(experience)

            if len(self.samples_Q)>=2*self.params['N_batch']:

                #Get random batch
                batch_Q_samples = sample(self.samples_Q,self.params['N_batch'])
                states = torch.Tensor(np.array([samp[0] for samp in batch_Q_samples]))
                actions = [samp[1] for samp in batch_Q_samples]
                rewards = torch.Tensor([samp[2] for samp in batch_Q_samples])
                states_next = torch.Tensor([samp[3] for samp in batch_Q_samples])

                Q_cur = self.forwardPassQ(states)[list(range(len(actions))),actions]
                Q_next = torch.max(self.forwardPassQFrozen(states_next),dim=1)[0]

                if self.params['loss_method'] == 'smoothL1':
                    TD0_error = F.smooth_l1_loss(Q_cur,(rewards + self.params['gamma']*Q_next).detach())
                if self.params['loss_method'] == 'L2':
                    TD0_error = (rewards + self.params['gamma']*Q_next - Q_cur).pow(2).sum()#.clamp(max=1)


                self.optimizer.zero_grad()
                TD0_error.backward()
                if self.params['clamp_grad']:
                    for param in self.policy_NN.parameters():
                        param.grad.data.clamp_(-1, 1)
                self.optimizer.step()

            s = s_next
            a = a_next

            if show_plot:
                self.plotAll()
                self.fig.canvas.draw()

        if save_plot:
            self.plotAll()
            plt.savefig(self.fname)

        plt.close('all')

        print('puck-target dist: {:.2f}, R_tot/N_steps: {:.2f}'.format(self.agent.puckTargetDist(),R_tot/self.params['N_steps']))
        return(R_tot/self.params['N_steps'])



    def ACepisode(self,show_plot=True,save_plot=False):

        R_tot = 0

        if show_plot:
            self.showFig()

        self.agent.initEpisode()

        s = self.agent.getStateVec()
        a = torch.argmax(torch.squeeze(self.actor_NN(torch.unsqueeze(torch.Tensor(s,device=self.device),dim=0))))

        for i in range(self.params['N_steps']):
            self.params['epsilon'] *= .99

            self.agent.iterate(a)
            r = self.agent.reward()
            R_tot += r

            s_next = self.agent.getStateVec()
            a_next = torch.argmax(torch.squeeze(self.actor_NN(torch.unsqueeze(torch.Tensor(s_next,device=self.device),dim=0))))

            Q_cur = torch.squeeze(self.critic_NN(torch.unsqueeze(torch.Tensor(s,device=self.device),dim=0)))[a]
            Q_next = torch.squeeze(self.critic_NN(torch.unsqueeze(torch.Tensor(s_next,device=self.device),dim=0)))[a_next]

            pi = torch.squeeze(self.actor_NN(torch.unsqueeze(torch.Tensor(s,device=self.device),dim=0)))[a]

            if self.params['loss_method'] == 'smoothL1':
                TD0_error = F.smooth_l1_loss(Q_cur,(r + self.params['gamma']*Q_next).detach())
            if self.params['loss_method'] == 'L2':
                TD0_error = (r + self.params['gamma']*Q_next - Q_cur).pow(2).sum()#.clamp(max=1)

            J = Q_cur*torch.log(pi)

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            TD0_error.backward(retain_graph=True)
            J.backward()

            if self.params['clamp_grad']:
                for param in self.actor_NN.parameters():
                    param.grad.data.clamp_(-1, 1)
                for param in self.critic_NN.parameters():
                    param.grad.data.clamp_(-1, 1)

            self.actor_optimizer.step()
            self.critic_optimizer.step()

            s = s_next
            a = a_next

            if show_plot:
                self.plotAll()
                self.fig.canvas.draw()


        if save_plot:
            self.plotAll()
            plt.savefig(self.fname)

        plt.close('all')

        print('puck-target dist: {:.2f}, R_tot/N_steps: {:.2f}'.format(self.agent.puckTargetDist(),R_tot/self.params['N_steps']))
        return(R_tot/self.params['N_steps'])



    def ACERepisode(self,show_plot=True,save_plot=False):

        R_tot = 0

        if show_plot:
            self.showFig()

        self.agent.initEpisode()

        s = self.agent.getStateVec()
        a = torch.argmax(torch.squeeze(self.actor_NN(torch.unsqueeze(torch.Tensor(s,device=self.device),dim=0)))).to(self.device)

        for i in range(self.params['N_steps']):
            self.params['epsilon'] *= .99

            if i%11==0 and i>self.params['N_batch']:
                self.updateFrozenQ()

            self.agent.iterate(a)
            r = self.agent.reward()
            R_tot += r

            s_next = self.agent.getStateVec()
            a_next = torch.argmax(torch.squeeze(self.actor_NN(torch.unsqueeze(torch.Tensor(s_next,device=self.device),dim=0)))).to(self.device)

            experience = (s,a,r,s_next)
            self.samples_Q.append(experience)

            if len(self.samples_Q)>=2*self.params['N_batch']:

                #Get random batch
                batch_Q_samples = sample(self.samples_Q,self.params['N_batch'])
                states = torch.Tensor(np.array([samp[0] for samp in batch_Q_samples]),device=self.device)
                actions = [samp[1] for samp in batch_Q_samples]
                rewards = torch.Tensor([samp[2] for samp in batch_Q_samples],device=self.device)
                states_next = torch.Tensor([samp[3] for samp in batch_Q_samples],device=self.device)

                Q_cur = (self.critic_NN(states)[list(range(len(actions))),actions]).to(self.device)
                actions_next = torch.argmax(self.actor_NN(states_next),dim=1).to(self.device)
                Q_next = (self.target_critic_NN(states_next)[list(range(len(actions_next))),actions_next]).to(self.device)

                #Q_cur = torch.squeeze(self.critic_NN(torch.unsqueeze(torch.Tensor(s),dim=0)))[a]
                #Q_next = torch.squeeze(self.critic_NN(torch.unsqueeze(torch.Tensor(s_next),dim=0)))[a_next]

                pi = (self.actor_NN(states)[list(range(len(actions))),actions]).to(self.device)

                if self.params['loss_method'] == 'smoothL1':
                    TD0_error = F.smooth_l1_loss(Q_cur,(rewards + self.params['gamma']*Q_next).detach()).to(self.device)
                if self.params['loss_method'] == 'L2':
                    TD0_error = (rewards + self.params['gamma']*Q_next - Q_cur).pow(2).sum().to(self.device)#.clamp(max=1)

                J = (Q_cur*torch.log(pi)).sum().to(self.device)

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                TD0_error.backward(retain_graph=True)
                J.backward()

                if self.params['clamp_grad']:
                    for param in self.actor_NN.parameters():
                        param.grad.data.clamp_(-1, 1)
                    for param in self.critic_NN.parameters():
                        param.grad.data.clamp_(-1, 1)

                self.actor_optimizer.step()
                self.critic_optimizer.step()

            s = s_next
            a = a_next

            if show_plot:
                self.plotAll()
                self.fig.canvas.draw()


        if save_plot:
            self.plotAll()
            plt.savefig(self.fname)

        plt.close('all')

        print('puck-target dist: {:.2f}, R_tot/N_steps: {:.2f}'.format(self.agent.puckTargetDist(),R_tot/self.params['N_steps']))
        return(R_tot/self.params['N_steps'])




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
        self.plotAll()

        self.ax_wQ = self.axes[2,1]
        self.ax_theta_pi = self.axes[1,1]

        self.ax_loc_vals = self.axes[2,0]


    def showFig(self):
        plt.show(block=False)







#
