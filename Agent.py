import matplotlib.pyplot as plt
from random import randint,random,sample
import numpy as np
from math import atan,sin,cos,sqrt,ceil,floor,log
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import FileSystemTools as fst
from collections import namedtuple
from matplotlib.colors import LinearSegmentedColormap


Experience = namedtuple('exp_tup',('s','a','r','s_next'))


class DQN(nn.Module):

    def __init__(self,D_in,H,D_out,NL_fn=torch.tanh,softmax=False):
        super(DQN, self).__init__()

        self.lin1 = nn.Linear(D_in,H)
        self.lin2 = nn.Linear(H,D_out)
        self.NL_fn = NL_fn
        self.softmax = softmax

    def forward(self, x):
        x = self.lin1(x)
        #x = F.relu(x)
        #x = torch.tanh(x)
        x = self.NL_fn(x)
        x = self.lin2(x)
        if self.softmax:
            x = torch.softmax(x,dim=1)
        return(x)


class Agent:

    def __init__(self,**kwargs):

        self.agent_class = kwargs.get('agent_class',None)
        self.agent_class_name = self.agent_class.__name__
        self.agent = self.agent_class(**kwargs)

        self.params = {}
        self.params['gamma'] = kwargs.get('gamma',1.0)
        self.params['alpha'] = kwargs.get('alpha',10**-1)
        self.params['beta'] = kwargs.get('beta',None)
        self.params['epsilon'] = kwargs.get('epsilon',0.8)
        self.params['epsilon_decay'] = kwargs.get('epsilon_decay',0.99)

        self.params['N_steps'] = kwargs.get('N_steps',10**3)
        self.params['N_batch'] = int(kwargs.get('N_batch',20))
        self.params['N_hidden_layer_nodes'] = int(kwargs.get('N_hidden_layer_nodes',100))
        self.params['target_update'] = int(kwargs.get('target_update',12))
        self.params['double_DQN'] = kwargs.get('double_DQN',False)
        self.params['exp_buf_len'] = int(kwargs.get('exp_buf_len',10000))

        self.features = kwargs.get('features','DQN')
        self.params['NL_fn'] = kwargs.get('NL_fn','tanh')
        self.params['loss_method'] = kwargs.get('loss_method','smoothL1')
        self.params['clamp_grad'] = kwargs.get('clamp_grad',False)

        self.dir = kwargs.get('dir','misc_runs')
        self.date_time = kwargs.get('date_time',fst.getDateString())
        self.base_fname = fst.paramDictToFnameStr(self.params) + '_' + self.date_time
        self.img_fname = fst.combineDirAndFile(self.dir, self.base_fname + '.png')
        self.log_fname = fst.combineDirAndFile(self.dir, 'log_' + self.base_fname + '.txt')
        self.model_fname = fst.combineDirAndFile(self.dir, 'model_' + self.base_fname + '.model')


        self.R_tot_hist = []


        self.initLearningParams()

        self.createFigure()

        fst.writeDictToFile(self.params,self.log_fname)


    def initLearningParams(self):


        self.exp_pos = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('\nusing device:',self.device)

        self.dtype = torch.float32
        torch.set_default_dtype(self.dtype)

        if str(self.device) == 'cpu':
            torch.set_default_tensor_type(torch.FloatTensor)
        if str(self.device) == 'cuda':
            torch.set_default_tensor_type(torch.cuda.FloatTensor)


        NL_fn_dict = {'relu':F.relu, 'tanh':torch.tanh, 'sigmoid':F.sigmoid}



        if self.features == 'linear':
            self.fv_shape = self.agent.getFeatureVec(self.agent.getStateVec(),0).shape
            self.w_Q = torch.zeros(self.fv_shape, device=self.device, dtype=self.dtype, requires_grad=False)
            self.theta_pi = torch.zeros(self.fv_shape, device=self.device, dtype=self.dtype, requires_grad=True)


        if self.features == 'DQN':
            #This is using a "target" Q network and a "policy" Q network, with
            #experience replay.

            D_in, H, D_out = self.agent.N_state_terms, self.params['N_hidden_layer_nodes'], self.agent.N_actions

            NL_fn = NL_fn_dict[self.params['NL_fn']]

            self.policy_NN = DQN(D_in,H,D_out,NL_fn=NL_fn)
            self.target_NN = DQN(D_in,H,D_out,NL_fn=NL_fn)

            if str(self.device) == 'cuda':
                self.policy_NN.cuda()
                self.target_NN.cuda()

            self.target_NN.load_state_dict(self.policy_NN.state_dict())
            self.target_NN.eval()
            self.optimizer = optim.RMSprop(self.policy_NN.parameters())
            self.samples_Q = []

            self.episode = self.DQNepisode



        if self.features == 'AC':

            D_in, H, D_out = self.agent.N_state_terms, self.params['N_hidden_layer_nodes'], self.agent.N_actions

            NL_fn = NL_fn_dict[self.params['NL_fn']]

            #The "actor" is the policy network, the "critic" is the Q network.
            self.actor_NN = DQN(D_in,H,D_out,NL_fn=NL_fn,softmax=True)
            self.critic_NN = DQN(D_in,H,D_out,NL_fn=NL_fn)
            self.value_NN = DQN(D_in,H,1,NL_fn=NL_fn)
            #self.target_critic_NN = DQN(D_in,H,D_out,NL_fn=NL_fn)

            if str(self.device) == 'cuda':
                self.actor_NN.cuda()
                self.critic_NN.cuda()
                #self.target_critic_NN.cuda()

            #self.target_critic_NN.load_state_dict(self.critic_NN.state_dict())
            #self.target_critic_NN.eval()
            self.actor_optimizer = optim.RMSprop(self.actor_NN.parameters())
            self.critic_optimizer = optim.RMSprop(self.critic_NN.parameters())
            self.value_optimizer = optim.RMSprop(self.value_NN.parameters())
            self.samples_Q = []

            self.episode = self.ACepisode




    def updateFrozenQ(self):
        if self.features == 'DQN':
            self.target_NN.load_state_dict(self.policy_NN.state_dict())
        if self.features == 'AC':
            self.target_critic_NN.load_state_dict(self.critic_NN.state_dict())


    def updateEpsilon(self):
        self.params['epsilon'] *= self.params['epsilon_decay']





    def softmaxAction(self,state_vec):

        pi_vals = torch.squeeze(self.actor_NN(torch.unsqueeze(state_vec,dim=0)))
        action = np.random.choice(list(range(len(pi_vals))),p=pi_vals.detach().numpy())
        return(action)

        '''probs = torch.squeeze(self.actor_NN(torch.unsqueeze(s, dim=0)))
        print(probs)
        m = Categorical(probs)
        a = m.sample()'''

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


    def addExperience(self,experience):
        if len(self.samples_Q) < self.params['exp_buf_len']:
            self.samples_Q.append(None)
        self.samples_Q[self.exp_pos] = experience
        self.exp_pos = (self.exp_pos + 1) % self.params['exp_buf_len']




    def DQNepisode(self,show_plot=True,save_plot=False):

        self.R_tot = 0

        if show_plot:
            self.showFig()

        self.initEpisode()

        s = self.getStateVec()

        for i in range(self.params['N_steps']):

            if i%int(self.params['N_steps']/10) == 0:
                print('iteration ',i)

            self.updateEpsilon()

            if i%self.params['target_update']==0 and i>self.params['N_batch']:
                self.updateFrozenQ()

            a = self.epsGreedyAction(s)
            r, s_next = self.iterate(a)

            self.R_tot += r.item()
            self.R_tot_hist.append(self.R_tot/(i+1))

            a_next = self.epsGreedyAction(s_next)

            self.addExperience(Experience(s,a,r,s_next))

            if len(self.samples_Q)>=2*self.params['N_batch']:

                #Get random batch
                batch_Q_samples = sample(self.samples_Q,self.params['N_batch'])
                experiences = Experience(*zip(*batch_Q_samples))

                states = torch.stack(experiences.s)
                actions = experiences.a
                rewards = torch.stack(experiences.r)
                states_next = torch.stack(experiences.s_next)

                Q_cur = self.forwardPassQ(states)[list(range(len(actions))),actions]

                if self.params['double_DQN']:
                    actions_next = torch.argmax(self.forwardPassQ(states_next),dim=1)
                    Q_next = self.forwardPassQFrozen(states_next)[list(range(len(actions_next))),actions_next]
                else:
                    Q_next = torch.max(self.forwardPassQFrozen(states_next),dim=1)[0]


                if self.params['loss_method'] == 'smoothL1':
                    TD0_error = F.smooth_l1_loss(Q_cur,(rewards + self.params['gamma']*Q_next).detach())
                if self.params['loss_method'] == 'L2':
                    TD0_error = (rewards + self.params['gamma']*Q_next - Q_cur).pow(2).sum()


                self.optimizer.zero_grad()
                TD0_error.backward()
                if self.params['clamp_grad']:
                    for param in self.policy_NN.parameters():
                        param.grad.data.clamp_(-1, 1)
                self.optimizer.step()

            s = s_next

            if show_plot:
                self.plotAll()
                self.fig.canvas.draw()

        if save_plot:
            self.plotAll()
            plt.savefig(self.img_fname)

        plt.close('all')

        self.saveModel()

        print('self.R_tot/N_steps: {:.2f}'.format(self.R_tot/self.params['N_steps']))
        return(self.R_tot/self.params['N_steps'])



    def ACepisode(self,show_plot=True,save_plot=False):

        self.R_tot = 0

        if show_plot:
            self.showFig()

        self.initEpisode()

        s = self.getStateVec()
        #a = torch.argmax(torch.squeeze(self.actor_NN(torch.unsqueeze(s,dim=0))))
        a = self.softmaxAction(s)
        '''probs = torch.squeeze(self.actor_NN(torch.unsqueeze(s, dim=0)))
        print(probs)
        m = Categorical(probs)
        a = m.sample()
        print(a)'''

        for i in range(self.params['N_steps']):

            r, s_next = self.iterate(a)

            self.R_tot += r.item()
            self.R_tot_hist.append(self.R_tot/(i+1))

            if i%int(self.params['N_steps']/10) == 0:
                print('iteration {}, R_tot/i = {:.3f}'.format(i,self.R_tot_hist[-1]))



            #a_next = torch.argmax(torch.squeeze(self.actor_NN(torch.unsqueeze(s_next,dim=0))))
            a_next = self.softmaxAction(s_next)
            '''probs2 = torch.squeeze(self.actor_NN(torch.unsqueeze(s_next, dim=0)))
            m2 = Categorical(probs2)
            a_next = m2.sample()'''

            #V_cur = torch.squeeze(self.value_NN(torch.unsqueeze(s, dim=0)))[a]
            #V_next = torch.squeeze(self.value_NN(torch.unsqueeze(s_next, dim=0)))[a_next]
            V_cur = self.value_NN(torch.unsqueeze(s, dim=0))
            V_next = self.value_NN(torch.unsqueeze(s_next, dim=0))

            Q_cur = torch.squeeze(self.critic_NN(torch.unsqueeze(s, dim=0)))[a]
            Q_next = torch.squeeze(self.critic_NN(torch.unsqueeze(s_next, dim=0)))[a_next]



            pi = torch.squeeze(self.actor_NN(torch.unsqueeze(s, dim=0)))[a]
            #print('pi: ',torch.squeeze(self.actor_NN(torch.unsqueeze(s, dim=0))))

            if self.params['loss_method'] == 'smoothL1':
                #TD0_error = F.smooth_l1_loss(Q_cur,(r + self.params['gamma']*Q_next).detach())
                #There seems to be a bug in this, where it says that the "derivative for target (the 2nd arg) isn't implemented".
                TD0_error = F.smooth_l1_loss(Q_cur, r + self.params['gamma']*Q_next)
            if self.params['loss_method'] == 'L2':
                #TD0_error = (r + self.params['gamma']*Q_next - Q_cur).pow(2).sum()
                TD0_error = (r + self.params['gamma']*V_next - V_cur).pow(2).sum()


            #J = -m.log_prob(a) * (Q_cur.detach())
            #J = -Q_cur.detach()*pi
            #J = -(Q_cur.detach())*torch.log(pi)
            J = -(r + self.params['gamma']*V_next.detach() - V_cur.detach())*torch.log(pi)
            #J = -(TD0_error.detach())*pi

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            self.value_optimizer.zero_grad()

            #TD0_error.backward(retain_graph=True)
            #J.backward()

            J.backward(retain_graph=True)
            TD0_error.backward()

            if self.params['clamp_grad']:
                for param in self.actor_NN.parameters():
                    param.grad.data.clamp_(-1, 1)
                for param in self.critic_NN.parameters():
                    param.grad.data.clamp_(-1, 1)

            self.actor_optimizer.step()
            self.critic_optimizer.step()
            self.value_optimizer.step()

            s = s_next
            a = a_next
            #m1 = m2

            if show_plot:
                self.plotAll()
                self.fig.canvas.draw()


        if save_plot:
            self.plotAll()
            plt.savefig(self.img_fname)

        plt.close('all')

        print('self.R_tot/N_steps: {:.2f}'.format(self.R_tot/self.params['N_steps']))
        return(self.R_tot/self.params['N_steps'])


    def ACERepisode(self,show_plot=True,save_plot=False):

        self.R_tot = 0

        if show_plot:
            self.showFig()

        self.agent.initEpisode()

        s = self.getStateVec()
        a = torch.argmax(torch.squeeze(self.actor_NN(torch.unsqueeze(torch.tensor(s,device=self.device,dtype=self.dtype),dim=0))))

        for i in range(self.params['N_steps']):
            self.updateEpsilon()

            if i%self.params['target_update']==0 and i>self.params['N_batch']:
                self.updateFrozenQ()

            self.agent.iterate(a)
            r = self.getReward()
            self.R_tot += r.item()

            s_next = self.getStateVec()
            a_next = torch.argmax(torch.squeeze(self.actor_NN(torch.unsqueeze(torch.tensor(s_next,device=self.device,dtype=self.dtype),dim=0))))

            self.addExperience(Experience(s,a,r,s_next))

            if len(self.samples_Q)>=2*self.params['N_batch']:

                #Get random batch
                batch_Q_samples = sample(self.samples_Q,self.params['N_batch'])
                experiences = Experience(*zip(*batch_Q_samples))

                states = torch.stack(experiences.s)
                actions = experiences.a
                rewards = torch.stack(experiences.r)
                states_next = torch.stack(experiences.s_next)

                Q_cur = (self.critic_NN(states)[list(range(len(actions))),actions])
                actions_next = torch.argmax(self.actor_NN(states_next),dim=1)
                #Q_next = (self.target_critic_NN(states_next)[list(range(len(actions_next))),actions_next])
                Q_next = (self.critic_NN(states_next)[list(range(len(actions_next))),actions_next])

                pi = (self.actor_NN(states)[list(range(len(actions))),actions])

                if self.params['loss_method'] == 'smoothL1':
                    TD0_error = F.smooth_l1_loss(Q_cur,(rewards + self.params['gamma']*Q_next).detach())
                if self.params['loss_method'] == 'L2':
                    TD0_error = (rewards + self.params['gamma']*Q_next - Q_cur).pow(2).sum()#.clamp(max=1)

                J = (Q_cur*torch.log(pi)).sum()

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
            plt.savefig(self.img_fname)

        plt.close('all')

        print('self.R_tot/N_steps: {:.2f}'.format(self.R_tot/self.params['N_steps']))
        return(self.R_tot/self.params['N_steps'])



    def iterate(self,a):
        r, s_next = self.agent.iterate(a)
        return(torch.tensor(r,device=self.device,dtype=self.dtype),torch.tensor(s_next,device=self.device,dtype=self.dtype))


    def initEpisode(self):
        self.agent.initEpisode()


    def resetStateValues(self):
        self.agent.resetStateValues()


    def getStateVec(self):
        return(torch.tensor(self.agent.getStateVec(),device=self.device,dtype=self.dtype))


    def getReward(self):
        return(torch.tensor(self.agent.reward(),device=self.device,dtype=self.dtype))






    def saveModel(self):
        if self.features == 'DQN':
            #print(self.forwardPassQ(torch.tensor([.2,.2,0,0,.8,.8],device=self.device,dtype=self.dtype)))
            torch.save(self.policy_NN.state_dict(), self.model_fname)


    def loadModelPlay(self, model_fname, show_plot=True, save_plot=False, make_gif=False, N_steps=10**3):

        self.policy_NN.load_state_dict(torch.load(model_fname))

        self.params['epsilon'] = 0
        self.R_tot = 0

        if show_plot:
            self.showFig()

        self.agent.initEpisode()


        for i in range(N_steps):

            if i%int(N_steps/10) == 0:
                print('iteration ',i)


            self.updateEpsilon()


            s = self.getStateVec()
            a = self.epsGreedyAction(s)

            self.agent.iterate(a)
            r = self.getReward()
            self.R_tot += r.item()
            self.R_tot_hist.append(self.R_tot/(i+1))

            if r.item() > 0:
                self.agent.resetTarget()

            if show_plot:
                self.plotAll()
                self.fig.canvas.draw()

            if make_gif:
                self.plotAll()
                plt.savefig(fst.combineDirAndFile(self.dir, str(i) + '.png'))


        if save_plot and not make_gif:
            self.plotAll()
            plt.savefig(self.img_fname)

        plt.close('all')

        print('self.R_tot/N_steps: {:.2f}'.format(self.R_tot/N_steps))
        return(self.R_tot/N_steps)





    def plotAll(self):
        self.drawState()
        self.plotStateParams()
        self.plotRtot()
        if self.features == 'DQN':
            self.plotWeights()
        #self.plotWeights()


    def drawState(self):
        self.agent.drawState(self.ax_state)


    def plotWeights(self):

        sv = self.getStateVec().detach().numpy().tolist()
        target_pos = sv[4:]

        if (target_pos != self.last_target_pos) or (self.last_target_pos is None):

            self.ax_wQ.clear()
            self.last_target_pos = target_pos
            N_disc = 50
            lims = self.agent.xlims
            pos = np.array([[[x,y] for y in np.linspace(lims[0],lims[1],N_disc)] for x in np.linspace(lims[0],lims[1],N_disc)])
            v = np.expand_dims(np.full((N_disc,N_disc),0),axis=2)
            sv = self.getStateVec()
            xt = np.expand_dims(np.full((N_disc,N_disc),sv[4]),axis=2)
            yt = np.expand_dims(np.full((N_disc,N_disc),sv[5]),axis=2)

            states = np.concatenate((pos,v,v,xt,yt),axis=2)

            states = torch.tensor(states,dtype=self.dtype)
            output = self.forwardPassQ(states)

            best_actions = (torch.argmax(output,dim=2)).detach().numpy()
            #best_actions = (torch.max(output,dim=2)[0]).detach().numpy()

            if self.col_bar is not None:
                self.col_bar.remove()

            col_plot = self.ax_wQ.matshow(best_actions.T,cmap=self.cm,origin='lower')
            #col_plot = self.ax_wQ.matshow(best_actions,cmap='Reds',origin='lower')
            self.ax_wQ.set_xlabel('x')
            self.ax_wQ.set_ylabel('y')
            self.col_bar = self.fig.colorbar(col_plot,ax=self.ax_wQ, ticks=[0,1,2,3], boundaries=np.arange(-.5,4.5,1))
            self.col_bar.ax.set_yticklabels(['U','D','L','R'])
            xt = sv[4]
            yt = sv[5]

            target = plt.Circle(((xt-lims[0])*N_disc,(yt-lims[0])*N_disc), 2.5*N_disc/20.0, color='black')
            self.ax_wQ.add_artist(target)





            self.ax_wQ2.clear()
            self.last_target_pos = target_pos
            N_disc = 40
            lims = self.agent.xlims
            pos = np.array([[[x,y] for y in np.linspace(lims[0],lims[1],N_disc)] for x in np.linspace(lims[0],lims[1],N_disc)])
            v = np.expand_dims(np.full((N_disc,N_disc),0),axis=2)
            sv = self.getStateVec()
            xt = np.expand_dims(np.full((N_disc,N_disc),sv[4]),axis=2)
            yt = np.expand_dims(np.full((N_disc,N_disc),sv[5]),axis=2)

            states = np.concatenate((pos,v,v,xt,yt),axis=2)

            states = torch.tensor(states,dtype=self.dtype)
            output = self.forwardPassQ(states)

            #best_actions = (torch.argmax(output,dim=2)).detach().numpy()
            max_Q = (torch.max(output,dim=2)[0]).detach().numpy()

            if self.col_bar2 is not None:
                self.col_bar2.remove()

            col_plot2 = self.ax_wQ2.matshow(max_Q.T,cmap='Reds',origin='lower')
            #col_plot2 = self.ax_wQ2.matshow(max_Q,cmap='Reds',origin='lower')
            self.ax_wQ2.set_xlabel('x')
            self.ax_wQ2.set_ylabel('y')
            self.col_bar2 = self.fig.colorbar(col_plot2,ax=self.ax_wQ2)
            xt = sv[4]
            yt = sv[5]

            target = plt.Circle(((xt-lims[0])*N_disc,(yt-lims[0])*N_disc), 2.5*N_disc/20.0, color='black')
            self.ax_wQ2.add_artist(target)


    def plotRtot(self):
        self.ax_R_tot.clear()
        self.ax_R_tot.plot(self.R_tot_hist[8:])


    def plotStateParams(self):
        self.agent.plotStateParams([self.ax_state_params1,self.ax_state_params2,self.ax_state_params3,self.ax_state_params4])


    def createFigure(self):

        self.fig, self.axes = plt.subplots(3,3,figsize=(12,8))
        self.ax_state = self.axes[0,0]
        self.ax_state_params1 = self.axes[0,1]
        self.ax_state_params2 = self.axes[0,2]
        self.ax_state_params3 = self.axes[1,2]
        self.ax_state_params4 = self.axes[1,1]

        self.ax_wQ = self.axes[2,1]
        self.ax_wQ2 = self.axes[2,0]
        self.ax_theta_pi = self.axes[1,1]

        #self.ax_loc_vals = self.axes[2,0]
        self.ax_R_tot = self.axes[2,2]
        self.col_bar = None
        self.col_bar2 = None

        self.cm = LinearSegmentedColormap.from_list('my_cm', ['tomato','dodgerblue','seagreen','orange'], N=4)

        self.last_target_pos = None
        self.plotAll()


    def showFig(self):
        plt.show(block=False)



#
