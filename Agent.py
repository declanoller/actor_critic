import matplotlib.pyplot as plt
from random import randint,random,sample
import numpy as np
from math import atan,sin,cos,sqrt,ceil,floor
from datetime import datetime
import torch


class Agent:

    def __init__(self,env,**kwargs):

        self.env = env
        self.agent_class = kwargs.get('agent_class',None)
        self.agent_class_name = self.agent_class.__name__
        self.agent = self.agent_class()

        self.time_step = kwargs.get('dt',10**-1)
        self.tot_time = kwargs.get('ep_time',100)
        self.steps = int(self.tot_time/self.time_step)

        self.gamma = kwargs.get('gamma',1.0)
        self.alpha = kwargs.get('alpha',0.005)
        self.beta = kwargs.get('beta',0.1)

        self.features = kwargs.get('features','linear')

        self.initLearningParams()


    def initLearningParams(self):


        self.dtype = torch.float64
        self.device = torch.device("cpu")

        torch.set_default_dtype(self.dtype)
        torch.set_default_tensor_type(torch.FloatTensor)


        if self.features == 'linear':

            self.fv_shape = self.agent.getFeatureVec(self.agent.getStateVec(),0).shape

            #init_smallnum = 0.01
            self.w_Q = torch.zeros(self.fv_shape, device=self.device, dtype=self.dtype, requires_grad=False)
            self.theta_pi = torch.zeros(self.fv_shape, device=self.device, dtype=self.dtype, requires_grad=True)
            '''self.w_Q = torch.randn(self.fv_shape, device=self.device, dtype=self.dtype, requires_grad=False)
            self.theta_pi = torch.randn(self.fv_shape, device=self.device, dtype=self.dtype, requires_grad=True)'''






    def resetStateValues(self):
        self.agent.resetStateValues()


    def softmaxActionAndVal(self,state_vec):

        #state_vec = self.agent.getStateVec()
        #returns the softmax action, as well as the value of pi for (s,a).
        exp_lin_combos = [torch.exp(torch.sum(torch.tensor(self.agent.getFeatureVec(state_vec,a),dtype=self.dtype)*self.theta_pi)) for a in range(self.agent.N_actions)]

        norm = sum(exp_lin_combos)
        p_actions = [t/norm for t in exp_lin_combos]
        action = np.random.choice(list(range(self.agent.N_actions)),p=p_actions)
        return(action,p_actions[action])


    def episode(self,show_plot=True,save_plot=False):

        self.agent.initEpisode()

        s = self.agent.getStateVec()
        a, pi_s_a = self.softmaxActionAndVal(s)
        self.agent.iterate(a)
        #Now it has pos_next and v_next
        self.agent.updateStateValues()

        for i in range(self.steps):
            #print('\ni: ',i)

            r = self.agent.reward()
            s_next = self.agent.getStateVec()
            a_next, _ = self.softmaxActionAndVal(s_next)

            Q_cur = torch.sum(torch.tensor(self.agent.getFeatureVec(s,a),dtype=self.dtype)*self.w_Q)
            Q_next = torch.sum(torch.tensor(self.agent.getFeatureVec(s_next,a_next),dtype=self.dtype)*self.w_Q)

            TD0_error = r + self.gamma*Q_next - Q_cur

            _, pi = self.softmaxActionAndVal(s)
            score = torch.log(pi)

            score.backward()

            with torch.no_grad():
                self.theta_pi += self.alpha*Q_cur*self.theta_pi.grad
                #self.theta_pi = self.theta_pi + self.alpha*self.theta_pi.grad*Q_cur
                self.theta_pi.grad.zero_()

            self.w_Q += self.beta*TD0_error*torch.tensor(self.agent.getFeatureVec(s,a),dtype=self.dtype)

            s = s_next
            a = a_next

            self.agent.iterate(a)
            self.agent.updateStateValues()

            if show_plot:
                self.plotAll()
                self.fig.canvas.draw()





    def plotAll(self):
        self.drawState()
        self.plotStateParams()
        self.plotWeights()


    def drawState(self):
        self.agent.drawState(self.ax_state)


    def plotWeights(self):
        self.ax_wQ.clear()
        #print(self.w_Q.view(1,-1).numpy())
        self.ax_wQ.plot(self.w_Q.view(1,-1).numpy().flatten(),label='w_Q weights')
        self.ax_wQ.legend()

        self.ax_theta_pi.clear()
        #print(self.w_Q.view(1,-1).numpy())
        self.ax_theta_pi.plot(self.theta_pi.view(1,-1).detach().numpy().flatten(),label='theta_pi weights')
        self.ax_theta_pi.legend()


    def plotStateParams(self):
        self.agent.plotStateParams([self.ax_state_params1,self.ax_state_params2,self.ax_state_params3])


    def createFigure(self):

        self.fig, self.axes = plt.subplots(2,3,figsize=(12,6))
        self.ax_state = self.axes[0,0]
        self.ax_state_params1 = self.axes[0,1]
        self.ax_state_params2 = self.axes[0,2]
        self.ax_state_params3 = self.axes[1,2]
        #self.plotAll()

        self.ax_wQ = self.axes[1,0]
        self.ax_theta_pi = self.axes[1,1]


    def showFig(self):

        plt.show(block=False)







#
