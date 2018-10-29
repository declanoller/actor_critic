import AgentTools as at
from Agent import Agent
from PuckworldAgent import PuckworldAgent
from PuckworldAgent_radial import PuckworldAgent_radial

'''

-AC is working...intermittently, only with value function advantage.
-seems like relu might work a little better for it?
-it seems like in one pytorch example, they're using the same network
to produce both the pi and V values... isn't that bad?
-gotta fix smoothL1 problem...

try:
    -also not using ER/frozen network, just using ER with the same network, see
    if it's noticeably worse

    -hot damn, I think DDQN might have solved it??? why?

    -try plain old DQN (no other network) with e-traces -- I think
    they could be really good here.

    -ooooh shit... I think in AC I may have been multiplying by the TD error for Q,
    when in fact you're just supposed to mult. by Q_cur?? You only do TD error if it's
    using V.


looks like:
-tanh lower var than relu
-L1 better than L2, but def don't wanna clamp grad
-80/10 is good
-not super clear about HLN... 200 performs best, but 100 is the worst. So it could be coincidence?


-do I want to be randomly initializing?

'''




at.varyParam(N_steps=10**5, N_runs=3, double_DQN='True', epsilon=0.9, epsilon_decay=0.9995, target_update=20, N_batch=80, N_hidden_layer_nodes=[5,10,15,20,40],features='DQN',NL_fn='tanh',loss_method='L2')

exit(0)

ag = Agent(agent_class=PuckworldAgent_radial,features='AC',double_DQN=True,N_steps=10**4,N_batch=80,target_update=100,N_hidden_layer_nodes=50,reward='sparse',loss_method='L2',NL_fn='tanh')

ag.episode(save_plot=True,show_plot=False)


at.gifFromModel('/home/declan/Documents/code/ActorCritic1/vary_target_update__07-12-38/model_gamma=1.00_alpha=0.10_epsilon=0.90_epsilon_decay=1.00_N_steps=300000_N_batch=80_N_hidden_layer_nodes=50_target_update=500_double_DQN=True_exp_buf_len=10000_NL_fn=tanh_loss_method=L2_clamp_grad=False_09-20-41.model', 100)

exit(0)




#at.varyParam(N_steps=10**5, N_runs=5, double_DQN=['False', 'True'], epsilon=1.0, epsilon_decay=0.1, target_update=20, N_batch=80, N_hidden_layer_nodes=50,features='DQN',NL_fn='tanh',loss_method='L2')





at.varyParam(N_steps=3*10**5,N_runs=3, beta=0.1, N_hidden_layer_nodes=50,features=['DQN','AC'],double_DQN=True,N_batch=80,target_update=400,NL_fn='relu',loss_method='L2',epsilon=.9,epsilon_decay=.9995)


at.varyParam(N_steps=3*10**5,N_runs=3, beta=0.1, N_hidden_layer_nodes=50,features='AC',NL_fn=['relu','tanh'],loss_method='L2')





at.varyParam(N_steps=3*10**5,N_runs=3, beta=[10**0, .5, 10**-1, .05, 10**-2, 10**-3, 10**-4],N_hidden_layer_nodes=50,features='AC',NL_fn='relu',loss_method='L2')



exit(0)






path = '/home/declan/Documents/code/ActorCritic1/misc_runs/'

at.plotRewardCurves([
path + 'reward_gamma=1.00_alpha=0.10_beta=0.10_epsilon=0.80_epsilon_decay=0.99_N_steps=100000_N_batch=20_N_hidden_layer_nodes=50_target_update=12_double_DQN=False_exp_buf_len=10000_NL_fn=relu_loss_method=L2_clamp_grad=False_01-06-42.txt',
path + 'reward_gamma=1.00_alpha=0.10_beta=0.10_epsilon=0.90_epsilon_decay=1.00_N_steps=100000_N_batch=80_N_hidden_layer_nodes=50_target_update=400_double_DQN=True_exp_buf_len=10000_NL_fn=tanh_loss_method=L2_clamp_grad=False_01-13-25.txt'
],['A2C','DQN'])


exit(0)


ag = Agent(agent_class=PuckworldAgent,features='DQN',N_steps=10**5,N_hidden_layer_nodes=50,epsilon=.9,epsilon_decay=.9995,reward='sparse',loss_method='L2',double_DQN=True,N_batch=80,target_update=400)

ag.episode(save_plot=True,show_plot=False)














#ag = Agent(agent_class=PuckworldAgent,N_steps=3*10**5,N_batch=80,target_update=20,N_hidden_layer_nodes=50,epsilon=0.1,epsilon_decay=1.0,NL_fn='tanh',clamp_grad=False,loss_method='L2',double_DQN=[False,True])

path = '/home/declan/Documents/code/ActorCritic1/misc_runs/'
path = '/home/declan/Documents/code/ActorCritic1/vary_target_update__07-12-38/'
model = 'model_gamma=1.00_alpha=0.10_epsilon=0.90_epsilon_decay=1.00_N_steps=300000_N_batch=80_N_hidden_layer_nodes=50_target_update=500_double_DQN=True_exp_buf_len=10000_NL_fn=tanh_loss_method=L2_clamp_grad=False_09-20-41'
ag.loadModelPlay(path+model+'.model',show_plot=True,save_plot=True)

exit(0)





at.varyParam(N_steps=5*10**4,N_runs=3,N_batch=80,target_update=20,N_hidden_layer_nodes=50,epsilon=.9,epsilon_decay=.9995,NL_fn=['tanh','relu'],clamp_grad=False,loss_method='L2',double_DQN=True)

at.varyParam(N_steps=5*10**4,N_runs=3,N_batch=80,target_update=20,N_hidden_layer_nodes=50,epsilon=.9,epsilon_decay=.9995,NL_fn='tanh',clamp_grad=[False,True],loss_method='L2',double_DQN=True)

at.varyParam(N_steps=5*10**4,N_runs=3,N_batch=80,target_update=20,N_hidden_layer_nodes=50,epsilon=.9,epsilon_decay=.9995,NL_fn='tanh',clamp_grad=False,loss_method=['L2','smoothL1'],double_DQN=True)

at.varyParam(N_steps=5*10**4,N_runs=3,N_batch=80,target_update=20,N_hidden_layer_nodes=[20,50,100,200],epsilon=.9,epsilon_decay=.9995,NL_fn='tanh',clamp_grad=False,loss_method='L2',double_DQN=True)

at.varyParam(N_steps=5*10**4,N_runs=3,N_batch=80,target_update=[10,50,100,500,2000],N_hidden_layer_nodes=50,epsilon=.9,epsilon_decay=.9995,NL_fn='tanh',clamp_grad=False,loss_method='L2',double_DQN=True)

at.varyParam(N_steps=5*10**4,N_runs=3,N_batch=80,target_update=20,N_hidden_layer_nodes=50,epsilon=.9,epsilon_decay=.9995,NL_fn='tanh',clamp_grad=False,loss_method='L2',double_DQN=True,a=[.1,.5,2,5,10])


exit(0)




#
