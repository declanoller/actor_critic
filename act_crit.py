import AgentTools as at
from Agent import Agent
from PuckworldAgent import PuckworldAgent

'''

try:
    -also not using ER/frozen network, just using ER with the same network, see
    if it's noticeably worse

    -hot damn, I think DDQN might have solved it??? why?

    -try plain old DQN (no other network) with e-traces -- I think
    they could be really good here.


looks like:
-tanh lower var than relu
-L1 better than L2, but def don't wanna clamp grad
-80/10 is good
-not super clear about HLN... 200 performs best, but 100 is the worst. So it could be coincidence?


-do I want to be randomly initializing?

'''


ag = Agent(agent_class=PuckworldAgent,features='AC',N_steps=10**5,N_hidden_layer_nodes=50,reward='sparse',loss_method='L2')

ag.episode(save_plot=True,show_plot=False)


exit(0)


at.varyParam(N_steps=5*10**4,reward=['shaped','sparse'],N_runs=1,N_batch=80,target_update=200,N_hidden_layer_nodes=50,epsilon=.9,epsilon_decay=.999,double_DQN=True)

exit(0)





at.gifFromModel('/home/declan/Documents/code/ActorCritic1/vary_target_update__07-12-38/model_gamma=1.00_alpha=0.10_epsilon=0.90_epsilon_decay=1.00_N_steps=300000_N_batch=80_N_hidden_layer_nodes=50_target_update=500_double_DQN=True_exp_buf_len=10000_NL_fn=tanh_loss_method=L2_clamp_grad=False_09-20-41.model',10**3)

exit(0)




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
