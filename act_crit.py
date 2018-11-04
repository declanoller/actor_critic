import AgentTools as at
from Agent import Agent
from PuckworldAgent import PuckworldAgent
from PuckworldAgent_radial import PuckworldAgent_radial

'''

'''



at.varyParam(N_steps=10**5, N_runs=3, advantage=True, beta=[0, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 1], target_update=12, N_batch=20, N_hidden_layer_nodes=50, features='AC', NL_fn='relu')


exit(0)


at.varyParam(N_steps=10**5, N_runs=3, advantage=False, beta=[10**-2, 10**-1, .5, 1, 10], target_update=12, N_batch=20, N_hidden_layer_nodes=50, features='AC', NL_fn='relu')


exit(0)




at.plotRewardCurvesByVaryParam('/home/declan/Documents/code/ActorCritic1/vary_beta__22-32-30')

exit(0)

at.varyParam(agent_class=PuckworldAgent, N_steps=4*10**4, N_runs=1, epsilon=[0.9], epsilon_decay=0.9995, target_update=500, N_batch=80, N_hidden_layer_nodes=50, features='DQN')

exit(0)




at.gifFromModel('/home/declan/Documents/code/ActorCritic1/save_runs/DDQN/vary_target_update__07-12-38/model_gamma=1.00_alpha=0.10_epsilon=0.90_epsilon_decay=1.00_N_steps=300000_N_batch=80_N_hidden_layer_nodes=50_target_update=500_double_DQN=True_exp_buf_len=10000_NL_fn=tanh_loss_method=L2_clamp_grad=False_09-20-41.model', 199)

exit(0)




#
