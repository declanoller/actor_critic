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



at.varyParam(N_steps=4*10**4, N_runs=1, beta=[.05], epsilon=0.9, epsilon_decay=0.9995, target_update=12, N_batch=20, N_hidden_layer_nodes=50, features='AC', ACER=False, NL_fn='relu')

exit(0)



at.varyParam(agent_class=PuckworldAgent, N_steps=4*10**4, N_runs=1, epsilon=[0.9], epsilon_decay=0.9995, target_update=500, N_batch=80, N_hidden_layer_nodes=50, features='DQN')

exit(0)


at.plotRewardCurvesByVaryParam('/home/declan/Documents/code/ActorCritic1/vary_reward__02-02-08', offsets=[0, -0.16])

exit(0)


at.gifFromModel('/home/declan/Documents/code/ActorCritic1/save_runs/DDQN/vary_target_update__07-12-38/model_gamma=1.00_alpha=0.10_epsilon=0.90_epsilon_decay=1.00_N_steps=300000_N_batch=80_N_hidden_layer_nodes=50_target_update=500_double_DQN=True_exp_buf_len=10000_NL_fn=tanh_loss_method=L2_clamp_grad=False_09-20-41.model', 199)

exit(0)




#
