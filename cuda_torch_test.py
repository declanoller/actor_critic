import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import time

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


def getInputAndLabel():

    #[x1,x2,x3,x4,x5,x6] to...
    #[x1+x2,x3*x4,x5**2+x6,x1+x2*x3]
    x = np.random.randn(6)
    y = np.array([x[0]+x[1], x[2]*x[3], x[4]**2+x[5], x[1]+x[2]*x[3]])
    return(x,y)


dtype = torch.float
#device = torch.device('cpu')
device = torch.device('cuda')
print('\nusing device:',device)

NN1 = DQN(6,100,4)

if str(device) == 'cpu':
    pass
if str(device) == 'cuda':
    print('setting NN to cuda device')
    NN1 = NN1.to(device)

#exit(0)

optimizer = optim.RMSprop(NN1.parameters())

N_loops = 10**5
input_and_labels = [getInputAndLabel() for i in range(N_loops)]


###################################
print('all at once:')
t0 = time()

input_tensor = torch.tensor([i[0] for i in input_and_labels],dtype=dtype,device=device)
label_tensor = torch.tensor([i[1] for i in input_and_labels],dtype=dtype,device=device)

output_tensor = NN1(input_tensor)

loss = (output_tensor - label_tensor).pow(2).sum()
optimizer.zero_grad()
loss.backward()
optimizer.step()

print('\n\nrun time: {:.3f}'.format(time()-t0))


###################################
print('\n\nindividually:')
t0 = time()

for i in range(N_loops):


    input_tensor = torch.tensor(input_and_labels[i][0],dtype=dtype,device=device)
    label_tensor = torch.tensor(input_and_labels[i][1],dtype=dtype,device=device)

    output_tensor = NN1(input_tensor)

    loss = (output_tensor - label_tensor).pow(2).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



print('\n\nrun time: {:.3f}'.format(time()-t0))




#
