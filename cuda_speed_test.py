import torch
import numpy as np
import timeit


# torch Tensor on CPU
x = torch.rand(1, 64)
y = torch.rand(5000, 64)
tot_time = timeit.timeit('z=(x*y).sum(dim=1)',setup='import torch;x = torch.rand(1, 64);y = torch.rand(5000, 64)',number=10000)
print('cpu op took:',tot_time)

# torch Tensor on GPU
x, y = x.cuda(), y.cuda()
tot_time = timeit.timeit('z = (x*y).sum(dim=1)',setup='import torch;x = torch.rand(1, 64);y = torch.rand(5000, 64);x, y = x.cuda(), y.cuda()',number=10000)
print('gpu op took:',tot_time)

# numpy ndarray on CPU
x = np.random.random((1, 64))
y = np.random.random((5000, 64))
tot_time = timeit.timeit('z = (x*y).sum(axis=1)',setup='import numpy as np;x = np.random.random((1, 64));y = np.random.random((5000, 64))',number=10000)
print('numpy op took:',tot_time)
