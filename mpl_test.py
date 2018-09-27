import matplotlib.pyplot as plt
import numpy as np

import FileSystemTools as fst

fst.makeLabelDateDir('meow')






exit(0)


x = np.linspace(0,5,20)

y1 = x**2

y2 = np.sin(x)

f1 = plt.figure()
f2 = plt.figure()

ax1 = f1.add_subplot(111)
ax2 = f2.add_subplot(111)

ax1.plot(x,y1)
ax2.plot(x,y2)



plt.show()







#
