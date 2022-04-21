
import matplotlib
# %matplotlib
matplotlib.use('Qt5Agg')
# matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
import numpy as np
a=[]
b=[]
fig,ax=plt.subplots()
for i in range(100):
    a.append(np.random.uniform(-1,1))
    b.append(np.random.uniform(-1,1))
    ax.plot(a,b)
    plt.pause(0.5)
# plt.show(block=True)
print(a)
print(b)
