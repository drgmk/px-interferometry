import numpy as np
import scipy.special
import matplotlib.pyplot as plt

def vis(p, u):
    '''V for uniform source.'''
    return 2 * p[0] * scipy.special.jv(1, np.pi*u*p[1]) / (np.pi*u*p[1])

# range of b/lambda
u = np.linspace(1,1e4,100)

# 2mm source at 7m in radians
theta = 2e-3/7

fig, ax = plt.subplots()

ax.plot(u, np.abs(vis([1,theta], u)))

ax.set_ylabel('absolute visibility')
ax.set_xlabel('baseline / $\lambda$')

fig.savefig('uniform_vis.png')
