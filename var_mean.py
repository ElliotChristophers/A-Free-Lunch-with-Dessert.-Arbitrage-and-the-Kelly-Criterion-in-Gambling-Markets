import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

p = np.linspace(0,1,1001)
fig = plt.figure()
ax = fig.add_subplot()
colors = ['0.85', '0.5', '0']
for i, k in enumerate([24,99,399]):
    ax.plot(p,p*(1-p)/(1+k*np.sqrt(1-4*(p-0.5)**2)),linewidth=0.75,color=colors[i],label=fr'$k={k}$')
ax.set_xlabel(r'$\mu$',fontsize=13)
ax.set_ylabel(r'$\sigma^2$',fontsize=13)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(loc='upper right')
plt.show()