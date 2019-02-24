import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from distutils.version import LooseVersion
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

# `normed` is being deprecated in favor of `density` in histograms
if LooseVersion(matplotlib.__version__) >= '2.1':
    density_param = {'density': True}
else:
    density_param = {'normed': True}

# kernels: gaussian , epanechnikov , tophat

X_plot = np.linspace(0, 6000, 1000)[:, np.newaxis]

fig, ax = plt.subplots()
cap_label = ['true v pred', 'true v mean']
c_text = ['#1f77b4','#ff7f0e']
for i in range(2):
    X = distComp[i,:][:, np.newaxis]
    kde = KernelDensity(kernel='gaussian', bandwidth=600).fit(X)
    log_dens = kde.score_samples(X_plot)
    ax.plot(X_plot[:, 0], np.exp(log_dens), '-',
            label=cap_label[i].format(kernel))
    ax.scatter(X[:, 0], -5e-7 - 1e-5 * i - 1e-6 * np.random.random(X.shape[0]), marker = '+', color = c_text[i])

#plt.yticks([], [])
ax.legend(loc='upper right')
#plt.grid(axis = 'both')

plt.show()