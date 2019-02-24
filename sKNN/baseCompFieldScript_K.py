import os
import argparse
import numpy as np
import scipy.io as sio
import random
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

density_param = {'density': True}

dir_name = "/Volumes/My Passport/nn_dict"
os.chdir(dir_name)
fname =  "nn_dict.mat"
data_tmp = sio.loadmat(fname)
nn_dict = data_tmp['nn_dict']

pat_ids_test = nn_dict[:,0]
pat_elems = nn_dict[:,1]

distComp = np.zeros((2, len(pat_ids_test)))

for ii in range(len(pat_ids_test)):
    const_pat_id, const_elem_id = pat_ids_test[ii], pat_elems[ii]
    runfile('/Volumes/My Passport/baseCompField_K.py', wdir='/Volumes/My Passport')
    distComp[0,ii] = cos_corr(true_field, nn_field_mean)
    distComp[1,ii] = cos_corr(true_field, mean_field)
 
# create plot
fig, ax = plt.subplots(figsize=(10, 5))
index = np.arange(len(pat_ids_test))
bar_width = 0.3
rects1 = plt.bar(index, distComp[0,:], bar_width, label='true v pred')
rects2 = plt.bar(index + bar_width, distComp[1,:], bar_width, label='true v mean')

plt.xlabel('Patients')
plt.ylabel('Cos Corr')
plt.title('Cos Corr, larger blue is better')
plt.xticks(index + bar_width/2, range(len(pat_ids_test)))
plt.legend()
plt.tight_layout()
plt.show()

#bins_list = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000]
bins_list = [0.2, .4, .6, .8, 1.0]
plt.hist([distComp[0,:], distComp[1,:]], bins=bins_list)
plt.xticks(bins_list)
plt.title('histogram on dist')
plt.legend(('true v pred', 'true v mean'))
plt.show()


# kernels: gaussian , epanechnikov , tophat
X_plot = np.linspace(0, 1, 1000)[:, np.newaxis]
fig, ax = plt.subplots()
cap_label = ['true v pred', 'true v mean']
c_text = ['#1f77b4','#ff7f0e']
for i in range(2):
    X = distComp[i,:][:, np.newaxis]
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(X)
    log_dens = kde.score_samples(X_plot)
    ax.plot(X_plot[:, 0], np.exp(log_dens), '-',
            label=cap_label[i].format(kernel))
    ax.scatter(X[:, 0], -0.05 - 0.1 * i - 1e-1 * np.random.random(X.shape[0]), marker = '+', color = c_text[i])

ax.legend(loc='upper left')

plt.show()

np.median(distComp[0,:])
np.median(distComp[1,:])
np.mean(distComp[0,:])
np.mean(distComp[1,:])
