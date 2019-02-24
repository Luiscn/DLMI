import os
import argparse
import numpy as np
import scipy.io as sio
import random
import time
import matplotlib.pyplot as plt
import matplotlib as mpl


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
    runfile('/Volumes/My Passport/baseComp_full.py', wdir='/Volumes/My Passport')
    distComp[0,ii] = sDist(true_mag, nn_mag)
    distComp[1,ii] = sDist(true_mag, mean_mag)
 
# create plot
fig, ax = plt.subplots(figsize=(10, 5))
index = np.arange(len(pat_ids_test))
bar_width = 0.3
rects1 = plt.bar(index, distComp[0,:], bar_width, label='true v pred')
rects2 = plt.bar(index + bar_width, distComp[1,:], bar_width, label='true v mean')

plt.xlabel('Patients')
plt.ylabel('Dist')
plt.title('Dists, less blue is better')
plt.xticks(index + bar_width/2, range(len(pat_ids_test)))
plt.legend()
plt.tight_layout()
plt.show()

bins_list = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
plt.hist([distComp[0,:], distComp[1,:]], bins=bins_list)
plt.xticks(bins_list)
plt.title('histogram on dist')
plt.legend(('true v pred', 'true v mean'))
plt.show()

np.median(distComp[0,:])
np.median(distComp[1,:])
np.mean(distComp[0,:])
np.mean(distComp[1,:])
