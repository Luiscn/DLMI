import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import time

#dir_name = "/Volumes/My Passport/nn_dict"
#os.chdir(dir_name)
#fname =  "nn_dict.mat"
#data_tmp = sio.loadmat(fname)
#nn_dict = data_tmp['nn_dict']

nn_dict = np.zeros((K*3 + 2,)).astype(int)
nn_dict = nn_dict[np.newaxis,:]

pat_rand = np.random.choice(32,36,replace=True)
item_rand = np.random.choice(993,36,replace=False)

for i in range(len(pat_rand)):
    test_pat = pat_ids[pat_rand[i]]
    test_item = idx_g[item_rand[i]]
    runfile('/Volumes/My Passport/sKNN_full_K.py', wdir='/Volumes/My Passport')

    nn_dict_itm = np.concatenate((np.array(test_id)[np.newaxis, :], nn_flatten),axis=1)
    nn_dict = np.concatenate((nn_dict,nn_dict_itm.astype(int)))

nn_dict = np.delete(nn_dict,0,axis=0)
nn_dict = nn_dict.astype(int)

table = {}
table['nn_dict'] = nn_dict
os.chdir("/Volumes/My Passport/nn_dict")
fname = "nn_dict.mat"
sio.savemat(fname, table)

#nn_dict = np.delete(nn_dict,4,axis=0).astype(int)
