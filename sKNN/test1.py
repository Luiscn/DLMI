import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from pkUtils import *
#from cosUtils import *

def sDist(s1, s2):
    return np.linalg.norm(s1-s2)

DELEMENTS=np.asanyarray([10, 12, 16, 57, 58, 62, 78, 95, 96, 143, 231, 272, 283, 287, 317, 398, 458, 475, 479, 509, 510,
           511, 608, 667, 668, 777, 778, 831, 860, 927, 928]);
goodele = np.linspace(1,1024,1024)
idx_g = np.delete(goodele,DELEMENTS-1)

#pat_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 101, 102, 201, 202, 203, 204, 205, 206,
#          207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
#          221, 222]
pat_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 101, 102, 201, 202, 203, 204, 205, 206,
          207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
          221]

numTrain = 993
V_LENGTH = 140
V_WIDTH = 44
V_HEIGHT = 44

test_id = [210, 42]

dir_name = "/Volumes/My Passport/ALLSKULL/PAT" + "%03d" % test_id[0] + "KW"
os.chdir(dir_name)
fname =  "P" + str(test_id[0]) + "Elem" + "%04d" % test_id[1] + ".mat"
data_tmp = sio.loadmat(fname)
test_data = data_tmp['density']
dir_name = "/Volumes/My Passport/ALLSIMSattP/PAT" + "%03d" % test_id[0] + "KW"
os.chdir(dir_name)
fname =  fname_l = "Pat" + str(test_id[0]) + "Element" + "%04d" % test_id[1] + "_att_FINAL.mat"
data_tmp = sio.loadmat(fname)
mag_tmp = data_tmp['mag2']
true_mag = mag_tmp[:,:,479]

nn_id_dict = {}
for pat_id in pat_ids:
    if pat_id != test_id[0]:
        start_time_sec = time.time()
#        print(pat_id)
        dir_name = "/Volumes/My Passport/ALLSKULL/PAT" + "%03d" % pat_id + "KW"
        os.chdir(dir_name)
        data_base = np.zeros([numTrain,V_WIDTH,V_HEIGHT,V_LENGTH])
        dist = np.zeros(len(idx_g))
        t = 0
        for i in ["%04d" % x for x in idx_g]:   
            fname =  "P" + str(pat_id) + "Elem" + str(i) + ".mat"
            data_tmp = sio.loadmat(fname)
            dist[t] = sDist(test_data, data_tmp['density'])
            t += 1

        ind = np.argmin(dist)
        nn_id_dict[str(pat_id)] = ind
        print(nn_id_dict)
        os.chdir("/Volumes/My Passport/")
        save_obj(nn_id_dict, 'nn_dict_' + str(test_id[0]) + '_' + str(test_id[1]))
        print("Done! in", time.time() - start_time_sec)

nn_dist = {}
for key, val in nn_id_dict.items():
#    int(key)
#    idx_g[val]
    dir_name = "/Volumes/My Passport/ALLSKULL/PAT" + "%03d" % int(key) + "KW"
    os.chdir(dir_name)
    fname =  "P" + str(int(key)) + "Elem" + "%04d" % idx_g[val] + ".mat"
    data_tmp = sio.loadmat(fname)
    nn_den = data_tmp['density']
 
    nn_dist[key] = sDist(test_data, nn_den)

idx = min(nn_dist, key=nn_dist.get)
ele = idx_g[nn_id_dict[idx]]
nn_skull = [int(idx), int(ele)]









