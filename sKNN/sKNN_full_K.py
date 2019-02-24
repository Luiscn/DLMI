import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import time
import matplotlib.pyplot as plt
import matplotlib as mpl

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

dir_name = "/Volumes/My Passport/loc_table"
os.chdir(dir_name)
fname =  "loc_tab.mat"
data_tmp = sio.loadmat(fname)
loc_tab = data_tmp['loc_table']

test_id = [test_pat, test_item]
midPoint = loc_tab[pat_ids.index(test_id[0]), list(idx_g).index(test_id[1])]

dir_name = "/Volumes/My Passport/ALLSKULL_full/PAT" + "%03d" % test_id[0] + "KW"
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

dist = 0
dist_dict = np.array([0,0,0])
dist_dict = dist_dict[np.newaxis,:]

K = 5
nn = np.zeros((K,3))

start_time_sec = time.time()
for pat_id in pat_ids:
    if pat_id != test_id[0]:
        
        dir_name = "/Volumes/My Passport/ALLSKULL_full/PAT" + "%03d" % pat_id + "KW"
        os.chdir(dir_name)
        for i in ["%04d" % x for x in idx_g]:
            mid_test = loc_tab[pat_ids.index(pat_id), list(idx_g).index(int(i))]
            if np.abs(midPoint - mid_test) <= 3:
                
                fname =  "P" + str(pat_id) + "Elem" + str(i) + ".mat"
                data_tmp = sio.loadmat(fname)
                
                dist = sDist(test_data[:,:,midPoint-25:midPoint+25],
                    data_tmp['density'][:,:,midPoint-25:midPoint+25])
                dist_itm_temp = np.array([pat_id,int(i),dist])
                dist_dict = np.concatenate((dist_dict,dist_itm_temp[np.newaxis,:]))
                
print("Done! in", time.time() - start_time_sec)

dist_dict = np.delete(dist_dict,0,axis=0)
for t in range(K):
    nn[t,:] = dist_dict[np.argsort(dist_dict[:,2])[t],:]
    
nn_flatten = np.reshape(nn, (1,-1))


        


