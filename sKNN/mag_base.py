import os
import argparse
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from pkUtils import *

pat_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 101, 102, 201, 202, 203, 204, 205, 206,
          207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
          221]

DELEMENTS=np.asanyarray([10, 12, 16, 57, 58, 62, 78, 95, 96, 143, 231, 272, 283, 287, 317, 398, 458, 475, 479, 509, 510,
           511, 608, 667, 668, 777, 778, 831, 860, 927, 928]);
goodele = np.linspace(1,1024,1024,dtype = int)
idx_g = np.delete(goodele,DELEMENTS-1).tolist()


def getMag():
    mag = np.zeros((993,44,44))
    for pat_id in pat_ids:
        start_time_sec = time.time()
        for elem in idx_g:
            
            dir_name = "/Volumes/My Passport/ALLSIMSattP/PAT" + "%03d" % pat_id + "KW"
            os.chdir(dir_name)
            fname = "Pat" + str(pat_id) + "Element" + "%04d" % elem + "_att_FINAL.mat"
            data_tmp = sio.loadmat(fname, verify_compressed_data_integrity=False)
            mag_tmp = data_tmp['mag2']
            mag[idx_g.index(elem),:,:] = mag_tmp[:,:,458]
            
        mag_data={}
        mag_data["mag"] = mag
        fname_w = "P" + str(pat_id) + "mag.mat"

        dir_name = "/Volumes/My Passport/mag_data"
        os.chdir(dir_name)
        sio.savemat(fname_w, mag_data)
        
        print(str(pat_id) + " done! in", time.time() - start_time_sec)
    return mag

def meanMag():
    meanElem = np.zeros((44,44))
    mean = np.zeros((44,44))
    for pat_id in pat_ids:
        
        dir_name = "/Volumes/My Passport/mag_data"
        os.chdir(dir_name)
        fname_w = "P" + str(pat_id) + "mag.mat"
        data_tmp = sio.loadmat(fname_w)
        meanElem += np.mean(data_tmp["mag"], axis = 0)
        plt.imshow(np.mean(data_tmp["mag"], axis = 0))
        plt.colorbar()
        plt.title(str(pat_id))
        plt.show()
        
    mean = meanElem / len(pat_ids)
    
    mag_data={}
    mag_data["mag_mean"] = mean
    fname_w = "mean_mag.mat"
    dir_name = "/Volumes/My Passport/mag_data"
    os.chdir(dir_name)
#    sio.savemat(fname_w, mag_data)
        
    return mean

#res = getMag()
mean = meanMag()










