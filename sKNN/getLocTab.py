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

pat_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 101, 102, 201, 202, 203, 204, 205, 206,
          207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
          221]

loc_tab = np.zeros((32, 993))

start_time_sec = time.time()

pat_ct = 0
for pat_num in pat_id:
    
    dir_raw_name = "/Volumes/My Passport/ALLSIMSattP/PAT" + "%03d" % pat_num +"KW"
    os.chdir(dir_raw_name)

    itm_ct = 0

    nul=[]
    for ct in ["%04d" % x for x in range(1, 1025)]:
        fname_l = "Pat" + str(pat_num) + "Element" + str(ct) + "_att_FINAL"
        mat = sio.loadmat(fname_l, verify_compressed_data_integrity=False)
    
        try:
            medium_s = mat['medium']
            val = medium_s[0,0]
            dens = val["density"]

            midLine = dens[22,22,:]
            valBool=(midLine-1100)/np.abs(midLine-1100)+1
            idxNonzero = np.nonzero(valBool)
            indMin = idxNonzero[0][0]
            indMax = idxNonzero[0][-1]
            indMean_c = np.round((indMax + indMin)/2)
            loc_tab[pat_ct, itm_ct] = indMean_c
            itm_ct += 1
            
        except KeyError:
            nul.append(ct)
            print(ct, "does not has medium, trying next")
        
    pat_ct += 1

table = {}
table['loc_table'] = loc_tab.astype(int)
os.chdir("/Volumes/My Passport/loc_table")
fname = "loc_tab.mat"
sio.savemat(fname, table)

print("Done! in", time.time() - start_time_sec)


