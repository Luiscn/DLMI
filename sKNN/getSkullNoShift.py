import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt
import skimage.io
import time

#pat_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 101, 102, 201, 202, 203, 204, 205, 206,
#          207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
#          221, 222]

pat_id = [221]

start_time_sec = time.time()

for pat_num in pat_id:
    nul = []
    ict = 0

    for ct in ["%04d" % x for x in range(1, 1025)]:
        dir_raw_name = "/Volumes/My Passport/ALLSIMSattP/PAT" + "%03d" % pat_num +"KW"
        os.chdir(dir_raw_name)
#        fname_l = "Pat" + str(pat_num) + "Element" + str(ct) + "KWAVEatt"
        fname_l = "Pat" + str(pat_num) + "Element" + str(ct) + "_att_FINAL"
        mat = scipy.io.loadmat(fname_l, verify_compressed_data_integrity=False)
        dir_skull_name = "/Volumes/My Passport/ALLSKULL_full"
        os.chdir(dir_skull_name)
        if not os.path.isdir("PAT" + "%03d" % pat_num +"KW"):
            os.mkdir("PAT" + "%03d" % pat_num +"KW", mode=0o777)
        os.chdir("PAT" + "%03d" % pat_num +"KW")
    
        try:
            medium_s = mat['medium']
            val = medium_s[0,0]
            dens = val["density"]
            speed = val["sound_speed"]
            coeff = val["alpha_coeff"]
            
            medium={}
            medium["density"] = dens[:,:,:]
            medium["sound_speed"] = speed[:,:,:]
            medium["alpha_coeff"] = coeff[:,:,:]
            fname_w = "P" + str(pat_num) + "Elem" + str(ct) + ".mat"
            scipy.io.savemat(fname_w, medium)
#        break
        except KeyError:
            nul.append(ct)
            print(ct, "does not has medium, trying next")

print("Done! in", time.time() - start_time_sec)
    
