import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt
import skimage.io
import time

start_time_sec = time.time()
nul = []
for ct in["%04d" % x for x in range(1, 1025)]:
    os.chdir('/Users/luis/Dropbox/Pat215/SIMS8')
    fname_l = "Pat215Element" + str(ct) + "KWAVEatt"
    mat = scipy.io.loadmat(fname_l)
    os.chdir('/Users/luis/Documents/MedImag/skull/data2')
    try:
        medium_s = mat['medium']
        val = medium_s[0,0]
        dens = val["density"]
        speed = val["sound_speed"]
        coeff = val["alpha_coeff"]
        
        # dens_info
        diff = np.diff(dens)
        idx = np.zeros((44,44,2))
        
        dens_info = np.zeros((44,44,3))
        dens_mean = np.zeros((44,44))
        dens_center = np.zeros((44,44))
        dens_width = np.zeros((44,44))
        
        for i in range(44):
            for j in range(44):
                idx[i,j,0] = np.nonzero(diff[i,j,:])[0][1]
                idx[i,j,1] = np.nonzero(diff[i,j,:])[0][-1]
                dens_mean[i,j] = np.mean(dens[i,j,int(idx[i,j,0]+1):int(idx[i,j,1]+1)])
                
                dens_center[i,j] = np.sum(dens[i,j,int(idx[i,j,0]+1):int(idx[i,j,1])+1]
                *range(int(idx[i,j,0]+1),int(idx[i,j,1]+1))) / np.sum(dens[i,j,int(idx[i,j,0]+1):int(idx[i,j,1]+1)])
                
        dens_width = idx[:,:,1] - idx[:,:,0]
        
        dens_info[:,:,0] = dens_mean
        dens_info[:,:,1] = dens_center
        dens_info[:,:,2] = dens_width
        
        # speed_info
        diff = np.diff(speed)
        idx = np.zeros((44,44,2))
        
        speed_info = np.zeros((44,44,3))
        speed_mean = np.zeros((44,44))
        speed_center = np.zeros((44,44))
        speed_width = np.zeros((44,44))
        
        for i in range(44):
            for j in range(44):
                idx[i,j,0] = np.nonzero(diff[i,j,:])[0][1]
                idx[i,j,1] = np.nonzero(diff[i,j,:])[0][-1]
                speed_mean[i,j] = np.mean(speed[i,j,int(idx[i,j,0]+1):int(idx[i,j,1]+1)])
                
                speed_center[i,j] = np.sum(speed[i,j,int(idx[i,j,0]+1):int(idx[i,j,1])+1]
                *range(int(idx[i,j,0]+1),int(idx[i,j,1]+1))) / np.sum(speed[i,j,int(idx[i,j,0]+1):int(idx[i,j,1]+1)])
                
        speed_width = idx[:,:,1] - idx[:,:,0]
        
        speed_info[:,:,0] = speed_mean
        speed_info[:,:,1] = speed_center
        speed_info[:,:,2] = speed_width
        
        medium={}
        medium["dens_info"] = dens_info
        medium["speed_info"] = speed_info

        fname_w = "P215-Elem" + str(ct) + ".mat"
        
        scipy.io.savemat(fname_w, medium)
#        break
    except KeyError:
        nul.append(ct)
        print(ct, "does not has medium, trying next")

print("Done! in", time.time() - start_time_sec)
    
