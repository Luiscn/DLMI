import os
import argparse
import numpy as np
import scipy.io as sio
import random
import time
import matplotlib.pyplot as plt
import matplotlib as mpl


DELEMENTS=np.asanyarray([10, 12, 16, 57, 58, 62, 78, 95, 96, 143, 231, 272, 283, 287, 317, 398, 458, 475, 479, 509, 510,
           511, 608, 667, 668, 777, 778, 831, 860, 927, 928]);
goodele = np.linspace(1,1024,1024)
idx_g = np.delete(goodele,DELEMENTS-1)

def sDist(s1, s2):
    return np.linalg.norm(s1-s2)

#if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description='Skull KNN Demostration')
#    parser.add_argument(
#        '--pat_id', type=int, default=207,
#        help="patient ID, ex, 1, 102, 210")
#    parser.add_argument(
#        '--elem_id', type=int, default=702,
#        help='Element ID, ex, 1 ~ 1024')
#args = parser.parse_args()
#
#test_id = [args.pat_id, args.elem_id]
    
test_id = [const_pat_id, const_elem_id]

#dir_name = "/Volumes/My Passport/ALLSKULL_full/PAT" + "%03d" % test_id[0] + "KW"
#os.chdir(dir_name)
#fname =  "P" + str(test_id[0]) + "Elem" + "%04d" % test_id[1] + ".mat"
#data_tmp = sio.loadmat(fname)
#test_data = data_tmp['density']
dir_name = "/Volumes/My Passport/ALLSIMSattP/PAT" + "%03d" % test_id[0] + "KW"
os.chdir(dir_name)
fname  = "Pat" + str(test_id[0]) + "Element" + "%04d" % test_id[1] + "_att_FINAL.mat"
data_tmp = sio.loadmat(fname)
mag_tmp = data_tmp['mag2']
true_mag = mag_tmp[:,:,458]

dir_name = "/Volumes/My Passport/nn_dict"
os.chdir(dir_name)
fname =  "nn_dict.mat"
data_tmp = sio.loadmat(fname)
nn_dict = data_tmp['nn_dict']
for i in range(len(nn_dict)):
    if nn_dict[i,0] == test_id[0] and nn_dict[i,1] == test_id[1]:
        nn_entry = nn_dict[i,:]
        break
    
K = 2
nn_skull = np.zeros((K,2)).astype(int)
for i in range(K): 
    idx = str(nn_entry[2+3*i])
    ele = nn_entry[3+3*i]
    nn_skull[i,:] = [int(idx), int(ele)]

K_coeff = np.linspace(1,K,K)/np.sum(np.linspace(1,K,K))
x1, K_coeff_x, x2 = np.meshgrid(np.linspace(1,44,44), K_coeff, np.linspace(1,44,44))
nn_mag = np.zeros((K,44,44))
for i in range(K):   
    # Magnitudes
    dir_name = "/Volumes/My Passport/ALLSIMSattP/PAT" + "%03d" % int(nn_skull[i,0]) + "KW"
    os.chdir(dir_name)
    fname = "Pat" + str(nn_skull[i,0]) + "Element" + "%04d" % nn_skull[i,1] + "_att_FINAL.mat"
    data_tmp = sio.loadmat(fname)
    mag_tmp = data_tmp['mag2']
    nn_mag[i,:,:] = mag_tmp[:,:,458]
    
nn_mag_mean = np.sum(nn_mag * K_coeff_x, axis = 0)

# mean mag
dir_name = "/Volumes/My Passport/mag_data"
os.chdir(dir_name)
fname = "mean_mag.mat"
data_tmp = sio.loadmat(fname)
mean_mag = data_tmp['mag_mean']

cmin = np.min([np.min(true_mag), np.min(nn_mag_mean), np.min(mean_mag)])
cmax = np.max([np.max(true_mag), np.max(nn_mag_mean), np.max(mean_mag)])
fig = plt.figure(figsize=(12, 3))
plt.subplot(132)
plt.imshow(true_mag, vmin = cmin, vmax = cmax)
plt.colorbar()
plt.title('TestSample, P-' + str(test_id[0]) + '-E-' + str(test_id[1]))
plt.xticks([], [])
plt.yticks([], [])
plt.subplot(133)
plt.imshow(nn_mag_mean, vmin = cmin, vmax = cmax)
plt.colorbar()
plt.title('NN(s), normalized')
plt.xticks([], [])
plt.yticks([], [])
plt.subplot(131)
plt.imshow(mean_mag, vmin = cmin, vmax = cmax)
plt.colorbar()
plt.title("Mean, magitudes of all patients")
plt.xticks([], [])
plt.yticks([], [])
#plt.tight_layout()
plt.show()















