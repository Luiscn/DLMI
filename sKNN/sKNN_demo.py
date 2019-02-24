import os
import argparse
import numpy as np
import scipy.io as sio
import random
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from pkUtils import *

def sDist(s1, s2):
    return np.linalg.norm(s1-s2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Skull KNN Demostration')
    parser.add_argument(
        '--pat_id', type=int, default=1,
        help="patient ID, ex, 1, 102, 210")
    parser.add_argument(
        '--elem_id', type=int, default=1,
        help='Element ID, ex, 1 ~ 1024')
args = parser.parse_args()

DELEMENTS=np.asanyarray([10, 12, 16, 57, 58, 62, 78, 95, 96, 143, 231, 272, 283, 287, 317, 398, 458, 475, 479, 509, 510,
           511, 608, 667, 668, 777, 778, 831, 860, 927, 928]);
goodele = np.linspace(1,1024,1024)
idx_g = np.delete(goodele,DELEMENTS-1)

test_id = [args.pat_id, args.elem_id]

dir_name = "/Volumes/My Passport/ALLSKULL/PAT" + "%03d" % test_id[0] + "KW"
os.chdir(dir_name)
fname =  "P" + str(test_id[0]) + "Elem" + "%04d" % test_id[1] + ".mat"
data_tmp = sio.loadmat(fname)
test_data = data_tmp['density']
dir_name = "/Volumes/My Passport/ALLSIMSattP/PAT" + "%03d" % test_id[0] + "KW"
os.chdir(dir_name)
fname  = "Pat" + str(test_id[0]) + "Element" + "%04d" % test_id[1] + "_att_FINAL.mat"
data_tmp = sio.loadmat(fname)
mag_tmp = data_tmp['mag2']
true_mag = mag_tmp[:,:,458]

os.chdir("/Volumes/My Passport")
nn_id_dict = load_obj('nn_dict_' + str(test_id[0]) + '_' + str(test_id[1]))
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

# Magnitudes
dir_name = "/Volumes/My Passport/ALLSIMSattP/PAT" + "%03d" % int(idx) + "KW"
os.chdir(dir_name)
fname = "Pat" + idx + "Element" + "%04d" % ele + "_att_FINAL.mat"
data_tmp = sio.loadmat(fname)
mag_tmp = data_tmp['mag2']
nn_mag = mag_tmp[:,:,458]

cmin, cmax = np.minimum(np.min(true_mag), np.min(nn_mag)), np.maximum(np.max(true_mag), np.max(nn_mag))

fig = plt.figure(figsize=(11, 8))
plt.subplot(221)
plt.imshow(true_mag, vmin = cmin, vmax = cmax)
plt.colorbar()
plt.title('TestSample, PatientID = ' + str(test_id[0]) + ', ElementID = ' + str(test_id[1]))
plt.xticks([], [])
plt.yticks([], [])
plt.subplot(222)
plt.imshow(nn_mag, vmin = cmin, vmax = cmax)
plt.colorbar()
plt.title('NN, PatientID = ' + str(nn_skull[0]) + ', ElementID = ' + str(nn_skull[1]))
plt.xticks([], [])
plt.yticks([], [])
#plt.show()

# Skull xsec
dir_name = "/Volumes/My Passport/ALLSKULL/PAT" + "%03d" % int(idx) + "KW"
os.chdir(dir_name)
fname = "P" + idx + "Elem" + "%04d" % ele + ".mat"
data_tmp = sio.loadmat(fname)
nnn_den = data_tmp['density']

#fig = plt.figure(figsize=(10, 10))
plt.subplot(425)
plt.imshow(test_data[22,:,:])
plt.title('TestSample, PatientID = ' + str(test_id[0]) + ', ElementID = ' + str(test_id[1]) + ' (horizontal)')
plt.xticks([], [])
plt.yticks([], [])
plt.subplot(426)
plt.imshow(nnn_den[22,:,:])
plt.title('NN, PatientID = ' + str(nn_skull[0]) + ', ElementID = ' + str(nn_skull[1]) + ' (horizontal)')
plt.xticks([], [])
plt.yticks([], [])
plt.subplot(427)
plt.imshow(test_data[:,22,:])
plt.title('TestSample, PatientID = ' + str(test_id[0]) + ', ElementID = ' + str(test_id[1]) + ' (vertical)')
plt.xticks([], [])
plt.yticks([], [])
plt.subplot(428)
plt.imshow(nnn_den[:,22,:])
plt.title('NN, PatientID = ' + str(nn_skull[0]) + ', ElementID = ' + str(nn_skull[1]) + ' (vertical)')
plt.xticks([], [])
plt.yticks([], [])
plt.tight_layout()
plt.show()















