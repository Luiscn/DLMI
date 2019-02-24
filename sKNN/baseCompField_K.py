import os
import argparse
import numpy as np
import scipy.io as sio
import random
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy.linalg import norm

DELEMENTS=np.asanyarray([10, 12, 16, 57, 58, 62, 78, 95, 96, 143, 231, 272, 283, 287, 317, 398, 458, 475, 479, 509, 510,
           511, 608, 667, 668, 777, 778, 831, 860, 927, 928]);
goodele = np.linspace(1,1024,1024)
idx_g = np.delete(goodele,DELEMENTS-1)


def cos_corr(in1, in2):
    in1 = np.reshape(in1, (1,-1))
    in2 = np.reshape(in2, (-1,1))
    return np.abs(np.dot(in1,np.conj(in2))) / (norm(in1)*norm(in2))
    
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
phz_tmp = data_tmp['phz']
true_mag = mag_tmp[:,:,458]
true_phz = phz_tmp[:,:,458]
true_field = true_mag * np.exp(1j * true_phz)

dir_name = "/Volumes/My Passport/nn_dict"
os.chdir(dir_name)
fname =  "nn_dict.mat"
data_tmp = sio.loadmat(fname)
nn_dict = data_tmp['nn_dict']
for i in range(len(nn_dict)):
    if nn_dict[i,0] == test_id[0] and nn_dict[i,1] == test_id[1]:
        nn_entry = nn_dict[i,:]
        break
    
K = 5
nn_skull = np.zeros((K,2)).astype(int)
for i in range(K): 
    idx = str(nn_entry[2+3*i])
    ele = nn_entry[3+3*i]
    nn_skull[i,:] = [int(idx), int(ele)]

K_coeff = np.linspace(1,K,K)/np.sum(np.linspace(1,K,K))
x1, K_coeff_x, x2 = np.meshgrid(np.linspace(1,44,44), K_coeff, np.linspace(1,44,44))
nn_field = np.zeros((K,44,44)).astype('complex128')
for i in range(K):   
    # Magnitudes
    dir_name = "/Volumes/My Passport/ALLSIMSattP/PAT" + "%03d" % int(nn_skull[i,0]) + "KW"
    os.chdir(dir_name)
    fname = "Pat" + str(nn_skull[i,0]) + "Element" + "%04d" % nn_skull[i,1] + "_att_FINAL.mat"
    data_tmp = sio.loadmat(fname)
    mag_tmp = data_tmp['mag2']
    phz_tmp = data_tmp['phz']
    nn_mag = mag_tmp[:,:,458]
    nn_phz = phz_tmp[:,:,458]
    nn_field[i,:,:] = nn_mag * np.exp(1j * nn_phz)
    
nn_field_mean = np.sum(nn_field * K_coeff_x, axis = 0)

# mean mag
dir_name = "/Volumes/My Passport/phz_data"
os.chdir(dir_name)
fname = "mean_field.mat"
data_tmp = sio.loadmat(fname)
mean_field = data_tmp['field_mean']








