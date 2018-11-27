import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
import random

def sDist(s1, s2):
    return np.linalg.norm(s1-s2)

DELEMENTS=np.asanyarray([10, 12, 16, 57, 58, 62, 78, 95, 96, 143, 231, 272, 283, 287, 317, 398, 458, 475, 479, 509, 510,
           511, 608, 667, 668, 777, 778, 831, 860, 927, 928]);
goodele = np.linspace(1,1024,1024)
idx_g = np.delete(goodele,DELEMENTS-1)

os.chdir('/Users/luis/Documents/MedImag/skull/mag_label')
fname = "P215-label_480.mat"
label = sio.loadmat(fname)
label_mag = label['label_mag']

os.chdir('/Users/luis/Documents/MedImag/skull/data2')

numTrain = 993
V_DEPTH = 3
V_WIDTH = 44
V_HEIGHT = 44
data = np.zeros([numTrain,V_WIDTH,V_HEIGHT,V_DEPTH])

t=0
for i in idx_g:   
    fname = 'P215-Elem' + "%04d" % i
    data_tmp = sio.loadmat(fname)
    data[t,:,:,:] = data_tmp['dens_info']
    t+=1

os.chdir('/Users/luis/Documents/MedImag/skull/data')

numTrain = 993
V_LENGTH = 50
V_WIDTH = 44
V_HEIGHT = 44
data_b = np.zeros([numTrain,V_WIDTH,V_HEIGHT,V_LENGTH])
t=0
for i in idx_g:   
    fname = 'P215-Elem' + "%04d" % i
    data_tmp = sio.loadmat(fname)
    data_b[t,:,:,:] = data_tmp['density']
    t+=1


os.chdir('/Users/luis/Documents/MATLAB/skull')

fname = "phz459.mat"
phz_in = sio.loadmat(fname)
phz_in = phz_in['phz459']
phz_in = np.unwrap(phz_in, axis=1)
phz_in = np.unwrap(phz_in, axis=2)

fname = "noskull.mat"
noskull = sio.loadmat(fname)
mag_nsk = noskull['mag2']
phz_nsk = noskull['phz']
phz_nsk = np.unwrap(phz_nsk, axis=1)
phz_nsk = np.unwrap(phz_nsk, axis=2)

os.chdir('/Users/luis/Documents/MedImag/skull')

data[:,:,:,0] = data[:,:,:,0]/np.max(data[:,:,:,0])
data[:,:,:,1] = data[:,:,:,1]/np.max(data[:,:,:,1])
data[:,:,:,2] = data[:,:,:,2]/np.percentile(data[:,:,:,2],99.9)

data_train = data_b[:900,:,:,:]
data_test = data_b[900:,:,:,:]

K = 2 # k nearest

ind = np.zeros((len(data_test), K))
skull_diff = np.zeros((len(data_test), K))
for i in range(len(data_test)):
    dist = np.zeros(len(data_train))
    for j in range(len(data_train)):
        dist[j] = sDist(data_test[i,:,:,:], data_train[j,:,:,:])
        ind[i,:] = np.argpartition(dist, K)[:K]
        skull_diff[i,:] = np.sort(dist)[0:1]
    

field_train = label_mag[:,:,:900]
field_test = label_mag[:,:,900:]

K_f = 2 # k nearest

ind_f = np.zeros((np.shape(field_test)[2], K_f))
field_diff = np.zeros((np.shape(field_test)[2], K_f))
for i in range(np.shape(field_test)[2]):
    dist_f = np.zeros(np.shape(field_train)[2])
    for j in range(np.shape(field_train)[2]):
        dist_f[j] = sDist(field_test[:,:,i], field_train[:,:,j])
        ind_f[i,:] = np.argpartition(dist_f, K_f)[:K_f]
        field_diff[i,:] = np.sort(dist_f)[0:1]








    