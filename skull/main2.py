import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random

from keras import backend as K

import tensorflow as tf
from keras.optimizers import SGD, Adam

from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from zern2 import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.chdir('/Users/luis/Documents/MATLAB/skull')
z_coeff_dic = sio.loadmat('z_coeff.mat')
z_coeff = z_coeff_dic['a4s']
z_real = np.real(z_coeff)
z_imag = np.imag(z_coeff)
z_cb = np.concatenate((z_real, z_imag), axis = 1)
DELEMENTS=np.asanyarray([10, 12, 16, 57, 58, 62, 78, 95, 96, 143, 231, 272, 283, 287, 317, 398, 458, 475, 479, 509, 510,
           511, 608, 667, 668, 777, 778, 831, 860, 927, 928]);
z_cb = np.delete(z_cb,DELEMENTS,0) # z_cb: zernike coeff real & imag conbined
goodele = np.linspace(1,1024,1024)
idx_g = np.delete(goodele,DELEMENTS-1)


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

data = np.expand_dims(data, axis=4)

os.chdir('/Users/luis/Documents/MedImag/skull')

z_cb = z_cb[np.newaxis,np.newaxis,np.newaxis,:]
z_cb = np.rollaxis(z_cb, 3, 0)

z_tanh = np.tanh(z_cb/100)

data[:,:,:,0,0] = data[:,:,:,0,0]/np.max(data[:,:,:,0,0])
data[:,:,:,1,0] = data[:,:,:,1,0]/np.max(data[:,:,:,1,0])
data[:,:,:,2,0] = data[:,:,:,2,0]/np.percentile(data[:,:,:,2,0],99.9)


model=net3(V_WIDTH,V_HEIGHT,V_DEPTH)
model.summary()

sgd = SGD(lr=1e-2, decay=2e-2, momentum=0.9, nesterov=True)
#adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

ct_p, ct_n = 0, 0
idx_p, idx_n = [], []
for i in range(993):
    if z_tanh[i,0,0,0,:][28] > 0.4:
        ct_p += 1
        idx_p.append(i)
    elif  z_tanh[i,0,0,0,:][28] < -0.4:
        ct_n += 1
        idx_n.append(i)

## Fit model
#earlystopper = EarlyStopping(patience=10, verbose=1)
checkpointer = ModelCheckpoint('./model/mian2_model2.h5', verbose=1, save_best_only=True)

#results = model.fit(data[0:900,:,:,:,:], data[0:900,:,:,:,:], validation_split=0.1, batch_size=8, epochs=200,
#                     callbacks=[checkpointer])

random_idx = idx_n[:220]+idx_p[:220]
random.shuffle(random_idx)
#results = model.fit(data[random_idx,:,:,:,:], z_tanh[random_idx,:,:,:,:],
#                         validation_split=0.1, batch_size=8, epochs=200,
#                         callbacks=[checkpointer])

model = load_model('./model/mian2_model2.h5')
preds_test = model.predict(data[:,:,:,:,:], verbose=1)


test_point = 40
fig = plt.figure(figsize=(8, 8))
plt.subplot(211)
plt.plot(z_tanh[test_point,0,0,0,:])
plt.title("GT")
plt.subplot(212)
plt.plot(preds_test[test_point,0,0,0,:])
plt.show()


z_mean = np.mean(z_tanh,axis=0)
plt.plot(z_mean[0,0,0,:])
plt.title('mean of GT')
plt.show()

t_in = np.zeros((1,44,44,3,1))
t_out = model.predict(t_in, verbose=1)
plt.plot(t_out[0,0,0,0,:])
plt.title('output given all-zero as input')
plt.show()

t_in = np.ones((1,44,44,3,1))
t_out = model.predict(t_in, verbose=1)
plt.plot(t_out[0,0,0,0,:])
plt.title('output given all-one as input')
plt.show()

random.shuffle(idx_p)
random.shuffle(idx_n)

for tt in idx_p[:20]:
    plt.plot(preds_test[tt,0,0,0,:])
plt.title('outputs given multiple (pos) inputs')
plt.show()

for tt in idx_n[:20]:
    plt.plot(preds_test[tt,0,0,0,:])
plt.title('outputs given multiple (neg) inputs')
plt.show()


plt.imshow(data[0,:,:,2,0])
plt.colorbar()
plt.show()




