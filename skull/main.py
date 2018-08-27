import os
import numpy as np
import scipy.io as sio

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras import backend as K

import tensorflow as tf
from keras.optimizers import SGD, Adam

from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from unet_lin_do import *

os.chdir('/Users/luis/Documents/MATLAB/skull/skullSim')

numTrain = 100

V_LENGTH = 100
V_WIDTH = 40
V_HEIGHT = 35

data = np.zeros([numTrain,100,40,35])
label = np.zeros([numTrain,1,4])
for i in range(numTrain):
    os.chdir('id' + str(i+1))
    data_tmp = sio.loadmat('skl.mat')
    data[i,:,:,:] = data_tmp['skl']
    label_tmp = sio.loadmat('label.mat')
    label[i,:,:] = label_tmp['label']
    os.chdir('..')

os.chdir('/Users/luis/Documents/MedImag/skull')

data = np.expand_dims(data, axis=4)
for add_ax in [3,4,5]:
    label = np.expand_dims(label, axis=add_ax)
    
label_thk = label[:,:,0,:,:]
label_q1 = label[:,:,1,:,:]
label_q2 = label[:,:,2,:,:]
label_den = label[:,:,3,:,:]

model=net3(V_LENGTH, V_WIDTH, V_HEIGHT)
model.summary()

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Fit model
earlystopper = EarlyStopping(patience=10, verbose=1)
checkpointer = ModelCheckpoint('./model/model_4.h5', verbose=1, save_best_only=True)
results = model.fit(data[0:90,:,:,:,:], label[0:90,:,1,:,:], validation_split=0.1, batch_size=8, epochs=200,
                     callbacks=[checkpointer, earlystopper])

model = load_model('./model/model_4.h5')
preds_test = model.predict(data[90:100,:,:,:,:], verbose=1)




r_pred = np.squeeze(preds_test)
r_labl = np.squeeze(label_thk[90:100,:,:,:,:])
r = np.concatenate((np.expand_dims(r_pred,axis=1),np.expand_dims(r_labl,axis=1)),axis=1)













