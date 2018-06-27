import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
import skimage.io
from skimage.transform import resize
from skimage.morphology import label
from skimage.color import rgb2gray, gray2rgb

from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

#from unet import *
from unet_lin import *
#from unet_lin_2x2 import *
#from unet_lin_do import *
#from unet_sc_do import *
#from unet_sc import *
#from unet_7layers import *
#from unet_5layers import *

from arrayUtils import imlinmap, resultCompare

from keras.optimizers import SGD

# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = './input_more/train/'
TEST_PATH = './input/test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

# Get and resize train images and masks
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1),dtype=np.uint8)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/recon/' + 'recon_' + id_ + '.png')[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
    for mask_file in next(os.walk(path + '/true_pt/'))[2]:
        mask_ = imread(path + '/true_pt/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask

# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/recon/' + 'recon_' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
    for mask_file in next(os.walk(path + '/true_pt/'))[2]:
        mask_ = imread(path + '/true_pt/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    Y_test[n] = mask

print('Done!')

# Build U-Net model
model=unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
model.summary()

# Fit model
#earlystopper = EarlyStopping(patience=10, verbose=1)
#checkpointer = ModelCheckpoint('./model/model_26.h5', verbose=1, save_best_only=True)
#results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=8, epochs=200,
#                     callbacks=[checkpointer])

# Predict on train, val and test
model = load_model('./model/model_20.h5')
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
# preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)


# Perform a sanity check on some random training samples
ix = 32
skimage.io.imshow(X_train[ix]) # original image to be reconstructed
skimage.io.show()

#qweX=rgb2gray(X_train[ix])
#n,bins,patches = plt.hist(qweX, 100, facecolor='blue', alpha=0.5)
#plt.savefig("./result/histX.png")
#plt.show()

skimage.io.imshow(Y_train[ix,:,:,0]) # ground truth of the ix-th sample
skimage.io.show()

#qweY=Y_train[ix,:,:,0]
#n,bins,patches = plt.hist(qweY, 100, facecolor='blue', alpha=0.5)
#plt.savefig("./result/histY.png")
#plt.show()


preds_train_test = preds_train[ix,:,:,0] # scores of the ix-th sample's prediction
#preds_train_test=imlinmap(preds_train_test,[np.min(preds_train_test), np.max(preds_train_test)], [0,1])

#qwe123=preds_train_test
#n,bins,patches = plt.hist(qwe123, 100, facecolor='blue', alpha=0.5)
#plt.savefig("./result/hist.png")
#plt.show()

#preds_train_test=imlinmap(preds_train_test,[np.min(preds_train_test), np.max(preds_train_test)], [0,1])
preds_train_test=imlinmap(preds_train_test,[np.percentile(preds_train_test,0.5), 255], [0,1])
skimage.io.imshow(preds_train_test)
skimage.io.show()

X_train_test=X_train[ix,:,:,:]
X_train_test=10*np.log10(X_train_test/np.max(X_train_test) +0.001).astype(np.float32)
#X_train_test=imlinmap(X_train_test,[np.min(X_train_test), np.max(X_train_test)], [0, 1]).astype(np.float32)
X_train_test = imlinmap(X_train_test, [-40, 0], [0, 1])
skimage.io.imshow(X_train_test)
skimage.io.show()

score=preds_train_test
#score=imlinmap(score,[np.percentile(score, 0.5), np.percentile(score, 100)],[np.min(score),np.max(score)])
score=10*np.log10(score/np.max(score) +0.001)
#score=imlinmap(score, [np.min(score), np.max(score)], [0,1]).astype(np.float32)
score = imlinmap(score, [-40, 0], [0, 1]).astype(np.float32)
skimage.io.imshow(score)
skimage.io.show()

skimage.io.imsave('./result/tr-in_lin.png',X_train[ix])
skimage.io.imsave('./result/tr-gndTruth.png',Y_train[ix,:,:,0])
skimage.io.imsave('./result/tr-out_lin.png',preds_train_test)
skimage.io.imsave('./result/tr-in_log(-40to0dB).png',X_train_test)
skimage.io.imsave('./result/tr-out_log(-40to0dB).png',score)

# # Perform a sanity check on some random testing samples
ix = 24
skimage.io.imshow(X_test[ix]) # original image to be reconstructed
skimage.io.show()
skimage.io.imshow(Y_test[ix,:,:,0]) # ground truth of the ix-th sample
skimage.io.show()

preds_test_test = preds_test[ix,:,:,0] # scores of the ix-th sample's prediction
#preds_test_test=imlinmap(preds_test_test,[np.min(preds_test_test), np.max(preds_test_test)], [0,1])
preds_test_test=imlinmap(preds_test_test,[np.percentile(preds_test_test, 0.5), 255], [0,1])
skimage.io.imshow(preds_test_test)
skimage.io.show()

X_test_test=X_test[ix,:,:,:]
X_test_test=10*np.log10(X_test_test/np.max(X_test_test) +0.001).astype(np.float32)
#X_train_test=imlinmap(X_test_test,[np.min(X_train_test), np.max(X_test_test)], [0, 1]).astype(np.float32)
X_test_test = imlinmap(X_test_test, [-40, 0], [0, 1])
skimage.io.imshow(X_test_test)
skimage.io.show()

score=preds_test_test
#score=imlinmap(score,[np.percentile(score, 0.5), np.percentile(score, 100)],[np.min(score),np.max(score)])
score=10*np.log10(score/np.max(score) +0.001)
#score=imlinmap(score, [np.min(score), np.max(score)], [0,1]).astype(np.float32)
score = imlinmap(score, [-40, 0], [0, 1]).astype(np.float32)
skimage.io.imshow(score)
skimage.io.show()

skimage.io.imsave('./result/te-in_lin.png',X_test[ix])
skimage.io.imsave('./result/te-gndTruth.png',Y_test[ix,:,:,0])
skimage.io.imsave('./result/te-out_lin.png',preds_test_test)
skimage.io.imsave('./result/te-in_log(-40to0dB).png',X_test_test)
skimage.io.imsave('./result/te-out_log(-40to0dB).png',score)
