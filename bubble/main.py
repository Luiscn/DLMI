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

from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf

from unet import *
#from unet_7layers import *
#from unet_5layers import *

from arrayUtils import imlinmap

from keras.optimizers import Adam
from keras.optimizers import SGD

# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = './input/train/'
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

# Check if training data looks all right
# ix = random.randint(0, len(train_ids)-1)
# skimage.io.imshow(X_train[ix])
# skimage.io.show()
# plt.imshow(np.squeeze(Y_train[ix]))
# plt.show()

# Build U-Net model
model=unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
model.summary()

# Fit model
#earlystopper = EarlyStopping(patience=2, verbose=1)
#checkpointer = ModelCheckpoint('model_9.h5', verbose=1, save_best_only=True)
#results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=8, epochs=30,
#                     callbacks=[earlystopper, checkpointer])

# Predict on train, val and test
model = load_model('model_9.h5')
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
# preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# # Perform a sanity check on some random training samples
ix = 322
skimage.io.imshow(X_train[ix]) # original image to be reconstructed
skimage.io.show()
plt.imshow(Y_train[ix,:,:,0]) # ground truth of the ix-th sample
plt.show()
preds_train_test = preds_train[ix,:,:,0] # scores of the ix-th sample's prediction
skimage.io.imshow(preds_train_test)
skimage.io.show()

X_train_test=X_train[ix,:,:,:]
X_train_test=10*np.log10(X_train_test/np.max(X_train_test) +0.001).astype(np.float32)
#X_train_test=imlinmap(X_train_test,[np.min(X_train_test), np.max(X_train_test)], [0, 1]).astype(np.float32)
X_train_test = imlinmap(X_train_test, [-40, 0], [0, 1])
skimage.io.imshow(X_train_test)
skimage.io.show()

score=preds_train_test[1:127,1:127]
score=10*np.log10(score/np.max(score) +0.0000001)
#score=imlinmap(score, [np.min(score), np.max(score)], [0,1]).astype(np.float32)
score = imlinmap(score, [-40, 0], [0, 1]).astype(np.float32)
skimage.io.imshow(score)
skimage.io.show()

skimage.io.imsave('tr9-1.png',X_train[ix])
skimage.io.imsave('tr9-2.png',Y_train[ix,:,:,0])
skimage.io.imsave('tr9-3.png',preds_train_test)
skimage.io.imsave('tr9-4-40.png',X_train_test)
skimage.io.imsave('tr9-5-40.png',score)

# # Perform a sanity check on some random testing samples
ix = 14
skimage.io.imshow(X_test[ix]) # original image to be reconstructed
skimage.io.show()
skimage.io.imshow(Y_test[ix,:,:,0]) # ground truth of the ix-th sample
skimage.io.show()
preds_test_test = preds_test[ix,:,:,0] # scores of the ix-th sample's prediction
skimage.io.imshow(preds_test_test)
skimage.io.show()

X_test_test=X_test[ix,:,:,:]
X_test_test=10*np.log10(X_test_test/np.max(X_test_test) +0.001).astype(np.float32)
#X_train_test=imlinmap(X_test_test,[np.min(X_train_test), np.max(X_test_test)], [0, 1]).astype(np.float32)
X_test_test = imlinmap(X_test_test, [-40, 0], [0, 1])
skimage.io.imshow(X_test_test)
skimage.io.show()

score=preds_test_test[1:127,1:127]
score=10*np.log10(score/np.max(score) +0.0000001)
#score=imlinmap(score, [np.min(score), np.max(score)], [0,1]).astype(np.float32)
score = imlinmap(score, [-40, 0], [0, 1]).astype(np.float32)
skimage.io.imshow(score)
skimage.io.show()

skimage.io.imsave('te9-1.png',X_test[ix])
skimage.io.imsave('te9-2.png',Y_test[ix,:,:,0])
skimage.io.imsave('te9-3.png',preds_test_test)
skimage.io.imsave('te9-4-40.png',X_test_test)
skimage.io.imsave('te9-5-40.png',score)
