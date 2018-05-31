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

from keras.optimizers import Adam
from keras.optimizers import SGD

from unet import *
# from IoU import *
# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = './input/train/'
TEST_PATH = './input/train/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

# Get and resize train images and masks
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/recon/' + 'recon_' + id_ + '.png')[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/true/'))[2]:
        mask_ = imread(path + '/true/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask

# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/recon/' + 'recon_' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

print('Done!')

# Check if training data looks all right
# ix = random.randint(0, len(train_ids)-1)
# skimage.io.imshow(X_train[ix])
# skimage.io.show()
# plt.imshow(np.squeeze(Y_train[ix]))
# plt.show()

# Build U-Net model
model=unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
# Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Fit model
earlystopper = EarlyStopping(patience=2, verbose=1)
checkpointer = ModelCheckpoint('bubble_model.h5', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=8, epochs=30,
                     callbacks=[earlystopper, checkpointer])

# Predict on train, val and test
model = load_model('bubble_model.h5')
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
# preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
# preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
# preds_train_t = (preds_train > 0.3).astype(np.uint8)
# preds_val_t = (preds_val > 0.5).astype(np.uint8)
# preds_test_t = (preds_test > 0.5).astype(np.uint8)

# # Perform a sanity check on some random training samples
ix = 0
skimage.io.imshow(X_train[ix]) # original image to be reconstructed
skimage.io.show()
plt.imshow(Y_train[ix,:,:,0]) # ground truth of the ix-th sample
plt.show()
preds_train_test = preds_train[ix,:,:,0] # scores of the ix-th sample's prediction
skimage.io.imshow(preds_train_test)
skimage.io.show()
# preds_train_t_test = preds_train_t[ix,:,:,0]
# skimage.io.imshow(preds_train_t_test)
#  skimage.io.show()
