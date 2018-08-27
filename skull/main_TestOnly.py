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
from skimage.transform import resize, rescale
from skimage.morphology import label
from skimage.color import rgb2gray, gray2rgb

from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

#from unet import *
#from unet_lin import *
#from unet_lin_2x2 import *
from unet_lin_do import *
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

TEST_PATH = './lat_test/test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed



# Build U-Net model
model=net3(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
model.summary()

# Predict on train, val and test
model = load_model('./model/model_29.h5')
preds_test = model.predict(X_test, verbose=1)


