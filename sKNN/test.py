import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from pkUtils import *

def sDist(s1, s2):
    return np.linalg.norm(s1-s2)

DELEMENTS=np.asanyarray([10, 12, 16, 57, 58, 62, 78, 95, 96, 143, 231, 272, 283, 287, 317, 398, 458, 475, 479, 509, 510,
           511, 608, 667, 668, 777, 778, 831, 860, 927, 928]);
goodele = np.linspace(1,1024,1024)
idx_g = np.delete(goodele,DELEMENTS-1)

#pat_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 101, 102, 201, 202, 203, 204, 205, 206,
#          207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
#          221, 222]
pat_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 101, 102, 201, 202, 203, 204, 205, 206,
          207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
          221]

aa = np.asarray([[1,2],[3,4]])
np.reshape(aa, (1,-1))
