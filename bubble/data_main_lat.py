#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate data for one source location

Created on Fri May 18 12:39:05 2018

@author: btracey
"""

import os
import arrayUtils as bf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mp
from PIL import Image
import skimage.io
import skimage.color
from skimage.transform import rescale

def simulateOneSource(filePostfix,DOPLOTS=1):
# pick a test source location
# xs_mm is source x-location; should be between -5 and +5
# zs_mm is source z-location; between 2 and 12 mm
# ampSrc = amplidue; between 1 and 5
#    examples:
# xs_mm,zs_mm = (-2.5, 8)
# ampSrc = 2.1;
    
    # set constants
    FS = 40e6
    TCAPTURE_MS = 0.01 # total data length
    AVGWIN_MS = 0.002  # averaging window for DAS
    SNR_LIN = 10
    
    ######### DATA SIMULATION ##########
    # set up array geometry (in mm)
    xArray_mm,zArray_mm=bf.defineArray()
    
    xgrid_mm = np.linspace(-5,5,128)
    zgrid_mm = np.linspace(2,12,128) 
    
    data, ptSrc = bf.latResPht(xgrid_mm, zgrid_mm, xArray_mm,zArray_mm,FS, TCAPTURE_MS, SNR_LIN)
        
    ########### IMAGE FORMATION ###########
#    [t1,tst] = bf.singleLocationDAS(data_sp,FS,delay_ms,AVGWIN_MS)  
    
    das = bf.formDASimage(data,xgrid_mm,zgrid_mm,FS,AVGWIN_MS,xArray_mm,zArray_mm)
    das = das.T
    
    dl = bf.normalizeImage(das)

    # make a comparison ground truth image
    gndTruth = dl.copy()
    gndTruth = skimage.color.rgb2gray(gndTruth)

    dl = skimage.color.gray2rgb(dl, 1)

    fname_truth = "true_id" + filePostfix + ".png"
    fname_recon = "recon_id" + filePostfix + ".png"

    if not os.path.isdir('data/train/id'+filePostfix):
        os.makedirs('data/train/id'+filePostfix, mode=0o777)
    os.chdir('data/train/id'+filePostfix)

    if not os.path.isdir('recon/'):
        os.mkdir('recon/',mode=0o777)
    os.chdir('recon/')
    skimage.io.imsave(fname_recon,dl)
    os.chdir('..')

    if not os.path.isdir('true_pt/'):
        os.mkdir('true_pt/',mode=0o777)
    os.chdir('true_pt/')
    skimage.io.imsave(fname_truth,ptSrc)
    os.chdir('../../../..')

    return

#########################
    

# example of calling for one source location
#simulateOneSource(2.0,7.3,3,'test',1)

# RUN SIMULATOR TO MAKE A BUNCH OF INPUT/OUTPUT IMAGE PAIRS
numPix = 1 # CHANGE ME!!

for runNumber in range(numPix):

    
    print("running realization number " + str(runNumber))
    # simulateOneSource(xs[0],zs[0],amp[0],str(runNumber),1)
    simulateOneSource(str(runNumber),1)


