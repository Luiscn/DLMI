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

def simulateOneSource(xs_mm,zs_mm,ampSrc,filePostfix,DOPLOTS=1):
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
    
    
    
    # simulate recieved data
    delay_ms = bf.getTimeDelays(xs_mm,zs_mm,xArray_mm,zArray_mm)
    
    signals = bf.simulateSignal(FS,delay_ms,TCAPTURE_MS)    
    data = ampSrc * bf.addNoiseToSignal(signals,SNR_LIN)
    
    if 0*DOPLOTS:
        plt.figure()
        plt.imshow(data)
        plt.title('Simulated data')
        plt.show()
    
    ########### IMAGE FORMATION ###########
    [t1,tst] = bf.singleLocationDAS(data,FS,delay_ms,AVGWIN_MS)
    
    xgrid_mm = np.linspace(-5,5,128)
    zgrid_mm = np.linspace(2,12,128)   
    
    das = bf.formDASimage(data,xgrid_mm,zgrid_mm,FS,AVGWIN_MS,xArray_mm,zArray_mm)
    das = das.T
    
    dl = bf.normalizeImage(das)  # changes to dB

    dl = bf.imlinmap(dl, [0, 1], [0, 255]).astype('uint8')
    # make a comparison ground truth image
    gndTruth = dl.copy()
    gndTruth = skimage.color.rgb2gray(gndTruth)
    powerThreshold = 0.5 # CHANGE ME set to 0.5 for halfphower
    lowVals = gndTruth < np.max(dl)*powerThreshold
    gndTruth[lowVals]=0
    highVals = gndTruth >= np.max(dl)*powerThreshold
    gndTruth[highVals]=255
    # gndTruth = bf.imlinmap(gndTruth, [0, 1], [0, 1])

    dl = skimage.color.gray2rgb(dl, 1)
    
    if 0*DOPLOTS:
        skimage.io.imshow(dl)
        skimage.io.show()
        skimage.io.imshow(gndTruth)
        skimage.io.show()

    fname_truth = "true_id" + filePostfix + ".png"
    fname_recon = "recon_id" + filePostfix + ".png"

    if not os.path.isdir('input/train/id'+filePostfix):
        os.makedirs('input/train/id'+filePostfix, mode=0o777)
    os.chdir('input/train/id'+filePostfix)

    if not os.path.isdir('recon/'):
        os.mkdir('recon/',mode=0o777)
    os.chdir('recon/')
    skimage.io.imsave(fname_recon,dl)
    # img=Image.fromarray(dl)
    # img.save(fname_recon)
    # mp.pyplot.imsave(fname_recon,dl,vmin=-30,vmax=0,cmap='gray')
    os.chdir('..')

    if not os.path.isdir('true/'):
        os.mkdir('true/',mode=0o777)
    os.chdir('true/')
    skimage.io.imsave(fname_truth,gndTruth)
    os.chdir('../../../..')

    return

#########################
    

# example of calling for one source location
#simulateOneSource(2.0,7.3,3,'test',1)

# RUN SIMULATOR TO MAKE A BUNCH OF INPUT/OUTPUT IMAGE PAIRS
numPix = 500 # CHANGE ME!!

for runNumber in range(numPix):
    xs = -5.0 + 10. * np.random.rand(1) # source x between -4 to 5
    zs = 2.0 + 10. * np.random.rand(1) # source z between 2 to 12
    amp = 1.0 + 3. * np.random.rand(1)
    
    print("running realization number " + str(runNumber))
    # simulateOneSource(xs[0],zs[0],amp[0],str(runNumber),1)
    simulateOneSource(xs[0],zs[0],1,str(runNumber),1)


