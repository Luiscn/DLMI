# -*- coding: utf-8 -*-
"""
Delay and sum calculations

"""
import numpy as np

def defineArray(numEl=64,cenFreq=5e6,c=1500):
    # compute positions of 1-D array at lambda/2 spacing
    
    # find array spacing
    c = 1500;
    wavelen = c / cenFreq
    dx = wavelen/2;
    
    # get array dimensions in x and z
    x = np.linspace(0,dx*numEl,numEl);
    x -= np.mean(x);
    z = np.zeros(x.shape)
    
    xArray_mm = x*1000
    zArray_mm = z*1000
   
    return(xArray_mm,zArray_mm)

    
def getTimeDelays(xSrc_mm,zSrc_mm,xArray_mm,zArray_mm,c=1500):
    # compute time delsys from (xSrc,ySrc) to 
    # 1-d array of elements at positions xArray and zArray
    delx = (xArray_mm - xSrc_mm)/1000
    delz = (zArray_mm - zSrc_mm)/1000
    d = np.sqrt(delx**2 + delz**2)
    
    delaysMs = d / c * 1000
    
    return(delaysMs)
    
def gausPuls(fSamp,pulseDurMs,tStartMs=0,fCen=5e6):
 
    nsamp = np.floor(pulseDurMs/1000*fSamp);
    #print(nsamp)
    t = np.linspace(0,pulseDurMs*1000,nsamp)
    t += tStartMs*1000;
    tone = np.cos(2*np.pi*fCen*t)
    
    puls = np.hanning(nsamp)*tone;
    tMs = t /1000;
    
    return(puls,tMs)
    
def simulateSignal(fSamp,delaysMs,tCaptureMs):
    
    # first dimension of delays should be channels
    nchan = delaysMs.shape[0];
    pulseLen_ms = 0.002 # CONSTANT
    
    nsamp = np.floor(tCaptureMs/1000*fSamp);
    nsamp = int(nsamp)

    tMs = np.linspace(0,tCaptureMs,nsamp)

    signals = np.zeros([nchan,nsamp])
    for ichan in range(nchan):
        thisPuls,thisTms = gausPuls(fSamp,pulseLen_ms)
        thisTms += delaysMs[ichan]
        #print(thisPuls.size)
        signals[ichan,:] = np.interp(tMs,thisTms,thisPuls);
        
    
    return(signals)
    
def addNoiseToSignal(signals,snrLin):
    
    # crude SNR calculator
    mxSig = np.max(signals)
    mxNoise = mxSig/snrLin
    
    noise = mxNoise * np.random.randn(signals.shape[0],signals.shape[1])
    
    data = signals + noise;
  
    return(data)
    
def singleLocationDAS(data,fSamp,estDelaysMs,avgWinMs):
    # do time-alignment for single position
  
    # first dimension of delays should be channels
    nchan = estDelaysMs.shape[0];

    # get time vector for averaging window 
    nsampAvg = np.floor(avgWinMs/1000*fSamp);
    nsampAvg = int(nsampAvg)
    tMs = np.linspace(0,avgWinMs,nsampAvg)
    windowDAS = np.zeros([nchan,nsampAvg])
  
    # get time vector for data
    nsampData = data.shape[1];
    tDataMs = np.linspace(0,nsampData/fSamp*1000,nsampData)
   
    for ichan in range(nchan):
        tThisAvgWIn = tMs + estDelaysMs[ichan]
        windowDAS[ichan,:] = np.interp(tThisAvgWIn,tDataMs,data[ichan,:])

    # sum over elements
    t1=np.sum(windowDAS,axis=0)
    dasOutput = np.dot(t1,t1) # amp square 
    
    return dasOutput, windowDAS

def formDASimage(data,xgrid_mm,zgrid_mm,fSamp,avgWin_ms,xArray_mm,zArray_mm):
    # do beamforming on imaging grid points
    
    nx = xgrid_mm.size
    nz = zgrid_mm.size
    
    dasImage = np.zeros((nx,nz), dtype='float')
    
    for ix in range(nx):
        for iz in  range(nz):
            xs = xgrid_mm[ix]
            zs = zgrid_mm[iz]
            delay_ms = getTimeDelays(xs,zs,xArray_mm,zArray_mm)
            dasP,jnk = singleLocationDAS(data,fSamp,delay_ms,avgWin_ms)
            dasImage[ix,iz] = dasP  
    return dasImage

def normalizeImage(imageMatrix):
    # generate nomralized 60 dB image
    img = np.abs(imageMatrix)
    img = img/np.max(img)
    #img = 10*np.log10(img)
    
    return(img)
    
def imlinmap(im, limIn, limOut):
    ratio=(limOut[1]-limOut[0])/(limIn[1]-limIn[0])
    im=im-limIn[0]
    im=im*ratio
    im=im+limOut[0]
    im=np.clip(im, limOut[0], limOut[1])
    return im

    
    

    
