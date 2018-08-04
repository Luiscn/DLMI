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
    
def gausPuls(fSamp,pulseDurMs,tStartMs=0,fCen=0.5e6):
 
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

def fnPht(xgrid_mm, zgrid_mm, xArray_mm,zArray_mm,FS, TCAPTURE_MS, SNR_LIN):
    # fake node phantoms
    ptSrc = np.zeros((xgrid_mm.size,zgrid_mm.size))
    data_sp = 0
    xs = 8 * (np.random.rand(1) - 0.5)
    zs = 3 + np.random.rand(1) * 8
    x_spacing = 10/128 * (3 * np.random.rand(1)+2)
    z_spacing = 10/128 * (3 * np.random.rand(1)+2)
    
    for i in [-1, 1]:
        xs_mm = xs + i * x_spacing
        zs_mm = zs
       
        # simulate recieved data
        delay_ms = getTimeDelays(xs_mm,zs_mm,xArray_mm,zArray_mm)
        signals = simulateSignal(FS,delay_ms,TCAPTURE_MS)
        data = addNoiseToSignal(signals,SNR_LIN)
        data_sp = data_sp + data
        
        # make a comparison ground truth image
        xBest = np.argmin(np.abs(xgrid_mm-xs_mm))
        zBest = np.argmin(np.abs(zgrid_mm-zs_mm))
        ptSrc[zBest,xBest] = 1
    
    for j in [-1, 0, 1]:
        xs_mm = xs
        zs_mm = zs + j * z_spacing
       
        # simulate recieved data
        delay_ms = getTimeDelays(xs_mm,zs_mm,xArray_mm,zArray_mm)
        signals = simulateSignal(FS,delay_ms,TCAPTURE_MS)
        data = addNoiseToSignal(signals,SNR_LIN)
        data_sp = data_sp + data
        
        # make a comparison ground truth image
        xBest = np.argmin(np.abs(xgrid_mm-xs_mm))
        zBest = np.argmin(np.abs(zgrid_mm-zs_mm))
        ptSrc[zBest,xBest] = 1
    
    
    return data_sp, ptSrc

def latResPht(xgrid_mm, zgrid_mm, xArray_mm,zArray_mm,FS, TCAPTURE_MS, SNR_LIN):
    # lateral resolution pht
    ptSrc = np.zeros((xgrid_mm.size,zgrid_mm.size))
    data_sp = 0
    zs_list = np.linspace(2.5, 11.5, 5)
    xspacing_mm = 10/128 * np.linspace(1,5,5)
    for i in range(1):
        for j in [-1, 1]:
            xs_mm = j*xspacing_mm[i+1+1+1+1]
            zs_mm = zs_list[2]
           
            # simulate recieved data
            delay_ms = getTimeDelays(xs_mm,zs_mm,xArray_mm,zArray_mm)
            signals = simulateSignal(FS,delay_ms,TCAPTURE_MS)
            data = addNoiseToSignal(signals,SNR_LIN)
            data_sp = data_sp + data
            
            # make a comparison ground truth image
            xBest = np.argmin(np.abs(xgrid_mm-xs_mm))
            zBest = np.argmin(np.abs(zgrid_mm-zs_mm))
            ptSrc[zBest,xBest] = 1
        
    return data_sp, ptSrc

def bCloud(center, cloudRange, numBubble, xgrid_mm, zgrid_mm, xArray_mm,zArray_mm,FS, TCAPTURE_MS, SNR_LIN):
    # bubble clouds
    data_sp = 0
    ptSrc = np.zeros((xgrid_mm.size,zgrid_mm.size))
    for bub in range(numBubble):
        xs_mm = center[0] + cloudRange * (np.random.rand(1)-0.5) # source x between -5 to 5
        zs_mm = center[1] + cloudRange * (np.random.rand(1)-0.5) # source z between 2 to 12
        ampSrc = 1.0 + 1. * np.random.rand(1)
    
        # simulate recieved data
        delay_ms = getTimeDelays(xs_mm,zs_mm,xArray_mm,zArray_mm)
        signals = simulateSignal(FS,delay_ms,TCAPTURE_MS)    
        data = ampSrc * addNoiseToSignal(signals,SNR_LIN)
        data_sp = data_sp + data
        
        # make a comparison ground truth image
        xBest = np.argmin(np.abs(xgrid_mm-xs_mm))
        zBest = np.argmin(np.abs(zgrid_mm-zs_mm))
        ptSrc[zBest,xBest] = ampSrc
     
#    ptSrc = ptSrc / np.max(ptSrc)
    
    return data_sp, ptSrc

def bLine(xgrid_mm, zgrid_mm, xArray_mm,zArray_mm,FS, TCAPTURE_MS, SNR_LIN):
    # bubble lines
    ptSrc = np.zeros((xgrid_mm.size,zgrid_mm.size))
    data_sp = 0
    num1 = 6 + np.random.randint(15) # 10 - 25
    space1 = 10 / num1
    spaceInit1 = space1 * np.random.rand(1)+2
    xs_mm = -4+3* np.random.rand(1) # source x between -5 to 5
    for bub in range(num1):
        zs_mm =  spaceInit1 + bub * space1 + 0.2*(np.random.rand(1)-0.5)# source z between 2 to 12
        ampSrc = 1.0 + 1. * np.random.rand(1)
    
        # simulate recieved data
        delay_ms = getTimeDelays(xs_mm,zs_mm,xArray_mm,zArray_mm)
        signals = simulateSignal(FS,delay_ms,TCAPTURE_MS)    
        data = ampSrc * addNoiseToSignal(signals,SNR_LIN)
        data_sp = data_sp + data
        
        # make a comparison ground truth image
        xBest = np.argmin(np.abs(xgrid_mm-xs_mm))
        zBest = np.argmin(np.abs(zgrid_mm-zs_mm))
        ptSrc[zBest,xBest] = ampSrc
    
    num2 = 6 + np.random.randint(15) # 5 - 25
    space2 = 10 / num2
    spaceInit2 = space2 * np.random.rand(1)+2
    xs_mm = 1+3* np.random.rand(1) # source x between -5 to 5
    for bub in range(num2):
        zs_mm =  spaceInit2 + bub * space2 +0.2*(np.random.rand(1)-0.5)# source z between 2 to 12
        ampSrc = 1.0 + 1. * np.random.rand(1)
    
        # simulate recieved data
        delay_ms = getTimeDelays(xs_mm,zs_mm,xArray_mm,zArray_mm)
        signals = simulateSignal(FS,delay_ms,TCAPTURE_MS)    
        data = ampSrc * addNoiseToSignal(signals,SNR_LIN)
        data_sp = data_sp + data
        
        # make a comparison ground truth image
        xBest = np.argmin(np.abs(xgrid_mm-xs_mm))
        zBest = np.argmin(np.abs(zgrid_mm-zs_mm))
        ptSrc[zBest,xBest] = ampSrc
        
    num3 = 6 + np.random.randint(15) # 10 - 25
    space3 = 10 / num3
    spaceInit3 = space3 * np.random.rand(1) -5
    zs_mm = 3+3* np.random.rand(1) # source z between 2 to 12
    for bub in range(num3):
        xs_mm =  spaceInit3 + bub * space3 +0.2*(np.random.rand(1)-0.5)# source x between -5 to 5
        ampSrc = 1.0 + 1. * np.random.rand(1)
    
        # simulate recieved data
        delay_ms = getTimeDelays(xs_mm,zs_mm,xArray_mm,zArray_mm)
        signals = simulateSignal(FS,delay_ms,TCAPTURE_MS)    
        data = ampSrc * addNoiseToSignal(signals,SNR_LIN)
        data_sp = data_sp + data
        
        # make a comparison ground truth image
        xBest = np.argmin(np.abs(xgrid_mm-xs_mm))
        zBest = np.argmin(np.abs(zgrid_mm-zs_mm))
        ptSrc[zBest,xBest] = ampSrc
    
    num4 = 6 + np.random.randint(15) # 10 - 25
    space4 = 10 / num4
    spaceInit4 = space4 * np.random.rand(1)-5
    zs_mm = 8+3* np.random.rand(1) # source z between 2 to 12
    for bub in range(num4):
        xs_mm =  spaceInit4 + bub * space4 +0.2*(np.random.rand(1)-0.5)# source x between -5 to 5
        ampSrc = 1.0 + 1. * np.random.rand(1)
    
        # simulate recieved data
        delay_ms = getTimeDelays(xs_mm,zs_mm,xArray_mm,zArray_mm)
        signals = simulateSignal(FS,delay_ms,TCAPTURE_MS)    
        data = ampSrc * addNoiseToSignal(signals,SNR_LIN)
        data_sp = data_sp + data
        
        # make a comparison ground truth image
        xBest = np.argmin(np.abs(xgrid_mm-xs_mm))
        zBest = np.argmin(np.abs(zgrid_mm-zs_mm))
        ptSrc[zBest,xBest] = ampSrc
    
    return data_sp, ptSrc

def resultCompare(out,truth):
    import skimage.io
    idx=truth!=0
    out=imlinmap(out,[0,np.max(out)],[0,1])
    truth=imlinmap(truth,[np.min(truth),np.max(truth)],[0,1])
    imgR=out.copy()
#    imgR[idx]=truth[idx]
    imgR[idx]=1
    imgGB=out.copy()
    imgGB[idx]=0
    show=np.zeros((truth.shape[0],truth.shape[1],3))
    show[:,:,0]=imgR
    show[:,:,1]=imgGB
    show[:,:,2]=imgGB
    skimage.io.imsave('./result/show.png',show)
    return show

def imlinmap(im, limIn, limOut):
    ratio=(limOut[1]-limOut[0])/(limIn[1]-limIn[0])
    im=im-limIn[0]
    im=im*ratio
    im=im+limOut[0]
    im=np.clip(im, limOut[0], limOut[1])
    return im

    
    

    