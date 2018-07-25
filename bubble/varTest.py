out = np.zeros([128,128,100])
for i in range(100):
    runfile('intermediateResult.py', wdir='/Users/luis/Documents/MedImag/bubble')
    out[:,:,i] = output_img[0,:,:,0]

var = np.var(out, axis=2)
var_log=10*np.log10(var/np.max(var) +0.001)
var_1 = imlinmap(var,[np.min(var), np.max(var)],[0,1])
var_2 = imlinmap(var_log,[-40, 0],[0,1])

skimage.io.imshow(var_1)
skimage.io.show()

skimage.io.imshow(var_2)
skimage.io.imsave('./result/UQ.png',var_2)
skimage.io.show()

out_pool = np.zeros([32,32,100])
for i in range(100):
    out_pool[:,:,i] = rescale(imlinmap(out[:,:,i],[0, np.max(out)],[0,1]),1/4)

vmax = np.amax(np.clip(out_pool,0,None),axis=2)
vmin = np.amin(np.clip(out_pool,0,None),axis=2)
vecMax = np.reshape(vmax,(1,-1))
vecMin = np.reshape(vmin,(1,-1))
vecMin = np.squeeze(vecMin)
vecMax = np.squeeze(vecMax)
t = np.arange(len(vecMax))

plt.plot(t,10*np.log10(vecMin/np.max(vecMax)),'b',
         t,10*np.log10(vecMax/np.max(vecMax)),'r')
plt.savefig('./result/ff.png', dpi=800, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)

secL = 776
secU = 825

plt.plot(t[secL:secU],10*np.log10(vecMin[secL:secU]/np.max(vecMax[secL:secU])),'b',
         t[secL:secU],10*np.log10(vecMax[secL:secU]/np.max(vecMax[secL:secU])),'r')
plt.savefig('./result/f.png', dpi=800, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)