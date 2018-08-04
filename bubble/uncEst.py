out_mean = np.mean(out,axis = 2)
stdVar = np.sqrt(var)
t = np.linspace(54,74,20)

def lin2dec60(x,ref):
    return 10*np.log10(x/ref+0.000001)

#real_y = 64
#ref = np.max(out_mean[real_y,:])
#fig = plt.figure()
#ax = plt.subplot(111)
#ax.plot(t,out_mean[real_y,54:74],label='mean')
#ax.plot(t,out_mean[real_y,54:74]-stdVar[real_y,54:74],label='mean + std deviation')
#ax.plot(t,out_mean[real_y,54:74]+stdVar[real_y,54:74],label='mean - std deviation')
#plt.grid(True)
#ax.legend(loc='lower center')
#plt.savefig('./result/mean+sV.png', dpi=800, facecolor='w', edgecolor='w',
#        orientation='portrait', papertype=None, format=None,
#        transparent=False, bbox_inches=None, pad_inches=0.1,
#        frameon=None)
#plt.show()

real_y = 64
ref = np.max(out_mean[real_y,:])
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(t,np.percentile(out[real_y,54:74,:],75,axis = 1),label='75th percentile')
ax.plot(t,np.median(out[real_y,54:74,:],axis = 1),label='median')
ax.plot(t,np.percentile(out[real_y,54:74,:],25,axis = 1),label='25th percentile')
plt.xlabel('positions (pixel)')
plt.ylabel('linear magnitude')
plt.grid(True)
ax.legend(loc='upper left')
plt.savefig('./result/mean+sV.png', dpi=800, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)
plt.show()

#fake_y = 56
#fig = plt.figure()
#ax = plt.subplot(111)
#ax.plot(t,out_mean[fake_y,54:74],label='mean')
#ax.plot(t,out_mean[fake_y,54:74]+stdVar[fake_y,54:74],label='mean + std deviation')
#ax.plot(t,out_mean[fake_y,54:74]-stdVar[fake_y,54:74],label='mean - std deviation')
#plt.grid(True)
#ax.legend()
#plt.savefig('./result/mean+sV_fake.png', dpi=800, facecolor='w', edgecolor='w',
#        orientation='portrait', papertype=None, format=None,
#        transparent=False, bbox_inches=None, pad_inches=0.1,
#        frameon=None)
#plt.show()

fake_y = 72
ref = np.max(out_mean[real_y,:])
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(t,np.percentile(out[fake_y,54:74,:],75,axis = 1),label='75th percentile')
ax.plot(t,np.median(out[fake_y,54:74,:],axis = 1),label='median')
ax.plot(t,np.percentile(out[fake_y,54:74,:],25,axis = 1),label='25th percentile')
plt.xlabel('positions (pixel)')
plt.ylabel('linear magnitude')
plt.grid(True)
ax.legend(loc='upper left')
plt.savefig('./result/mean+sV_fake.png', dpi=800, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)
plt.show()