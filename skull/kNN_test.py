# good: 40, 42, 52
# medium: 60, 35
# bad: 48, 66, 78, 72(bad at mag), 56(bad at phz)
test_sample = 48
fig = plt.figure(figsize=(8, 8))
plt.subplot(221)
plt.imshow(label_mag[:,:,test_sample+900])
colorbar_min_mag, colorbar_max_mag = np.min(label_mag[:,:,test_sample+900]), np.max(label_mag[:,:,test_sample+900])
pcm = plt.pcolor(label_mag[:,:,test_sample+900],
               norm = mpl.colors.Normalize(vmin=colorbar_min_mag* .9, vmax=colorbar_max_mag* 1.1))
plt.colorbar(pcm)
plt.title('mag_ture')
plt.subplot(222)
pred = np.mean(label_mag[:,:,[int(idx) for idx in ind[test_sample,:]]],axis=2)*2/3 + label_mag[:,:,int(ind[test_sample,1])]/3
plt.imshow(pred)
pcm = plt.pcolor(pred,
               norm = mpl.colors.Normalize(vmin=colorbar_min_mag* .9, vmax=colorbar_max_mag* 1.1))
plt.colorbar(pcm)
plt.title('mag_pred')
plt.subplot(223)
plt.imshow(phz_in[test_sample+900,:,:])
colorbar_min_phz, colorbar_max_phz = np.min(phz_in[test_sample+900,:,:]), np.max(phz_in[test_sample+900,:,:])
pcm = plt.pcolor(phz_in[test_sample+900,:,:],
               norm = mpl.colors.Normalize(vmin=colorbar_min_phz* .98, vmax=colorbar_max_phz* 1.02))
plt.colorbar(pcm)
plt.title('phz_ture')
plt.subplot(224)
pred = np.mean(phz_in[[int(idx) for idx in ind[test_sample,:]],:,:],axis=0)*2/3 + phz_in[int(ind[test_sample,1]),:,:]/3
plt.imshow(pred)
pcm = plt.pcolor(pred,
               norm = mpl.colors.Normalize(vmin=colorbar_min_phz* .98, vmax=colorbar_max_phz* 1.02))
plt.colorbar(pcm)
plt.title('phz_pred')
plt.tight_layout()
#pname = 'p' + str(test_sample) + '.png'
#plt.savefig('./pic/' + pname)
plt.show() 

fig = plt.figure(figsize=(10, 10))
plt.subplot(331)
plt.imshow(data_b[test_sample+900,:,22,:])
plt.title('true')
plt.subplot(332)
plt.imshow(data_b[int(ind[test_sample,:][0]),:,22,:])
plt.title('1-st NN')
plt.subplot(333)
plt.imshow(data_b[int(ind[test_sample,:][1]),:,22,:])
plt.title('2-rd NN')
plt.subplot(334)
plt.imshow(label_mag[:,:,test_sample+900])
pcm = plt.pcolor(label_mag[:,:,test_sample+900],
               norm = mpl.colors.Normalize(vmin=colorbar_min_mag* .98, vmax=colorbar_max_mag* 1.02))
plt.colorbar(pcm)
plt.title('true')
plt.subplot(335)
plt.imshow(label_mag[:,:,int(ind[test_sample,:][0])])
pcm = plt.pcolor(label_mag[:,:,int(ind[test_sample,:][0])],
               norm = mpl.colors.Normalize(vmin=colorbar_min_mag* .98, vmax=colorbar_max_mag* 1.02))
plt.colorbar(pcm)
plt.title('1-st NN')
plt.subplot(336)
plt.imshow(label_mag[:,:,int(ind[test_sample,:][1])])
pcm = plt.pcolor(label_mag[:,:,int(ind[test_sample,:][1])],
               norm = mpl.colors.Normalize(vmin=colorbar_min_mag* .98, vmax=colorbar_max_mag* 1.02))
plt.colorbar(pcm)
plt.title('2-rd NN')
plt.subplot(337)
plt.imshow(phz_in[test_sample+900,:,:])
pcm = plt.pcolor(phz_in[test_sample+900,:,:],
               norm = mpl.colors.Normalize(vmin=colorbar_min_phz* .98, vmax=colorbar_max_phz* 1.02))
plt.colorbar(pcm)
plt.title('true')
plt.subplot(338)
plt.imshow(phz_in[int(ind[test_sample,:][0]),:,:])
pcm = plt.pcolor(phz_in[int(ind[test_sample,:][0]),:,:],
               norm = mpl.colors.Normalize(vmin=colorbar_min_phz* .98, vmax=colorbar_max_phz* 1.02))
plt.colorbar(pcm)
plt.title('1-st NN')
plt.subplot(339)
plt.imshow(phz_in[int(ind[test_sample,:][1]),:,:])
pcm = plt.pcolor(phz_in[int(ind[test_sample,:][1]),:,:],
               norm = mpl.colors.Normalize(vmin=colorbar_min_phz* .98, vmax=colorbar_max_phz* 1.02))
plt.colorbar(pcm)
plt.title('2-rd NN')
plt.show()