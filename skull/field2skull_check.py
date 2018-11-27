# check if close fields corresponds to close skulls

# good: 
# medium: 
# bad: 1
test_sample = 0

colorbar_min_mag, colorbar_max_mag = np.min(label_mag[:,:,test_sample+900]), np.max(label_mag[:,:,test_sample+900])

fig = plt.figure(figsize=(10, 10))
plt.subplot(331)
plt.imshow(label_mag[:,:,test_sample+900])
pcm = plt.pcolor(label_mag[:,:,test_sample+900],norm = mpl.colors.Normalize(vmin=colorbar_min_mag* .98, vmax=colorbar_max_mag* 1.02))
plt.colorbar(pcm)
plt.title('true')
plt.subplot(332)
plt.imshow(label_mag[:,:,int(ind_f[test_sample,:][0])])
pcm = plt.pcolor(label_mag[:,:,int(ind_f[test_sample,:][0])],
               norm = mpl.colors.Normalize(vmin=colorbar_min_mag* .98, vmax=colorbar_max_mag* 1.02))
plt.colorbar(pcm)
plt.title('1-st NN')
plt.subplot(333)
plt.imshow(label_mag[:,:,int(ind_f[test_sample,:][1])])
pcm = plt.pcolor(label_mag[:,:,int(ind_f[test_sample,:][1])],
               norm = mpl.colors.Normalize(vmin=colorbar_min_mag* .98, vmax=colorbar_max_mag* 1.02))
plt.colorbar(pcm)
plt.title('2-rd NN')
plt.subplot(334)
plt.imshow(data_b[test_sample+900,22,:,:])
plt.title('true')
plt.subplot(335)
plt.imshow(data_b[int(ind_f[test_sample,:][0]),22,:,:])
plt.title('1-st NN')
plt.subplot(336)
plt.imshow(data_b[int(ind_f[test_sample,:][1]),22,:,:])
plt.title('2-rd NN')
plt.subplot(337)
plt.imshow(data_b[test_sample+900,:,22,:])
plt.title('true')
plt.subplot(338)
plt.imshow(data_b[int(ind_f[test_sample,:][0]),:,22,:])
plt.title('1-st NN')
plt.subplot(339)
plt.imshow(data_b[int(ind_f[test_sample,:][1]),:,22,:])
plt.title('2-rd NN')
plt.show()

fig = plt.figure(figsize=(10, 4))
plt.subplot(131)
plt.imshow(data_b[test_sample+900,:,:,25])
plt.title('true')
plt.subplot(132)
plt.imshow(data_b[int(ind_f[test_sample,:][0]),:,:,25])
plt.title('1-st NN')
plt.subplot(133)
plt.imshow(data_b[int(ind_f[test_sample,:][1]),:,:,25])
plt.title('2-rd NN')
plt.show()