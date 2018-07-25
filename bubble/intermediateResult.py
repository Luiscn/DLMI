output_fn = K.function([model.layers[0].input, K.learning_phase()], [model.layers[40].output])

input_img = X_test[0,:,:,:]
#print(input_img.shape)
#skimage.io.imshow(input_img)
#skimage.io.show()

output_img = output_fn([input_img.reshape(1,128,128,3),1])[0]
print(output_img.shape)

#fig = plt.figure(figsize=(8, 8))
#for i in range(1):
#	ax = fig.add_subplot(1,1,i+1)
#	ax.imshow(output_img[0,:,:,i])
#	plt.xticks(np.array([]))
#	plt.yticks(np.array([]))
#	plt.tight_layout()
#
#plt
#

