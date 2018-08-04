t = np.linspace(38,90,52)

#plt.plot(rgb2gray(X_test_test[6,0:128,]))
#plt.show()
#plt.plot(score[6,0:128])
#plt.show()
#
#plt.plot(rgb2gray(X_test_test[35,0:128,]))
#plt.show()
#plt.plot(score[35,0:128])
#plt.show()
#
#plt.plot(rgb2gray(X_test_test[64,0:128,]))
#plt.savefig('./result/lat3_1.png', dpi=800, facecolor='w', edgecolor='w',
#        orientation='portrait', papertype=None, format=None,
#        transparent=False, bbox_inches=None, pad_inches=0.1,
#        frameon=None)
#plt.show()
#plt.plot(score[64,0:128])
#plt.savefig('./result/lat3_2.png', dpi=800, facecolor='w', edgecolor='w',
#        orientation='portrait', papertype=None, format=None,
#        transparent=False, bbox_inches=None, pad_inches=0.1,
#        frameon=None)
#plt.show()
#
#plt.plot(rgb2gray(X_test_test[93,0:128,]))
#plt.show()
#plt.plot(score[93,0:128])
#plt.show()
#
#plt.plot(rgb2gray(X_test_test[123,0:128,]))
#plt.show()
#plt.plot(score[123,0:128])
#plt.show()

X_gray = rgb2gray(X_test_test)
X_test_rec = imlinmap(X_gray, [0, 1], [-40, 0]).astype(np.float32)
score_rec = imlinmap(score, [0, 1], [-40, 0]).astype(np.float32)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(t,X_test_rec[64,38:90,],label='original image')
ax.plot(t,score_rec[64,38:90],label='reconstructed image')
plt.ylabel('magnitude (dB)')
plt.xlabel('positions (pixel)')
plt.grid(True)
ax.legend(loc='upper left')
plt.savefig('./result/lat3.png', dpi=800, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)
plt.show()