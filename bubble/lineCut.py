plt.plot(rgb2gray(X_test_test[6,0:128,]))
plt.show()
plt.plot(score[6,0:128])
plt.show()

plt.plot(rgb2gray(X_test_test[35,0:128,]))
plt.show()
plt.plot(score[35,0:128])
plt.show()

plt.plot(rgb2gray(X_test_test[63,0:128,]))
plt.savefig('./result/lat1_1.png', dpi=800, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)
plt.show()
plt.plot(score[63,0:128])
plt.savefig('./result/lat1_2.png', dpi=800, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)
plt.show()

plt.plot(rgb2gray(X_test_test[93,0:128,]))
plt.show()
plt.plot(score[93,0:128])
plt.show()

plt.plot(rgb2gray(X_test_test[123,0:128,]))
plt.show()
plt.plot(score[123,0:128])
plt.show()


