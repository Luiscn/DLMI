# Perform a sanity check on some random testing samples
ix = 0
skimage.io.imshow(X_test[ix]) # original image to be reconstructed
skimage.io.show()
skimage.io.imshow(Y_test[ix,:,:,0]) # ground truth of the ix-th sample
skimage.io.show()

preds_test_test = preds_test[ix,:,:,0] # scores of the ix-th sample's prediction
#preds_test_test=imlinmap(preds_test_test,[np.min(preds_test_test), np.max(preds_test_test)], [0,1])
preds_test_test=imlinmap(preds_test_test,[np.percentile(preds_test_test, 0.5), 255], [0,1])
skimage.io.imshow(preds_test_test)
skimage.io.show()

X_test_test=X_test[ix,:,:,:]
X_test_test=10*np.log10(X_test_test/np.max(X_test_test) +0.001).astype(np.float32)
#X_train_test=imlinmap(X_test_test,[np.min(X_train_test), np.max(X_test_test)], [0, 1]).astype(np.float32)
X_test_test = imlinmap(X_test_test, [-40, 0], [0, 1])
skimage.io.imshow(X_test_test)
skimage.io.show()

score=preds_test_test
#score=imlinmap(score,[np.percentile(score, 0.5), np.percentile(score, 100)],[np.min(score),np.max(score)])
score=10*np.log10(score/np.max(score) +0.001)
#score=imlinmap(score, [np.min(score), np.max(score)], [0,1]).astype(np.float32)
score = imlinmap(score, [-40, 0], [0, 1]).astype(np.float32)
skimage.io.imshow(score)
skimage.io.show()

skimage.io.imsave('./result/te-in_lin.png',X_test[ix])
skimage.io.imsave('./result/te-gndTruth.png',Y_test[ix,:,:,0])
skimage.io.imsave('./result/te-out_lin.png',preds_test_test)
skimage.io.imsave('./result/te-in_log(-40to0dB).png',X_test_test)
skimage.io.imsave('./result/te-out_log(-40to0dB).png',score)

res = resultCompare(score, Y_test[ix,:,:,0])
skimage.io.imshow(res)
skimage.io.imsave('./result/resultCompare_te.png',res)