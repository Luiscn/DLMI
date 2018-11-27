from numpy.linalg import norm

def cos_corr(in1, in2):
    in1 = np.reshape(in1, (1,-1))
    in2 = np.reshape(in2, (-1,1))
    return np.abs(np.dot(in1,np.conj(in2))) / (norm(in1)*norm(in2))

percError_noskull = np.zeros(93)
phz_Error = np.zeros(93)
cos_corr_s_noskull = np.zeros(93)
percError = np.zeros(93)
cos_corr_s = np.zeros(93)

for test_sample in range(93):
    
    phz_pred = np.mean(phz_in[[int(idx) for idx in ind[test_sample,:]],:,:],axis=0)*2/3 + phz_in[int(ind[test_sample,1]),:,:]/3
    mag_pred = np.mean(label_mag[:,:,[int(idx) for idx in ind[test_sample,:]]],axis=2)*2/3 + label_mag[:,:,int(ind[test_sample,1])]/3  
    fieldOrig = label_mag[:,:,test_sample+900]*np.exp(1j*phz_in[test_sample+900,:,:])
    
    fieldPred_noskull = mag_nsk[:,:,458]*np.exp(1j*phz_nsk[:,:,458]) # no-effort preds
#    fieldPred = mag_pred*np.exp(1j*phz_pred)
    
    percError_noskull[test_sample] = np.sum(np.abs(fieldOrig-fieldPred_noskull**2) / np.sum(np.abs(fieldOrig**2)))
    phz_Error[test_sample] = np.mean(phz_pred) - np.mean(phz_in[test_sample+900,:,:])
    
    cos_corr_s_noskull[test_sample]  = cos_corr(fieldPred_noskull, fieldOrig)
     

for test_sample in range(93):
    
    phz_pred = np.mean(phz_in[[int(idx) for idx in ind[test_sample,:]],:,:],axis=0)*2/3 + phz_in[int(ind[test_sample,1]),:,:]/3
    mag_pred = np.mean(label_mag[:,:,[int(idx) for idx in ind[test_sample,:]]],axis=2)*2/3 + label_mag[:,:,int(ind[test_sample,1])]/3  
    fieldOrig = label_mag[:,:,test_sample+900]*np.exp(1j*phz_in[test_sample+900,:,:])
    
#    fieldPred = mag_nsk[:,:,459]*np.exp(1j*phz_nsk[:,:,459]) # no-effort preds
    fieldPred = mag_pred*np.exp(1j*phz_pred)
    
    percError[test_sample] = np.sum(np.abs(fieldOrig-fieldPred**2) / np.sum(np.abs(fieldOrig**2)))
    phz_Error[test_sample] = np.mean(phz_pred) - np.mean(phz_in[test_sample+900,:,:])
    
    cos_corr_s[test_sample]  = cos_corr(fieldPred, fieldOrig)
    
skull_diff_std = 2/3 * skull_diff[:,0] + 1/3 * skull_diff[:,1]
plt.plot(skull_diff_std) 
plt.title('skull_diff_std')
plt.show() 
  
plt.plot(percError)
plt.plot(percError_noskull)
plt.gca().legend(('knn vs true','no-skull vs true'))
plt.title('percentage error, 93 test samples')
plt.show()

plt.plot(cos_corr_s)
plt.plot(cos_corr_s_noskull)
plt.gca().legend(('knn vs true','no-skull vs true'))
plt.title('cosine correlation, 93 test samples')
plt.show()

plt.scatter(skull_diff_std, percError)
plt.title('error relation')
plt.show()


plt.scatter(skull_diff_std, percError_noskull)
plt.title('error relation')
plt.show()

plt.plot(phz_Error %  (2*np.pi))
plt.title('averaged phase diff (< 2 pi)')
plt.show()

#np.corrcoef(skull_diff_std, percError)

