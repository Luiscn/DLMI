for run in range(0,80,2):    
    test_point = run
    fig = plt.figure(figsize=(8, 8))
    plt.subplot(221)
    plt.imshow(label_mag[:,:,test_point+900])
    colorbar_min, colorbar_max = np.min(label_mag[:,:,test_point+900]), np.max(label_mag[:,:,test_point+900])
    pcm = plt.pcolor(label_mag[:,:,test_point+900],
                   norm = mpl.colors.Normalize(vmin=colorbar_min* .9, vmax=colorbar_max* 1.1))
    plt.colorbar(pcm)
    plt.title('mag_ture')
    plt.subplot(222)
    pred = np.mean(label_mag[:,:,[int(idx) for idx in ind[test_point,:]]],axis=2)*2/3 + label_mag[:,:,int(ind[test_point,1])]/3
    plt.imshow(pred)
    pcm = plt.pcolor(pred,
                   norm = mpl.colors.Normalize(vmin=colorbar_min* .91, vmax=colorbar_max* 1.1))
    plt.colorbar(pcm)
    plt.title('mag_pred')
    plt.subplot(223)
    plt.imshow(phz_in[test_point+900,:,:])
    colorbar_min, colorbar_max = np.min(phz_in[test_point+900,:,:]), np.max(phz_in[test_point+900,:,:])
    pcm = plt.pcolor(phz_in[test_point+900,:,:],
                   norm = mpl.colors.Normalize(vmin=colorbar_min* .98, vmax=colorbar_max* 1.02))
    plt.colorbar(pcm)
    plt.title('phz_ture')
    plt.subplot(224)
    pred = np.mean(phz_in[[int(idx) for idx in ind[test_point,:]],:,:],axis=0)*2/3 + phz_in[int(ind[test_point,1]),:,:]/3
    plt.imshow(pred)
    pcm = plt.pcolor(pred,
                   norm = mpl.colors.Normalize(vmin=colorbar_min* .98, vmax=colorbar_max* 1.02))
    plt.colorbar(pcm)
    plt.title('phz_pred')
    plt.tight_layout()
    pname = 'p' + str(run) + '.png'
    plt.savefig('./pic/' + pname)
    plt.show() 
    

#test_point = 40
#fig, (ax1,ax2) = plt.subplots(1, 2)
#z1_plot = ax1.imshow(label_mag[:,:,test_point+900])
#plt.title('mag_ture')
#fig.colorbar(z1_plot,ax=ax1)
#pred = np.mean(label_mag[:,:,[int(idx) for idx in ind[test_point,:]]],axis=2)
#z2_plot = ax2.imshow(pred)
#plt.title('mag_pred')
#fig.colorbar(z2_plot,ax=ax2)
#plt.tight_layout()
#plt.show() 
#
#fig, (ax3,ax4) = plt.subplots(1, 2)
#z3_plot = ax3.imshow(phz_in[test_point+900,:,:])
#plt.title('phz_ture')
#fig.colorbar(z3_plot,ax=ax3)
#pred = np.mean(phz_in[[int(idx) for idx in ind[test_point,:]],:,:],axis=0)
#z4_plot = ax4.imshow(pred)
#plt.title('phz_pred')
#fig.colorbar(z4_plot,ax=ax4)
#plt.tight_layout()
#plt.show()