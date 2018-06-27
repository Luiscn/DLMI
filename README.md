# DLMI
Simulated data: please decompress all ZIPs. (GitHub allows only 100 folders in the repo...) 

'/bubble' is the main working directory\

'/bubble/main.py': main function of the U-Net image reconstruction implementation

'/bubble/data_main.py': main function of the brain data generation - [70,80] point sources in each img. (data is generated already - '/bubble/input')\
'/bubble/data_main_pht.py': data generation implementation for resolution phantoms.

'/bubble/arrayUtils.py': sub-functions of the brain data generation

'/bubble/unet_lin.py': unet structures with linear output layer activation function.

model_20: trained from img with less point sources.\
model_24: trained from img with more point sources.\
all files ending in '.h5' are the trained model
