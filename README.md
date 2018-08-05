# DLMI
Simulated data: please decompress all ZIPs. (GitHub allows only 100 folders in the repo...) 

'/bubble' is the main working directory

'/bubble/input/' training and test set included. [1,10] point sources in each img.\
'/bubble/input_more/' training and test set included. [70,80] point sources in each img.\
'/bubble/input_ph/' training and test set included. Aprox. [70,80] point sources in each img, basically bubble clouds and lines.

'/bubble/main.py': main function of the U-Net image reconstruction implementation

'/bubble/data_main.py': main function of the brain data generation - [70,80] point sources in each img. (data is generated already - '/bubble/input')\
'/bubble/data_main_pht.py': data generation implementation for resolution phantoms.

'/bubble/arrayUtils.py': sub-functions of the brain data generation

'/bubble/unet_lin.py': unet structures with linear output layer activation function.

Uses model_29\
all files ending in '.h5' are the trained model
