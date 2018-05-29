# DLMI
Please decompress '/bubble/input.zip', '/kaggle_unet_bubble/input.zip' and '/kaggle_unet_bubble/input2.zip/'. (GitHub allows only 100 folders in the repo...) 

'/bubble' is my main working directory\
'/kaggle_unet_bubble' is a directory I used to compare results from two different datasets with same codes

'/bubble/main.py': main function of the U-Net image reconstruction implementation\
'/bubble/data_main.py': main function of the brain data generation (data is generated already - '/bubble/input')\
'/bubble/arrayUtils.py': sub-functions of the brain data generation\
'/bubble/unet.py': unet structure

'/kaggle_unet_bubble/unet': implementation for our project\
'/kaggle_unet_bubble/unet2': implementation for that image segmentation project\
(this 2 files are basically the same (except for the data path), for comparison purpose)

all files ending in '.h5' are the trained model
