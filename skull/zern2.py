from keras.models import Model
from keras.layers import Input, Dropout, Dense, Permute
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose, Conv3D
from keras.layers.pooling import MaxPooling2D, MaxPooling3D, AveragePooling3D
from keras.layers.merge import concatenate
from keras import backend as K

# Build U-Net model
def net3(V_1, V_2, V_3):
    inputs = Input((V_1, V_2, V_3, 1))
    
    p1 = MaxPooling3D((2, 2, 1)) (inputs)
    
    c1 = Conv3D(64, (2, 2, 1), strides=(2,2,1), activation='relu', padding='same') (p1)

    c1 = MaxPooling3D((2, 2, 1)) (c1)
    
    c2 = Conv3D(512, (2, 2, 1), strides=(1,1,1), activation='relu', padding='same') (c1)

    c2 = Conv3D(512, (2, 2, 1), strides=(2,2,1), activation='relu', padding='same') (c2)
    
    c3 = MaxPooling3D((2, 2, 3)) (c2)
    
    c4 = Dense(128, activation='linear')(c3)
    c5 = Dense(56, activation='linear')(c4)
    
    outputs = c5

    return Model(inputs=[inputs], outputs=[outputs])


numTrain = 993
V_LENGTH = 3
V_WIDTH = 44
V_HEIGHT = 44
model=net3(V_WIDTH,V_HEIGHT,V_LENGTH)
model.summary()

