from keras.models import Model
from keras.layers import Input, Dropout
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose, Conv3D
from keras.layers.pooling import MaxPooling2D, MaxPooling3D, AveragePooling3D
from keras.layers.merge import concatenate
from keras import backend as K

# Build U-Net model
def net3(V_LENGTH, V_WIDTH, V_HEIGHT):
    inputs = Input((V_LENGTH, V_WIDTH, V_HEIGHT, 1))
    s = Lambda(lambda x: x / 12) (inputs) # thk_max is 12

    c1 = Conv3D(8, (4, 4, 4), strides=(1,1,1), activation='relu', padding='same') (s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv3D(8, (4, 4, 4), strides=(1,1,1), activation='relu', padding='same') (c1)
    p1 = MaxPooling3D((3, 3, 3)) (c1)

    c2 = Conv3D(16, (4, 4, 4), strides=(1,1,1), activation='relu', padding='same') (p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv3D(16, (4, 4, 4), strides=(1,1,1), activation='relu', padding='same') (c2)
    p2 = MaxPooling3D((3, 3, 3)) (c2)

    c3 = Conv3D(32, (4, 4, 4), strides=(1,1,1), activation='relu', padding='same') (p2)
    c3 = Dropout(0.1)(c3)
    c3 = Conv3D(32, (4, 4, 4), strides=(1,1,1), activation='relu', padding='same') (c3)
    p3 = MaxPooling3D((3, 3, 3)) (c3)

#    c4 = Conv3D(64, (4, 4, 4), strides=(1,1,1), activation='relu', padding='same') (p3)
#    c4 = Dropout(0.1)(c4)
#    c4 = Conv3D(64, (4, 4, 4), strides=(1,1,1), activation='relu', padding='same') (c4)
#    p4 = MaxPooling3D((3, 3, 3)) (c4)
    
    p6 = AveragePooling3D((3, 1, 1)) (p3)

    outputs = Conv3D(1, (1, 1, 1), activation='relu') (p6)

    return Model(inputs=[inputs], outputs=[outputs])