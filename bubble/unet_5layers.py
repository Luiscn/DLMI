from keras.models import Model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

# Build U-Net model
def unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
	inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
	s = Lambda(lambda x: x / 255) (inputs)

	c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (s)
	c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
	p1 = MaxPooling2D((2, 2)) (c1)

	c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
	c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
	p2 = MaxPooling2D((2, 2)) (c2)

	c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
	c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)

	u4 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c3)
	u4 = concatenate([u4, c2])
	c4 = Conv2D(16, (3, 3), activation='relu', padding='same') (u4)
	c4 = Conv2D(16, (3, 3), activation='relu', padding='same') (c4)

	u5 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c4)
	u5 = concatenate([u5, c1])
	c5 = Conv2D(8, (3, 3), activation='relu', padding='same') (u5)
	c5 = Conv2D(8, (3, 3), activation='relu', padding='same') (c5)

	outputs = Conv2D(1, (1, 1), activation='sigmoid') (c5)

	return Model(inputs=[inputs], outputs=[outputs])