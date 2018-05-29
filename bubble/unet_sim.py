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
	# s = Lambda(lambda x: x / 255) (inputs)
	s = Lambda(lambda x: x )(inputs)

	c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (s)
	c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
	p3 = MaxPooling2D((2, 2)) (c3)

	c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
	c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
	p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

	c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
	c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

	u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
	u6 = concatenate([u6, c4])
	c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
	c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

	u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
	u7 = concatenate([u7, c3])
	c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
	c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

	outputs = Conv2D(1, (1, 1), activation='sigmoid') (c7)

	return Model(inputs=[inputs], outputs=[outputs])