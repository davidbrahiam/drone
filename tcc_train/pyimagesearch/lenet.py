# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class LeNet:
	@staticmethod
	def build(width, height, depth, classes):
		print("LeNet")
		# initialize the model
		model = Sequential()
		inputShape = (height, width, depth)

		# if we are using "channels first", update the input shape
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)

		# first set of CONV => RELU => POOL layers
		model.add(Conv2D(20, (5, 5), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# second set of CONV => sigmoid => POOL layers
		model.add(Conv2D(50, (5, 5), padding="same"))
		model.add(Activation("tanh"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("sigmoid"))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("sigmoid"))

		# return the constructed network architecture
		return model

		# 7 - all sigmoid
		# 8 - 3 tanh 1 sigmoid
		# 9 - all tanh
		# 10 - 2 relu 2 sigmoid	
		# 11 - all relu
		# 12 - 1 relu 1 tanh 2 sigmoid
		# 13 - 1 relu 1 tanh 1 relu 1 sigmoid ok mais para not
		# 14 - 12 com categorical_crossentropy  +-
		# 15 - 13 com categorical_crossentropy
		# 16 - 12 - 55 epocas not 
		# 17 - 1 relu 1 sigmoid 1 tanh 1 sigmoid not
		# 18 - 1 relu 1 tanh 1 relu 1 sigmoid categorical_crossentropy 35 eps not
		# 19 - 1 relu 1 sigmoid 1 relu 1 sigmoid categorical_crossentropy 35 eps not
		# 20 - 17 categorical_crossentropy 35 eps not
		# 21 - 1 relu 1 tanh 1 sigmoid 1 sigmoid categorical_crossentropy 35ps
		# 22 - 1 relu 1 sigmoid 1 sigmoid 1 sigmoid categorical_crossentropy 35 eps not
		# 23 - 1 relu 1 tanh 1 sigmoid 1 sigmoid categorical_crossentropy 45ps not
		# 24 - 1 relu 1 tanh 1 sigmoid 1 sigmoid categorical_crossentropy 25ps	winner 
		# 25 - 1 relu 1 tanh 1 sigmoid 1 sigmoid categorical_crossentropy 30ps 