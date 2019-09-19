# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras import regularizers
from keras.optimizers import rmsprop
from keras.layers.core import Dense
from keras import backend as K

class SmallVGGNet:
	@staticmethod
	def build(width, height, depth, classes):
		# Create model architecture
		weight_decay = 1e-4
		model = Sequential()

		# # initialize the model along with the input shape to be
		# # "channels last" and the channels dimension itself
		# model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# # if we are using "channels first", update the input shape
		# # and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		model.add(Conv2D(32, kernel_size=3, padding="same", activation="elu",
						kernel_regularizer=regularizers.l2(weight_decay), input_shape=inputShape))
		model.add(BatchNormalization())
		model.add(Conv2D(32, kernel_size=3, padding="same", activation="elu",
						kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=2))
		model.add(Dropout(0.2))

		model.add(Conv2D(64, kernel_size=3, padding="same", activation="elu",
						kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(BatchNormalization())
		model.add(Conv2D(64, kernel_size=3, padding="same", activation="elu",
						kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=2))
		model.add(Dropout(0.3))

		model.add(Conv2D(128, kernel_size=3, padding="same", activation="elu",
						kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(BatchNormalization())
		model.add(Conv2D(128, kernel_size=3, padding="same", activation="elu",
						kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=2))
		model.add(Dropout(0.4))

		model.add(Flatten())
		model.add(Dense(classes, activation="sigmoid"))

		# Compile the model, using an optimized rms this time, which we will adapt
		# during training
		optimized_rmsprop = rmsprop(lr=0.001,decay=1e-6)
		model.compile(loss="categorical_crossentropy", optimizer=optimized_rmsprop,
					metrics=["accuracy"])

		# print('small')
		# # initialize the model along with the input shape to be
		# # "channels last" and the channels dimension itself
		# model = Sequential()
		# inputShape = (height, width, depth)
		# chanDim = -1

		# # if we are using "channels first", update the input shape
		# # and channels dimension
		# if K.image_data_format() == "channels_first":
		# 	inputShape = (depth, height, width)
		# 	chanDim = 1

		# # CONV => RELU => POOL layer set
		# model.add(Conv2D(32, (3, 3), padding="same",
		# 	input_shape=inputShape))
		# model.add(Activation("relu"))
		# model.add(BatchNormalization(axis=chanDim))
		# model.add(MaxPooling2D(pool_size=(2, 2)))
		# model.add(Dropout(0.25))

		# # (CONV => RELU) * 2 => POOL layer set
		# model.add(Conv2D(64, (3, 3), padding="same"))
		# model.add(Activation("relu"))
		# model.add(BatchNormalization(axis=chanDim))
		# model.add(Conv2D(64, (3, 3), padding="same"))
		# model.add(Activation("relu"))
		# model.add(BatchNormalization(axis=chanDim))
		# model.add(MaxPooling2D(pool_size=(2, 2)))
		# model.add(Dropout(0.25))

		# # (CONV => RELU) * 3 => POOL layer set
		# model.add(Conv2D(128, (3, 3), padding="same"))
		# model.add(Activation("relu"))
		# model.add(BatchNormalization(axis=chanDim))
		# model.add(Conv2D(128, (3, 3), padding="same"))
		# model.add(Activation("relu"))
		# model.add(BatchNormalization(axis=chanDim))
		# model.add(Conv2D(128, (3, 3), padding="same"))
		# model.add(Activation("relu"))
		# model.add(BatchNormalization(axis=chanDim))
		# model.add(MaxPooling2D(pool_size=(2, 2)))
		# model.add(Dropout(0.25))

		# # first (and only) set of FC => RELU layers
		# model.add(Flatten())
		# model.add(Dense(512))
		# model.add(Activation("relu"))
		# model.add(BatchNormalization())
		# model.add(Dropout(0.5))

		# # softmax classifier
		# model.add(Dense(classes))
		# model.add(Activation("sigmoid"))

		# # return the constructed network architecture
		return model