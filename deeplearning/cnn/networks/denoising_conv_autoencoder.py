# coding=utf-8
# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense


class dcA:
    @staticmethod
    def build(width, height, depth):

        model = Sequential()

        model.add(Convolution2D(32, 3, 3, border_mode="same",
                                input_shape=(depth, height, width)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), border_mode="same"))

        # second set of CONV => RELU => POOL
        model.add(Convolution2D(32, 3, 3, border_mode="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # At this point the representation is (32,7,7)

        model.add(Convolution2D(32, 3, 3, border_mode="same"))
        model.add(Activation("relu"))
        model.add(UpSampling2D((2, 2)))

        model.add(Convolution2D(32, 3, 3, border_mode="same"))
        model.add(Activation("relu"))
        model.add(UpSampling2D((2, 2)))

        model.add(Convolution2D(1, 3, 3, border_mode="same"))
        model.add(Activation("sigmoid"))

        # the decoded part

        return model


