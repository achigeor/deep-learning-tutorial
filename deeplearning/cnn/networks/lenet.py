# coding=utf-8
# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

# The LeNet  class is defined on Line 13, followed by the build  method on Line 11.
# Whenever I define a new network architecture,
# I always place it in its own class (mainly for namespace and organization purposes)
# followed by creating a static build function


class LeNet:
    @staticmethod
    def build(width, height, depth, classes, weightsPath=None):
        # initialize the model
        model = Sequential()

        # first set of CONV => RELU => POOL
        # Our CONV  layer will learn 20 convolution filters, where each filter is of size 5 x 5.
        # The input dimensions of this value are the same width, height, and depth as our input images
        # in this case, the MNIST dataset
        # so we’ll have 28 x 28 inputs with a single channel for depth grayscale

        model.add(Convolution2D(20, 5, 5, border_mode="same",
                                input_shape=(depth, height, width)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL
        model.add(Convolution2D(50, 5, 5, border_mode="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # set of FC => RELU layers
        # we take the output of the preceding  MaxPooling2D  layer and flatten it into a single vector,
        # allowing us to apply dense/fully-connected layers.
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # softmax classifier
        # his line defines another Dense  class, but accepts a variable (i.e., not hardcoded) size.
        # This size is the number of class labels represented by the classes  variable.
        model.add(Dense(classes))
        # Finally, we apply a softmax classifier (multinomial logistic regression)
        # that will return a list of probabilities, one for each of the 10 class labels (Line 33).
        # The class label with the largest probability will be chosen as the final classification from the network.
        model.add(Activation("softmax"))

        # if a weights path is supplied (inicating that the model was
        # pre-trained), then load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)

        # return the constructed network architecture
        return model


