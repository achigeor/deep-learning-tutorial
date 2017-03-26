# coding=utf-8
from deeplearning.cnn.networks import LeNet
from sklearn.model_selection import train_test_split
from sklearn import datasets
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1,
                help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,
                help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
                help="(optional) path to weights file")
args = vars(ap.parse_args())

# grab the MNIST dataset (if this is your first time running this
# script, the download may take a minute -- the 55MB MNIST dataset
# will be downloaded)
print("[INFO] downloading MNIST...")
dataset = datasets.fetch_mldata("MNIST Original")

# reshape the MNIST dataset from a flat list of 784-dim vectors, to
# 28 x 28 pixel images, then scale the data to the range [0, 1.0]
# and construct the training and testing splits
data = dataset.data.reshape((dataset.data.shape[0], 28, 28))

# Our data  matrix now has the shape (70000, 28, 28)
# however, there is a problem, Keras assumes we are going to supply at least 1 channel per image,
# thus we need to add in an extra dimension to the data  array (Line 35).
# After this line executes, the new shape of the data  matrix will be: (70000, 1, 28, 28)
# and is now suitable for passing through our LeNet architecture.
data = data[:, np.newaxis, :, :]

# Perform a training and testing split, using 2/3 of the data for training and the remaining 1/3 for testing.
# We also reduce our images from the range [0, 255] to [0, 1.0], a common scaling technique.
(trainData, testData, trainLabels, testLabels) = train_test_split(
    data / 255.0, dataset.target.astype("int"), test_size=0.33)

# transform the training and testing labels into vectors in the
# range [0, classes] -- this generates a vector for each label,
# where the index of the label is set to `1` and all other entries
# to `0`; in the case of MNIST, there are 10 class labels

# Since we are using the categorical cross-entropy loss function, we need to apply the to_categorical function
# which converts our labels from integers to a vector, where each vector ranges from [0, classes].
# This function generates a vector for each class label,
# where the index of the correct label is set to 1 and all other entries are set to 0.
# We use cross-entropy loss, because we perform classification, and NOT regression or time series NN
trainLabels = np_utils.to_categorical(trainLabels, 10)
testLabels = np_utils.to_categorical(testLabels, 10)

# initialize the optimizer and model
# We’ll be training our network using Stochastic Gradient Descent (SGD) with a
# learning rate of 0.01 Categorical cross-entropy will be used as our loss function, a fairly standard choice when
# working with more than 2 classes.
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10,
                    weightsPath=args["weights"] if args["load_model"] > 0 else None)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# only train and evaluate the model if we *are NOT* loading a
# pre-existing model
if args["load_model"] < 0:
    print("[INFO] training...")
    # Training our network is accomplished by making a call to the .fit  method of the instantiated model.
    # We’ll allow our network to train for 20 epochs (indicating that our network will “see” each of the
    # training examples a total of 20 times to learn distinguishing filters for each digit class).
    # batch_size = 32 changed from 128, to check with my GPU (GTX 970)
    model.fit(trainData, trainLabels, batch_size=32, nb_epoch=20,
              verbose=1)

    # show the accuracy on the testing set
    print("[INFO] evaluating...")
    (loss, accuracy) = model.evaluate(testData, testLabels,
                                      batch_size=32, verbose=1)
    print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

# check to see if the model should be saved to file
if args["save_model"] > 0:
    print("[INFO] dumping weights to file...")
    model.save_weights(args["weights"], overwrite=True)

# randomly select a few testing digits
for i in np.random.choice(np.arange(0, len(testLabels)), size=(10,)):
    # classify the digit
    probs = model.predict(testData[np.newaxis, i])
    # The actual prediction  of our network is obtained by finding the index of the class label with the largest
    # probability. Remember, our network will return a set of probabilities via the softmax function, one for each
    # class label — the actual “prediction” of the network is therefore the class label with the largest probability.
    prediction = probs.argmax(axis=1)

    # Missing the part with CV to show images, REMEMBER todo when we do our implementation

    # show the prediction
    print("[INFO] Predicted: {}, Actual: {}".format(prediction[0],
                                                    np.argmax(testLabels[i])))
