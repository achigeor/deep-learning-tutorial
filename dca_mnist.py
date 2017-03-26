# coding=utf-8
from deeplearning.cnn.networks import dcA
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard

# import matplotlib.pyplot as plt

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

print trainData
noise_factor = 0.5
x_train_noisy = trainData + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=trainData.shape)
x_test_noisy = testData + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=testData.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

print("[INFO] compiling model...")
autoencoder = dcA.build(width=28, height=28, depth=1)

# here we use binary_crossentropy because we don't try to recognise digits,
# but to denoise the input image
autoencoder.compile(loss="binary_crossentropy", optimizer="adadelta",
                    metrics=["accuracy"])

print("[INFO] training...")
# Training our network is accomplished by making a call to the .fit  method of the instantiated model.
# We’ll allow our network to train for 20 epochs (indicating that our network will “see” each of the
# training examples a total of 20 times to learn distinguishing filters for each digit class).
# batch_size = 32 changed from 128, to check with my GPU (GTX 970)

autoencoder.fit(x_train_noisy, trainData,
                nb_epoch=10,
                batch_size=32,
                shuffle=True,
                validation_data=(x_test_noisy, testData))#,
                #callbacks=[TensorBoard(log_dir='/tmp/autoencoder', histogram_freq=0, write_graph=False)])

# show the accuracy on the testing set
# print("[INFO] evaluating...")
# (loss, accuracy) = model.evaluate(testData, testLabels,
#                                   batch_size=32, verbose=1)
# print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

# check to see if the model should be saved to file
# if args["save_model"] > 0:
#     print("[INFO] dumping weights to file...")
#     model.save_weights(args["weights"], overwrite=True)

decoded_imgs = autoencoder.predict(x_test_noisy)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display noised
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
