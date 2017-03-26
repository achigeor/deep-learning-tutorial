# coding=utf-8
from deeplearning.autoencoders.networks import bA
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.callbacks import TensorBoard

# import matplotlib.pyplot as plt

print("[INFO] downloading MNIST...")
dataset = datasets.fetch_mldata("MNIST Original")

# this flattens the dataset to 784 size vector ( remember we'll use dense connected layers)

# data = dataset.data.reshape((dataset.data.shape[1:], 784))

data = dataset.data

(trainData, testData, trainLabels, testLabels) = train_test_split(
    data / 255.0, dataset.target.astype("int"), test_size=0.33)
print trainData[0]

print("[INFO] compiling model...")
autoencoder, encoder, decoder = bA.build_with_model(enc_dim=32, data_dim=28*28)

# here we use binary_crossentropy because we don't try to recognise digits,
# but to denoise the input image
autoencoder.compile(loss="binary_crossentropy", optimizer="adadelta",
                    metrics=["accuracy"])

print("[INFO] training...")
# Training our network is accomplished by making a call to the .fit  method of the instantiated model.
# We’ll allow our network to train for 20 epochs (indicating that our network will “see” each of the
# training examples a total of 20 times to learn distinguishing filters for each digit class).
# batch_size = 32 changed from 128, to check with my GPU (GTX 970)

autoencoder.fit(trainData, trainData,
                nb_epoch=20,
                batch_size=32,
                shuffle=True,
                validation_data=(testData, testData),
                verbose=1)

encoded_imgs = encoder.predict(testData)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(encoded_imgs[i].reshape(28,28))
    #plt.imshow(testData[i].reshape(28, 28))
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