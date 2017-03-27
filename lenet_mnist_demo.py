# coding=utf-8
from keras.datasets import mnist
from deeplearning.cnn.networks import LeNet
from keras.optimizers import SGD
from keras import optimizers
from keras.utils import np_utils
import numpy as np


print("[INFO]: Loading dataset")
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images[:, np.newaxis, :, :]
test_images = test_images[:, np.newaxis, :, :]

train_labels = np_utils.to_categorical(train_labels, 10)
test_labels = np_utils.to_categorical(test_labels, 10)

print("[INFO] compiling model...")
opt = SGD(lr=0.01)
# opt = optimizers.nadam()

model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])


print("[INFO] training...")
# Training our network is accomplished by making a call to the .fit  method of the instantiated model.
# We’ll allow our network to train for 20 epochs (indicating that our network will “see” each of the
# training examples a total of 20 times to learn distinguishing filters for each digit class).
# batch_size = 32 changed from 128, to check with my GPU (GTX 970)
model.fit(train_images, train_labels, batch_size=32, nb_epoch=20,
          verbose=1)

# show the accuracy on the testing set
print("[INFO] evaluating...")
(loss, accuracy) = model.evaluate(test_images, test_labels,
                                  batch_size=32, verbose=1)
print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

# check to see if the model should be saved to file
print("[INFO] dumping weights to file...")
model.save_weights("lenet_weights", overwrite=True)

# randomly select a few testing digits
for i in np.random.choice(np.arange(0, len(test_labels)), size=(10,)):
    # classify the digit
    probs = model.predict(test_images[np.newaxis, i])
    # The actual prediction  of our network is obtained by finding the index of the class label with the largest
    # probability. Remember, our network will return a set of probabilities via the softmax function, one for each
    # class label — the actual “prediction” of the network is therefore the class label with the largest probability.
    prediction = probs.argmax(axis=1)

    # show the prediction
    print("[INFO] Predicted: {}, Actual: {}".format(prediction[0],
                                                    np.argmax(test_images[i])))
