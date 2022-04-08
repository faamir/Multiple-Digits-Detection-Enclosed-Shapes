import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, BatchNormalization
from keras import optimizers
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import keras

from keras.callbacks import Callback

# Save best model for CNN and Early stopping
class TerminateOnBaseline(Callback):
    """Callback that terminates training when either acc or val_acc reaches a specified baseline"""

    def __init__(self, monitor="val_loss", baseline=0.001):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get(self.monitor)
        if acc is not None:
            if acc <= self.baseline:
                print(
                    "Epoch %d: Reached baseline, terminating training" % (epoch)
                )
                self.model.stop_training = True


save_best = keras.callbacks.ModelCheckpoint(
    "./convolutional_net/saved_models/digit_classifier5.h5",
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
)

# MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

X_train = np.concatenate((X_train, X_test), axis=0)
y_train = np.concatenate((y_train, y_test), axis=0)

# threshold for remove noise
_, X_train_thresh = cv2.threshold(X_train, 127, 255, cv2.THRESH_BINARY)
_, X_test_thresh = cv2.threshold(X_test, 127, 255, cv2.THRESH_BINARY)

# reshape
X_train = X_train_thresh.reshape(-1, 28, 28, 1)
X_test = X_test_thresh.reshape(-1, 28, 28, 1)

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)


# Convolutional Neural Network
model = Sequential()
model.add(Conv2D(32, kernel_size=3, activation="relu", input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=3, activation="relu"))
model.add(BatchNormalization())
model.add(
    Conv2D(32, kernel_size=5, strides=2, padding="same", activation="relu")
)
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size=3, activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=3, activation="relu"))
model.add(BatchNormalization())
model.add(
    Conv2D(64, kernel_size=5, strides=2, padding="same", activation="relu")
)
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(10, activation="softmax"))

model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer="adam",
    metrics=["accuracy"],
)

model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    shuffle=True,
    batch_size=64,
    # validation_data=(X_test, y_test),
    validation_split=0.1,
    callbacks=[
        TerminateOnBaseline(monitor="val_loss", baseline=0.0005),
        save_best,
    ],
).history


# model.save("./convolutional_net/saved_models/digit_classifier4.h5")
