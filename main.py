# import modules and libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import utils

# dividing the dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# data preprocessing

# data normalization
x_train = x_train / 255
x_test = x_test / 255

# creating a neural network model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# model compilation
model.compile(optimizer=tf.keras.optimizers.SDG(), loss="sparse_categorical", metrics=['accuracy'])

# model training
model.fit(x_train, y_train, epochs=10)

# checking prediction accuracy
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy: ', test_acc)

# predicting
predictions = model.predict(x_train)

print(np.argmax(predictions[0]))
print(class_names[(y_train[0])])