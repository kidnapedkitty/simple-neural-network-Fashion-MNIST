# simple-neural-network-Fashion-MNIST
[![vvs](https://c.tenor.com/R8wjCxS2MCgAAAAC/oreki-black-and-white-wind.gif)](https://www.instagram.com/vvsalwayscodin/)

Implementation of a neural network model, to classify images of clothing, using `Tensorflow` in `python`.

Author:  _[Islam](https://www.linkedin.com/in/islammoldybayev)_

Dataset:  _[Fashion-MNIST](http://yann.lecun.com/exdb/mnist/)_

Important Dependancies: Tensorflow, Keras, Numpy
____
## Usage:
you can clone this GitHub repository by using `gh repo clone vvsalwayscodin/simple-neural-network-Fashion-MNIST`

install requirements by using `pip install requirements.txt`
____
## Labels:
Each training and test example is assigned to one of the following labels:

| Label | Description |
| --- | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |
----

## Explanation:
### Importing modules and libraries

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import utils
```
____
### Dividing the dataset
We should allocate 60k pictures for training and 10k pictures for the test
```python
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```
Then we also have data processing

____
### Data normalization
```python
x_train = x_train / 255
x_test = x_test / 255
```
Now we changed the range of pixel intensity to "From 0 to 1"

____

### Creating a neural network model
```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
```

____

### Model compilation
```python
model.compile(optimizer=tf.keras.optimizers.SDG(), loss="sparse_categorical", metrics=['accuracy'])
```

____
### Model training
```python
model.fit(x_train, y_train, epochs=10)
```
U can train the model in more epochs too*
____

### Checking prediction accuracy
```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy: ', test_acc)
```

____

### Predicting
```python
predictions = model.predict(x_train)

print(np.argmax(predictions[0]))
print(class_names[(y_train[0])])
```
