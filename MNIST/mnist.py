import numpy as np


from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import np_utils

from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

x_train /= 255
x_test /= 255

print(y_train[0])

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

print(y_train[0])

print(x_train.shape)

model = Sequential()
model.add(Conv2D(50, (5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(40, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(.2))
model.add(Conv2D(30, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=1000, epochs=10)

score = model.evaluate(x_test, y_test, verbose=1)
print("Test score: ", score[0])
print("Test accuracy: ", score[1])

model.save('mnist_cnn.h5')

