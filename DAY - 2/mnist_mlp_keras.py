# import libraries
import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from keras.datasets import mnist  
from keras.optimizers import Adam   

# load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# architecture
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

result = model.fit(x_train, y_train, epochs=10, batch_size=64)

(loss , accuracy) = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy*100}%")

print(result.history)
print(result.history.keys())
print(result.history.values())