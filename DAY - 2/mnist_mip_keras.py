# import libraries
import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from keras.datasets import mnist  
from keras.optimizers import Adam   

#load the dataset
(x_train, y_train), (y_train, y_test) = mnist.load_data()
# plt.imshow(x_train[0], cmap='gray')    
# plt.show()

# normalize
x_train = x_train.astype('float32') / 255.0
x_test = y_train.astype('float32') / 255.0

# to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#architecture
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))    
model.add(Dense(10, activation='softmax'))

# #compile
# model.compile(optimizer='adam', loss='categoricalcross_entropy')

# #train
# model = model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=1)

# #evaluate

model.compile(optimizer='Adam', loss='categorical_crossentropy')
model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=1)
