#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:00:48 2017
@author: neelabhpant
"""

import numpy as np
import os
import scipy.misc
from keras.layers import Dense
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt


def get_input(path):
    input_matrix = np.zeros((20000, 784))
    i = 0
    for filename in os.listdir(path):
        infilename = os.path.join(path, filename)
        img = scipy.misc.imread(infilename).astype(np.float32)
        temp = img.reshape(-1)
        input_matrix[i, :] = temp
        i += 1
    return input_matrix


def get_target(path):
    T = np.zeros((20000, 10))
    Target_matrix = np.zeros((1, 20000))
    Basic_path, directory, files = os.walk(path).next()
    i = 0
    for targets in files:
        Target_matrix[0, i] = float(targets[0])
        i += 1
    T = to_categorical(Target_matrix)
    return T


X = get_input('set1_20k')
X = X / X.max()
y = get_target('set1_20k')
Data_Matrix = np.concatenate((X, y), axis=1)
Data_Matrix = np.random.permutation(Data_Matrix)
X_new = Data_Matrix[:, 0:784]
y_new = Data_Matrix[:, 784:794]

model_1 = Sequential()
model_1.add(Dense(50, activation='relu', input_shape=(784,)))
model_1.add(Dense(50, activation='relu'))
model_1.add(Dense(10, activation='softmax'))
model_1.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model_1_training = model_1.fit(X_new, y_new, epochs=20, validation_split=0.3, verbose=False)

model_2 = Sequential()
model_2.add(Dense(100, activation='relu', input_shape=(784,)))
model_2.add(Dense(100, activation='relu'))
model_2.add(Dense(10, activation='softmax'))
model_2.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model_2_training = model_2.fit(X_new, y_new, epochs=20, validation_split=0.3, verbose=False)

model_3 = Sequential()
model_3.add(Dense(50, activation='relu', input_shape=(784,)))
model_3.add(Dense(50, activation='relu'))
model_3.add(Dense(10, activation='softmax'))
model_3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_3_training = model_3.fit(X_new, y_new, epochs=20, validation_split=0.3, verbose=False)

model_4 = Sequential()
model_4.add(Dense(100, activation='relu', input_shape=(784,)))
model_4.add(Dense(100, activation='relu'))
model_4.add(Dense(10, activation='softmax'))
model_4.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_4_training = model_4.fit(X_new, y_new, epochs=20, validation_split=0.3, verbose=False)

plt.plot(model_1_training.history['val_loss'], 'r', label='Model 1 SGD-2_50')
plt.plot(model_2_training.history['val_loss'], 'b', label='Model 2 SGD-2_100')
plt.plot(model_3_training.history['val_loss'], 'g', label='Model 3 ADAM-2_50')
plt.plot(model_4_training.history['val_loss'], 'c', label='Model 3 ADAM-2_100')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.show()

plt.plot(model_1_training.history['val_acc'], 'r', label='Model 1 SGD-2_50')
plt.plot(model_2_training.history['val_acc'], 'b', label='Model 2 SGD-2_100')
plt.plot(model_3_training.history['val_acc'], 'g', label='Model 3 ADAM-2_50')
plt.plot(model_4_training.history['val_acc'], 'c', label='Model 3 ADAM-2_100')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.show()
