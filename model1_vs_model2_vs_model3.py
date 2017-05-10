#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 9 14:10:19 2017
@author: neelabhpant
"""


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
import pandas as pd
import math
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

'''Cleaning Training Data'''

train_data = pd.read_csv('Titanic_Data/train.csv')
train_survived = [i for i in train_data['Survived']]
train_pclass = [i for i in train_data['Pclass']]
train_age = [29.699118 if math.isnan(i) else i for i in train_data['Age']]
train_age_was_missing = [1 if math.isnan(i) else 0 for i in train_data['Age']]
train_sibsp = [i for i in train_data['SibSp']]
train_parch = [i for i in train_data['Parch']]
train_fare = [i for i in train_data['Fare']]
train_male = [1 if i=='male' else 0 for i in train_data['Sex']]
train_embarked_from_cherbourg = [1 if i=='C' else 0 for i in train_data['Embarked']]
train_embarked_from_queenstown = [1 if i=='Q' else 0 for i in train_data['Embarked']]
train_embarked_from_southampton = [1 if i=='S' else 0 for i in train_data['Embarked']]
train_dict = {'survived':train_survived,
              'pclass':train_pclass,
              'age':train_age,
              'age_was_missing':train_age_was_missing,
              'sibsp':train_sibsp,
              'parch':train_parch,
              'fare':train_fare,
              'male':train_male,
              'embarked_from_cherbourg':train_embarked_from_cherbourg,
              'embarked_from_queenstown':train_embarked_from_queenstown,
              'embarked_from_southampton':train_embarked_from_southampton}
train_df = pd.DataFrame(train_dict)

# Saving the Predictors: train_predictors as Numpy array. Each row correponds to a Dataset
train_predictors = train_df.as_matrix(columns=train_df.columns[:10])
# Saving the number of columns on predictors: train_n_cols
train_n_cols = train_predictors.shape[1]
# Saving the targets: train_target as One Hot Encoding
train_target = to_categorical(train_df.survived)


'''Cleaning Testing Data'''

test_data = pd.read_csv('Titanic_Data/test.csv')
test_pclass = [i for i in test_data['Pclass']]
test_age = [29.699118 if math.isnan(i) else i for i in test_data['Age']]
test_age_was_missing = [1 if math.isnan(i) else 0 for i in test_data['Age']]
test_sibsp = [i for i in test_data['SibSp']]
test_parch = [i for i in test_data['Parch']]
test_fare = [i for i in test_data['Fare']]
test_male = [1 if i=='male' else 0 for i in test_data['Sex']]
test_embarked_from_cherbourg = [1 if i=='C' else 0 for i in test_data['Embarked']]
test_embarked_from_queenstown = [1 if i=='Q' else 0 for i in test_data['Embarked']]
test_embarked_from_southampton = [1 if i=='S' else 0 for i in test_data['Embarked']]
test_dict = {'pclass':test_pclass,
              'age':test_age,
              'age_was_missing':test_age_was_missing,
              'sibsp':test_sibsp,
              'parch':test_parch,
              'fare':test_fare,
              'male':test_male,
              'embarked_from_cherbourg':test_embarked_from_cherbourg,
              'embarked_from_queenstown':test_embarked_from_queenstown,
              'embarked_from_southampton':test_embarked_from_southampton}
test_df = pd.DataFrame(test_dict)
# Saving the Predictors: test_predictors as Numpy array. Each row correponds to a Dataset
test_predictors = test_df.as_matrix()

input_shape = (train_n_cols,)

# Specify model_1
model_1 = Sequential()
# Input Layers
model_1.add(Dense(10, activation='relu', input_shape = input_shape))
model_1.add(Dense(10, activation='relu'))
# Output Layer
model_1.add(Dense(2, activation='softmax'))
# Compile the model
model_1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Specify model_2
model_2 = Sequential()
# Input Layers
model_2.add(Dense(50, activation='relu', input_shape = input_shape))
model_2.add(Dense(50, activation='relu'))
# Output Layer
model_2.add(Dense(2, activation='softmax'))
# Compile the model
model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Specify model_3
model_3 = Sequential()
# Input Layers
model_3.add(Dense(100, activation='relu', input_shape=input_shape))
model_3.add(Dense(100, activation='relu'))
model_3.add(Dense(100, activation='relu'))
# Output Layer
model_3.add(Dense(2, activation='softmax'))
# Compile the model
model_3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



# Define early_stopping_monitor
# Stop optimization when the validation loss hasn't improved for 2 epochs: Patience=2
early_stopping_monitor = EarlyStopping(patience=2)
# Fit model_1
model_1_training = model_1.fit(train_predictors, train_target, nb_epoch=15, validation_split=0.2,
                 callbacks=[early_stopping_monitor], verbose=False)
# Fit model_2
model_2_training = model_2.fit(train_predictors, train_target, nb_epoch=15, validation_split=0.2,
                               callbacks=[early_stopping_monitor], verbose=False)
# Fit model_3
model_3_training = model_3.fit(train_predictors, train_target, nb_epoch=15, validation_split=0.2,
                               callbacks=[early_stopping_monitor], verbose=False)

# Create the plot
plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b',
        model_3_training.history['val_loss'], 'g')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.show()
