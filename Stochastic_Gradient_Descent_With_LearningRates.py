#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 9 12:05:43 2017
@author: neelabhpant
"""

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
import pandas as pd
import math
from keras.optimizers import SGD

'''Cleaning Training Data'''

train_data = pd.read_csv('Titanic_Data/train.csv')
train_survived = [i for i in train_data['Survived']]
train_pclass = [i for i in train_data['Pclass']]
train_age = [29.699118 if math.isnan(i) else i for i in train_data['Age']]
train_age_was_missing = [1 if math.isnan(i) else 0 for i in train_data['Age']]
train_sibsp = [i for i in train_data['SibSp']]
train_parch = [i for i in train_data['Parch']]
train_fare = [i for i in train_data['Fare']]
train_male = [1 if i == 'male' else 0 for i in train_data['Sex']]
train_embarked_from_cherbourg = [1 if i == 'C' else 0 for i in train_data['Embarked']]
train_embarked_from_queenstown = [1 if i == 'Q' else 0 for i in train_data['Embarked']]
train_embarked_from_southampton = [1 if i == 'S' else 0 for i in train_data['Embarked']]
train_dict = {'survived': train_survived,
              'pclass': train_pclass,
              'age': train_age,
              'age_was_missing': train_age_was_missing,
              'sibsp': train_sibsp,
              'parch': train_parch,
              'fare': train_fare,
              'male': train_male,
              'embarked_from_cherbourg': train_embarked_from_cherbourg,
              'embarked_from_queenstown': train_embarked_from_queenstown,
              'embarked_from_southampton': train_embarked_from_southampton}
train_df = pd.DataFrame(train_dict)
train_predictors = train_df.as_matrix(columns=train_df.columns[:10])
train_n_cols = train_predictors.shape[1]
train_target = to_categorical(train_df.survived)

'''Cleaning Testing Data'''

test_data = pd.read_csv('Titanic_Data/test.csv')
test_pclass = [i for i in test_data['Pclass']]
test_age = [29.699118 if math.isnan(i) else i for i in test_data['Age']]
test_age_was_missing = [1 if math.isnan(i) else 0 for i in test_data['Age']]
test_sibsp = [i for i in test_data['SibSp']]
test_parch = [i for i in test_data['Parch']]
test_fare = [i for i in test_data['Fare']]
test_male = [1 if i == 'male' else 0 for i in test_data['Sex']]
test_embarked_from_cherbourg = [1 if i == 'C' else 0 for i in test_data['Embarked']]
test_embarked_from_queenstown = [1 if i == 'Q' else 0 for i in test_data['Embarked']]
test_embarked_from_southampton = [1 if i == 'S' else 0 for i in test_data['Embarked']]
test_dict = {'pclass': test_pclass,
             'age': test_age,
             'age_was_missing': test_age_was_missing,
             'sibsp': test_sibsp,
             'parch': test_parch,
             'fare': test_fare,
             'male': test_male,
             'embarked_from_cherbourg': test_embarked_from_cherbourg,
             'embarked_from_queenstown': test_embarked_from_queenstown,
             'embarked_from_southampton': test_embarked_from_southampton}
test_df = pd.DataFrame(test_dict)
test_predictors = test_df.as_matrix()


def get_new_model(x):
    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape=(x,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return (model)


# Different Learning Rates to test the model
lr_to_test = [0.000001, 0.01, 1]

# Checking the model on different Learning Rates with Stochastic Gradient Descent as Optimizer

for lr in lr_to_test:
    print("\n\nTesting model with learning rate: %f\n" % lr)
    model = get_new_model(train_n_cols)
    my_optimizer = SGD(lr=lr)
    model.compile(optimizer=my_optimizer, loss='categorical_crossentropy')
    model.fit(train_predictors, train_target)


