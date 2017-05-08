#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 7 16:15:43 2017
@author: neelabhpant
"""


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
import pandas as pd
import math

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
train_predictors = train_df.as_matrix(columns=train_df.columns[:10])
train_n_cols = train_predictors.shape[1]
train_target = to_categorical(train_df.survived)

'''Setting up the model'''

model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(train_n_cols,)))
model.add(Dense(100, activation='tanh'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_predictors, train_target)


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
test_predictors = test_df.as_matrix()


'''Making the predictions'''

predictions = model.predict(test_predictors)
predicted_prob_true = predictions[:,1]

print(predicted_prob_true)
