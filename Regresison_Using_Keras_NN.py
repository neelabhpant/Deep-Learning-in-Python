import pandas as pd
import numpy as np


def get_data(csv_file):
    cps = pd.read_csv(csv_file, header=1)
    cps = cps.drop('race', 1)
    cps = cps.drop('hispanic', 1)

    # Cleaning the data

    wage_per_hour = [round(i, 2) for i in cps['wage']]
    union = []
    education_yrs = [i for i in cps['educ']]
    experience_yrs = [i for i in cps['exper']]
    age = [i for i in cps['age']]
    female = []
    marr = []
    south = []
    manufacturing = []
    construction = []

    for i in cps['union']:
        if i == 'Not':
            union.append(0)
        else:
            union.append(1)

    for sex in cps['sex']:
        if sex == 'M':
            female.append(0)
        else:
            female.append(1)

    for marriage in cps['married']:
        if marriage == 'Single':
            marr.append(0)
        else:
            marr.append(1)

    for i in cps['south']:
        if i == 'NS':
            south.append(0)
        else:
            south.append(1)

    for i in cps['sector']:
        if i == 'manuf':
            manufacturing.append(1)
        else:
            manufacturing.append(0)

    for i in cps['sector']:
        if i == 'const':
            construction.append(1)
        else:
            construction.append(0)

    # Creating a dictionary for the final DataFrame

    my_dict = {'wage_per_hour': wage_per_hour,
               'union': union,
               'education_yrs': education_yrs,
               'experience_yrs': experience_yrs,
               'age': age,
               'female': female,
               'marr': marr,
               'south': south,
               'manufacturing': manufacturing,
               'construction': construction}

    # Final DataFrame

    df = pd.DataFrame(my_dict)

    # Creating Predictors and Targets labels
    predictors = df.as_matrix(columns=df.columns[:9])
    targets = [i for i in df['wage_per_hour']]
    target = np.array(targets)

    return [df, predictors, target]


[df, predictors, target] = get_data('cps.csv')  # Getting Data, Predictors and Target
import keras  # i Importing required keras packages
from keras.layers import Dense
from keras.models import Sequential

n_cols = predictors.shape[1]  # Setting up dimensions of an input
model = Sequential()  # Setting up a Sequential model

# We have 2 hidden layers
model.add(Dense(50, activation='relu', input_shape=(
n_cols,)))  # Setting up 1st Dense layers where each node is connected to every node in the next layer
model.add(Dense(32, activation='relu'))  # Setting up 2nd layer

# 1 output layer
model.add(Dense(1))  # Setting up final layer

# Compiling the model using 'adam' optimizer and MSE as loss function
model.compile(optimizer='adam', loss='mean_squared_error')
print("Loss function: " + model.loss)

# Fitting the model
model.fit(predictors, target)