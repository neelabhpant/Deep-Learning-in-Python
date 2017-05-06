import pandas as pd
import numpy as np

cps = pd.read_csv('cps.csv', header=1) #Reading the csv file
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
    if i=='Not':
        union.append(0)
    else:
        union.append(1)
        
for sex in cps['sex']:
    if sex=='M':
        female.append(0)
    else:
        female.append(1)
        
for marriage in cps['married']:
    if marriage == 'Single':
        marr.append(0)
    else:
        marr.append(1)
        
for i in cps['south']:
    if i=='NS':
        south.append(0)
    else:
        south.append(1)
        
for i in cps['sector']:
    if i=='manuf':
        manufacturing.append(1)
    else:
        manufacturing.append(0)
        
for i in cps['sector']:
    if i=='const':
        construction.append(1)
    else:
        construction.append(0)

#Creating a dictionary for the final DataFrame        

my_dict = {'wage_per_hour':wage_per_hour,
          'union':union,
          'education_yrs':education_yrs,
          'experience_yrs':experience_yrs,
          'age':age,
          'female':female,
          'marr':marr,
          'south':south,
          'manufacturing':manufacturing,
          'construction':construction}

#Final DataFrame

df = pd.DataFrame(my_dict)

#Creating Predictors and Targets labels

predictors = df.as_matrix(columns=df.columns[:8])
targets = [i for i in df['wage_per_hour']]
targets = np.array(targets)