import numpy as np
from sklearn.metrics import mean_squared_error
from Feed_Forward import predict_with_network

weights_0 = {'node_0': np.array([2, 1]), 
             'node_1': np.array([1, 2]), 
             'output': np.array([1, 1])}

weights_1 = {'node_0': np.array([2, 1]),
             'node_1': np.array([ 1. ,  1.5]),
             'output': np.array([ 1. ,  1.5])}

input_data = [np.array([0, 3]), np.array([1, 2]), np.array([-1, -2]), np.array([4, 0])]

target_actuals = [1, 3, 5, 7]

model_output_0 = []
model_output_1 = []

for row in input_data:
    model_output_0.append(predict_with_network(row, weights_0))
    model_output_1.append(predict_with_network(row, weights_1))
    
mse_0 = mean_squared_error(model_output_0, target_actuals)
mse_1 = mean_squared_error(model_output_1, target_actuals)

print("Mean Squared Error with weights_0: %f" %mse_0)
print("Mean Squared Error with weights_1: %f" %mse_1)