from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

def pred(input_data, target, weights):
    return ((input_data * weights).sum())

def get_slope(input_data, target, weights):
    preds = pred(input_data, target, weights)
    error = target - preds
    slope = 2 * input_data * error
    return slope

def get_mse(input_data, target, weights):
    preds = pred(input_data, target, weights)
    return mean_squared_error([preds], [target])

weights = np.array([0, 2, 1])
input_data = np.array([1, 2, 3])
target = 0

n_updates = 20
mse_hist = []
for i in range(n_updates):
    slope = get_slope(input_data, target, weights)
    weights = weights + (learning_rate * slope)
    mse = get_mse(input_data, target, weights)
    mse_hist.append(mse)
    
plt.plot(mse_hist)
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.show()