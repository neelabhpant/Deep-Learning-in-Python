'''This helps in predicting no. of transaction when a customer has 'x' no. of children and
 'y' no. of bank accounts for four customers.'''

import numpy as np

#input_data = [#_of_children, #_of_accounts] for 4 customers
input_data = [np.array([3, 5]), np.array([ 1, -1]), np.array([0, 0]), np.array([8, 4])]
weights = {'node_0': np.array([2, 4]), 'node_1': np.array([ 4, -5]), 'output': np.array([2, 7])}

def relu(input):
    output = max(0, input)
    return output

def predict_with_network(input_data_row, weights):
    node_0_input = (input_data_row * weights['node_0']).sum()
    node_0_output = relu(node_0_input)
    
    node_1_input = (input_data_row * weights['node_1']).sum()
    node_1_output = relu(node_1_input)
    
    hidden_layer_outputs = np.array([node_0_output, node_1_output])
    
    input_to_final_layer = (hidden_layer_outputs * weights['output']).sum()
    model_output = relu(input_to_final_layer)
    return model_output

results = []

for input_data_row in input_data:
    results.append(predict_with_network(input_data_row, weights))
    
print(results)