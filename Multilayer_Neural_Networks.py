import numpy as np

def relu(input):
    output = max(0, input)
    return output

def predict_with_network(input_data):
    node_0_0_input = (weights['node_0_0'] * input_data).sum()
    node_0_0_output = relu(node_0_0_input)
    node_0_1_input = (weights['node_0_1'] * input_data).sum()
    node_0_1_output = relu(node_0_1_input)
    
    hidden_0_outputs = np.array([node_0_0_output, node_0_1_output])
    node_1_0_input = (weights['node_1_0'] * hidden_0_outputs).sum()
    node_1_0_output = relu(node_1_0_input)
    node_1_1_input = (weights['node_1_1'] * hidden_0_outputs).sum()
    node_1_1_output = relu(node_1_1_input)
    
    hidden_1_outputs = np.array([node_1_0_output, node_1_1_output])
    
    
    model_output = (weights['output'] * hidden_1_outputs).sum()
    
    return model_output

if __name__ == '__main__':
    input_data = np.array([3, 5])
    weights = {'node_0_0': np.array([2, 4]),
               'node_0_1': np.array([ 4, -5]),
               'node_1_0': np.array([-1,  1]),
               'node_1_1': np.array([2, 2]),
               'output': np.array([2, 7])}
    output = predict_with_network(input_data)
    print(output)