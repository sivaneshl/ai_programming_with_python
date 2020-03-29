import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.array([0.5, 0.1, -0.2])
target = 0.6
learnrate = 0.5

weights_input_hidden = np.array([[0.5, -0.6],
                                 [0.1, -0.2],
                                 [0.1,  0.7]])
weights_hidden_output = np.array([0.1, -0.3])

# forward pass
hidden_layer_input = np.matmul(x, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_input = np.matmul(hidden_layer_output, weights_hidden_output)
output_layer_output = sigmoid(output_layer_input)

## Backwards pass
## Calculate output error
error = target - output_layer_output

# Calculate error term for output layer
output_error_term = error * output_layer_output * (1-output_layer_output)

# Calculate error term for hidden layer
hidden_error_term = np.dot(output_error_term, weights_hidden_output) * hidden_layer_output * (1-hidden_layer_output)

# Calculate change in weights for hidden layer to output layer
delta_w_h_o = learnrate * output_error_term * hidden_layer_output

# Calculate change in weights for input layer to hidden layer
delta_w_i_h = learnrate * hidden_error_term * x[:,None]

print('Change in weights for hidden layer to output layer:')
print(delta_w_h_o)
print('Change in weights for input layer to hidden layer:')
print(delta_w_i_h)

