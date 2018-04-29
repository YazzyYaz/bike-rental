import numpy as np

class NeuralNetwork(object):
    """Numpy Neural Network Class"""
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.lr = learning_rate

        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5,
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5,
                                        (self.hidden_nodes, self.output_nodes))

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        self.activation_function = sigmoid

    def train(self, features, targets):
        """Train the Network on Features and Targets"""
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            ## Forward Pass Training
            hidden_inputs = np.dot(X, self.weights_input_to_hidden)
            hidden_outputs = self.activation_function(hidden_inputs)

            final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
            final_outputs = final_inputs

            ## Backward Pass Training
            error = y - final_outputs
            hidden_error = np.dot(error, self.weights_hidden_to_output.T)
            output_error_term = error
            hidden_error_term = hidden_error * (hidden_outputs * (1 - hidden_outputs))

            delta_weights_i_h += hidden_error_term * X[:, None]
            delta_weights_h_o += output_error_term * hidden_outputs[:, None]

        self.weights_hidden_to_output += (self.lr * delta_weights_h_o) / n_records
        self.weights_input_to_hidden += (self.lr * delta_weights_i_h) / n_records

    def run(self, features):
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs
        return final_outputs

def MSE(y, Y):
    return np.mean((y-Y)**2)
