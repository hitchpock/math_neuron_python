import scipy.special
import numpy as np


class NeuralNetwork:
    def __init__(self, input_, hidden, output, learning_rate):
        self.input_nodes = input_
        self.hidden_nodes = hidden
        self.output_nodes = output
        self.lr = learning_rate

        self.wih = np.random.normal(0.0, pow(self.input_nodes, -0.5),
                                    (self.hidden_nodes, self.input_nodes))

        self.who = np.random.normal(0.0, pow(self.hidden_nodes, -0.5),
                                    (self.output_nodes, self.hidden_nodes))

        self.activation_func = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr * np.dot((output_errors * final_outputs *
                                      (1.0 - final_outputs)),
                                     np.transpose(hidden_outputs))

        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs *
                                      (1.0 - hidden_outputs)),
                                     np.transpose(inputs))

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)

        return final_outputs
