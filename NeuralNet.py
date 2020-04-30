import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from utils import test_time

from GlobalConstants import hidden_layers, activations, optimizer


class NeuralNet:
    def __init__(self, input_shape):
        input_layer = Input(shape=input_shape)
        # Define architecture
        x = input_layer
        #assert len(hidden_layers) == len(
        #    activations), 'Different number of hidden layers and activations'
        for layer, activation in zip(hidden_layers, activations):
            x = Dense(layer, activation=activation)(x)
        self.anet = Model(input_layer, x)
        self.anet._name = 'ANET'
        self.history = []

        # Compile model
        self.anet.compile(optimizer=optimizer,
                          loss='mean_squared_error', metrics=['accuracy'])

    def random_possible_action(self, possible_actions):
        random_index = np.random.randint(len(possible_actions))
        return possible_actions[random_index]

    def best_action(self, possible_actions, state, player):
        """
        Returns the greedy best action
        """
        features = np.append(state, player).reshape((1, len(state)+1))
        action_probabilities = self.anet(features)
        # If there are only zeros
        if not np.any(action_probabilities):
            print('......only zeros in predictions, returning random action')
            return self.random_possible_action(possible_actions)
        action_probabilities = self.scale_actions(state, action_probabilities)
        return np.argmax(action_probabilities)

    def default_policy(self, possible_actions, state, player):
        """
        NOTE: predict takes 0.038 seconds, total runtime without assert is 0.038 (predict uses all)
        possible_actions: list of tuples
        state: ndarray
        player: tuple

        :returns: tuple, action
        """
        features = np.append(state, player).reshape((1, len(state)+1))
        action_probabilities = self.anet(features)
        # If there are only zeros
        if not np.any(action_probabilities):
            print('......only zeros in predictions, returning random action')
            return self.random_possible_action(possible_actions)
        action_probabilities = self.scale_actions(state, action_probabilities)
        return self.get_action(action_probabilities)

    def scale_actions(self, state, action_probabilities):
        # Make impossible actions have probability 0
        # If the board is not 0, set action_probabilities to 0
        # Predict gives (1,len(features)), take first to get (len(features), )
        action_probabilities = np.where(state, 0, action_probabilities)[0]
        # Normalize
        return action_probabilities/np.sum(action_probabilities)

    def get_action(self, scaled_predictions):
        """
        Randomly pick action, weighted by scale (illegal actions scaled to 0 by scaled_predictions)
        """
        return np.random.choice(len(scaled_predictions), p=scaled_predictions)

    def train_on_rbuf(self, train_X, train_y, batch_size):
        """
        :param train_X: training features, state+player
        :param train_y: training labels, D (distributions over actions from states)
        :param batch_size: batch size, int
        """
        history = self.anet.fit(train_X, train_y, epochs=3,
                      verbose=0, batch_size=batch_size)
        self.history.append(history)

    def save_params(self, path):
        """
        Saves weights and biases of network to file

        :param path: str, grid_size and round the params have been saved 

        https://www.tensorflow.org/tutorials/keras/save_and_load
        """
        self.anet.save_weights(path)
        print('...parameters have been saved to {}'.format(path))

    def load_params(self, path):
        """
        Load weights and biases from file to the network

        :param i: str, grid_size and round the params have been saved 
        """
        self.anet.load_weights(path)
