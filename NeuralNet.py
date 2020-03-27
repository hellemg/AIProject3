import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from utils import test_time

from GlobalConstants import hidden_layers, activations, lr, optimizer, input_shape, grid_size


class NeuralNet:
    def __init__(self):
        input_layer = Input(shape=input_shape)
        # Define architecture
        x = input_layer
        assert len(hidden_layers) == len(
            activations), 'Different number of hidden layers and activations'
        for layer, activation in zip(hidden_layers, activations):
            x = Dense(layer, activation=activation)(x)
        self.anet = Model(input_layer, x)
        self.anet._name = 'ANET'

        # Compile model
        self.anet.compile(optimizer=optimizer,
                          loss='categorical_crossentropy', metrics=['accuracy'])
        # self.anet.summary()
        # input()

    def default_policy(self, possible_actions, state, player):
        """
        NOTE: predict takes 0.038 seconds, total runtime without assert is 0.038 (predict uses all)
        possible_actions: list of tuples
        state: ndarray
        player: tuple

        :returns: tuple, action
        """
        features = np.append(state, player).reshape((1, len(state)+1))
        action_probabilities = self.anet.predict(features)
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

    def train_on_rbuf(self, rbuf_X, rbuf_y, batch_size):
        # TODO: Get random subset of rbuf for training
        """
        rbuf consists of states+players, D (distributions over actions from states)
        """
        print('...training on {} samples'.format(rbuf_X.shape[0]))
        self.anet.fit(rbuf_X, rbuf_y, epochs=50,
                      verbose=1, batch_size=batch_size)
        #input('...PRESS ANY KEY TO CONTINUE...')

    def save_params(self, i):
        """
        Saves weights and biases of network to file

        :param i: int, round the params have been saved (becomes part of the filename)

        https://www.tensorflow.org/tutorials/keras/save_and_load
        """
        self.anet.save_weights('./checkpoints/save_{}'.format(i))
        print('...parameters for round {} have been saved to file'.format(i))

    def load_params(self, i):
        """
        Load weights and biases from file to the network

        :param i: int, round the params have been saved 
        """
        self.anet.load_weights('./checkpoints/save_{}'.format(i))
