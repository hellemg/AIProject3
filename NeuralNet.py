import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model

from GlobalConstants import hidden_layers, activations, lr, optimizer, input_shape, grid_size

class NeuralNet:
    def __init__(self):
        input_layer = Input(shape=input_shape)
        # Define architecture
        x = input_layer
        for layer, activation in zip(hidden_layers, activations):
            x = Dense(layer, activation=activation)(x)
        self.anet = Model(input_layer, x)
        self.anet._name = 'ANET'

        # Compile model
        self.anet.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        # self.anet.summary()
        # input()

    def default_policy(self, possible_actions, state, player, epsilon):
        """
        possible_actions: list of tuples
        state: ndarray
        player: tuple

        :returns: tuple, action
        """
        features = self.convert_to_nn_input(state, player)
        action_probabilities = self.anet.predict(features)
        scaled_predictions = self.scale_actions(possible_actions, action_probabilities)
        return self.get_action(scaled_predictions, possible_actions, epsilon)

    def convert_to_nn_input(self, state, player):
        """
        state: ndarray
        player: tuple

        :returns: features, flattened ndarray
        """
        # Convert state and player to input for ANET
        state = np.array([bit for bit in state.flatten()])
        features = np.hstack((state.flatten(),np.array(player)))
        return features.reshape(1, features.shape[0])

    def scale_actions(self, possible_actions, action_probabilities):
        # Make impossible actions have probability 0
        action_probabilities = action_probabilities.reshape(grid_size, grid_size)
        scaled_predictions = []

        # Keep only possible actions
        for inds in possible_actions:
            scaled_predictions.append(action_probabilities[inds])
        divider = np.sum(scaled_predictions)
        # Things (I can think of) that makes divider == 0 is: no possible actions. anet only gives non-zero values for illegal moves
        if divider == 0:
            scaled_predictions = np.ones(len(possible_actions))/len(possible_actions)
        else:
            assert divider > 0, 'Divider in scaling action_predictions is 0'
            scaled_predictions = scaled_predictions/np.sum(scaled_predictions)
        return scaled_predictions

    def get_action(self, scaled_predictions, possible_actions, epsilon):
        """
        shape of scaled == shape of possible
        """
        # Returns action with highest probability (or random if epsilon-greedy)
        # Random index from 0 to last index in possible actions
        if len(possible_actions) == 1:
            return possible_actions[0]
        
        # Find the best action 
        best_action = np.where(scaled_predictions == np.max(scaled_predictions))[0][0]
        best_action = possible_actions[best_action]

        # If there is only one action, return it. Return best action with probability 1-epsilon
        if random.random() > epsilon:
            return best_action
        
        # Find actions that are not optimal
        non_best_actions = [a for a in possible_actions if a != best_action]
        random_index = np.random.randint(len(non_best_actions))
        # Return random action with probability epsilon
        return non_best_actions[random_index]

    def train_on_rbuf(self, rbuf):
        # TODO: Get random subset of rbuf for training
        # TODO: model.fit(features, D) - features is X, D is y
        """
        rbuf consists of states+players, D (distributions over actions from states)
        """


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

