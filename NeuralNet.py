import tensorflow as tf

from GlobalConstants import hidden_layers, activations, lr, optimizer, input_shape

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

    def convert_to_nn_input(self, state, player):
        """
        state: ndarray
        player: tuple

        :returns: features, flattened ndarray
        """
        # Convert state and player to input for ANET
        print('shape of state: {}\nplayer: {}'.format(state.shape, player))
        features = np.hstack((state.flatten(),np.array(player)))
        print('shape of features:', features.shape)
        return features

    def target_policy(self, features):
        """
        :param features: ndarray, flattened state with player

        :returns: predictions, probability for each action. Impossible actions set to 0
        """
        preds = self.anet.predict(features)
        print(preds)
        # TODO: features with filled value needs to have preds==0 (cant do the action)
        return preds

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
        raise NotImplementedError

    def load_params(self, i):
        """
        Load weights and biases from file to the network

        :param i: int, round the params have been saved 
        """
        self.anet.load_weights('./checkpoints/save_{}'.format(i))

