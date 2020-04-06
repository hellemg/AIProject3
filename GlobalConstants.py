import numpy as np
from tensorflow.keras.optimizers import Adagrad, Adam, SGD, RMSprop

# NOTE: Player 1 is max-player

# Hex game
grid_size = 3

verbose = False

visualize = False

# Flag to change save_name of NNs
is_live_demo = False

# Number of games in a batch
G = 210

# Number of simulations (and therefore rollouts) for each move
M = 500

# Player to start, P1: 1, P2: -1. Always use 1
P = 1
# Options for player
p1 = 1
p2 = -1

# ANET parameters
lr = 0.001

# Input shape to the network, one nodes per boardcell + 1 player-node
input_shape = grid_size**2+1

# Neurons per hidden layer and for the output layer
hidden_layers = [64, 32]+[grid_size**2]

# Activation functions (linear, sigmoid, tanh, relu)
activations = ['linear', 'relu']+['softmax']

# Optimizer (adagrad, Stochastic Gradient Descent (SGD), RMSProp, and Adam)
optimizer = {'adam': Adam(learning_rate=lr),
             'adagrad': Adagrad(learning_rate=lr),
             'sgd': SGD(learning_rate=lr),
             'rmsprop': RMSprop(learning_rate=lr)}['adam']


# TOPP parameters

# Number of ANETs to be cached for a TOPP - starting with an untrained net prior to episode 1
num_caches = 4

# Number of games to be played between any two ANET-agents in the TOPP. Should be even for fairness
num_games = 4

# ANET vs random rollout on leaf evaluation (speed up)
random_leaf_eval_fraction = 0.8
random_leaf_eval_decay =  0.8
