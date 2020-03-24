import numpy as np

# Hex game
grid_size = 4

# NOTE: Player 1 is max-player

verbose = True

visualize = False

# Number of games in a batch
G = 10

# Number of simulations (and therefore rollouts) for each move
M = 8000

# Player to start, P1: (1,0), P2: (0,1)
P = (0,1)


# ANET parameters
lr = 0.0001

# Input shape to the network, two nodes per boardcell + 2 player-nodes 
input_shape = 2*grid_size**2+2

# Neurons per hidden layer and for the output layer
hidden_layers = [128, 64, 32, grid_size**2]

# Activation functions (linear, sigmoid, tanh, relu)
activations = ['relu','relu', 'relu', 'softmax']

# Optimizer (adagrad, Stochastic Gradient Descent (SGD), RMSProp, and Adam)
optimizer = 'adam'

# TOPP parameters

# Number of ANETs to be cached for a TOPP - starting with an untrained net prior to episode 1
num_caches = 3

# Number of games to be played between any two ANET-agents in the TOPP
num_games = 3

