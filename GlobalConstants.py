import numpy as np
from tensorflow.keras.optimizers import Adagrad, Adam, SGD, RMSprop

# NOTE: Player 1 is max-player
"""
In the TOPP, ANETs will be given game states and produce output distributions, which will support action
choices by their respective agents. ANETs will not do any learning during the TOPP.
"""
# Displaying games
visualize = False

# What program to run, 'Test': 'Testspace', 'M': 'MCTS', 'T': 'TOPP', 'P': 'Play against'
run_key = 'T'

# Hex game, [3,10]
grid_size = 5

# Number of games in a batch
G = 100

# Number of simulations (and therefore rollouts) for each move
M = 5000

# Number of ANETs to be cached for a TOPP - starting with an untrained net prior to episode 1
# NOTE: 1 < num_caches <= G+1
# NOTE: save_interval = int(np.floor(G/(num_caches-1)))
num_caches = 11

# Number of games to be played between any two ANET-agents in the TOPP. Should be even for fairness
num_games = 10
# Number of turnaments to run (showcases generalization)
num_tournaments = 4
# Randomness or greedy action
policy = 'best' #'default' #'best'

# grid size, number of games G, number of simulations M, number of caches
save_path = './rung_long_OTH2/6_100_5000_11_'#'./rung_long_OTH/6_200_5000_21_' 

load_path = './large_run/5_200_5000_10_round_' #'./large_run/5_200_5000_10_round_'

##### ANET PARAMETERS #####

# Learning rate
lr = 0.001

# Input shape to the network, one node per boardcell + 1 player-node
input_shape = grid_size**2+1

# Neurons per hidden layer and for the output layer
hidden_layers = [128, 64]+[grid_size**2]

# Activation functions (linear, sigmoid, tanh, relu)
activations = ['linear', 'relu']+['softmax']

# Optimizer (adagrad, Stochastic Gradient Descent (SGD), RMSProp, and Adam)
optimizer = {'adam': Adam(learning_rate=lr),
             'adagrad': Adagrad(learning_rate=lr),
             'sgd': SGD(learning_rate=lr),
             'rmsprop': RMSprop(learning_rate=lr)}['adam']


##### OTHER PARAMETERS #####

# ANET vs random rollout on leaf evaluation (speed up)
random_leaf_eval_fraction = 0.96
random_leaf_eval_decay =  0.8

# Player to start, P1: 1, P2: -1. Always use 1
P = 1
# Options for player
p1 = 1
p2 = -1

# Printing of each players move
verbose = False