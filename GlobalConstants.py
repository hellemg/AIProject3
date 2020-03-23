import numpy as np

# Hex game
grid_size = 3

# NOTE: Player 1 is max-player

verbose = True

visualize = True

# Number of games in a batch
G = 10

# Number of simulations (and therefore rollouts) for each move
M = 1000

# Player to start, P1: (1,0), P2: (0,1)
P = (1,0)
