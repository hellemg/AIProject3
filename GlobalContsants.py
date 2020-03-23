BOARD_SIZE = 4


num_episodes = 1000
# table or neural_net
critic_type = 'neural_net'

neural_network_layers = (16, 64, 32, 1)

learning_rate_actor = 0.005
learning_rate_critic = 0.0001

trace_decay_actor = 0.85
trace_decay_critic = 0.85

discount_factor_actor = 0.95
discount_factor_critic = 0.95

epsilon = 0.9
epsilon_decay_rate = 0.007

display_game = True
frame_delay = 100
