from GlobalContsants import *

blk = False
blk_time = 0
if critic_type == 'neural_net' and BOARD_SIZE == 5 and BOARD_TYPE == 'T':
    critic_type = 'table'
    blk = True
    blk_time = 0.003
    learning_rate_actor = 0.9
    learning_rate_critic = 0.9

# OVERWRITES HERE
