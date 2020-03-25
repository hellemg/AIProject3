"""
Finally, in a post-training tournament, D might be used in a purely greedy fashion: the move with the highest probability is always 
chosen. This mirrors the general philosophy that the final target policy should follow a much more
exploitative than exploratory strategy. However, you may also continue to use it in an epsilon-greedy fashion or even
in a purely probabilistic manner, where moves are chosen stochastically based on the probabilities in D. Finding the
best such strategy may require a good deal of experimentation
"""

class TOPP:
    def __init__(self):
        pass

    def tournament(self):
        for i in range(num_games):
            print('gon play some game ')
