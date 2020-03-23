import numpy as np

from GlobalConstants import *
from Game import *
from Environment import Environment
from MCTS import MCTS

if __name__ == '__main__':
    Menu = {
        'T': 'Testspace',
        'M': 'MCTS',
    }['T']

    if Menu == 'Testspace':
        print('Welcome to testspace')
        env = Environment()
        s = env.generate_initial_state()
        env.game.display_board(s)
        print(env.game.cell_neighbours[0,1])
        actions = env.get_possible_actions_from_state(s)
        print(actions)
        for i in range(5):
            print('****************')
            player = ((i+1)%2, i%2)
            values = input("Input some comma seprated numbers : ")
            action = tuple(map(int,values.split(',')))
            print('Player {} does action {}'.format(player, action))
            s = env.generate_child_state_from_action(s, action, player)
            env.draw_game(s)
            did_win = env.check_game_done(s, player)
            print(did_win)

    elif Menu == 'MCTS':
        print('Welcome to MCTS')

        def get_player_number(p_num):
            # Return randomly P1 or P2
            if p_num == 3:
                return np.random.randint(2)
            # Defined p_num to be 0 for P1 and 1 for P2
            else:
                return p_num-1

        def get_init(game_type):
            if game_type == 'nim':
                return N
            elif game_type == 'ledge':
                return B_init

        p1_wins = 0
        p1_start = 0
        for j in range(G):
            env = Environment(game_type)
            mcts = MCTS(env)
            state = get_init(game_type)
            player_number = get_player_number(P)
            # Player number is 0 for P1 and 1 for P2
            p1_start += ((player_number+1)%2)
            while not env.check_game_done(state):
                possible_actions = env.get_possible_actions_from_state(state)
                # Do M simulations
                best_action = mcts.simulate(player_number, M, state)
                # Do the action, get next state
                state = env.generate_child_state_from_action(state, best_action, player_number, True)
                # Next players turn
                player_number += 1
            winner = (player_number-1) % 2+1
            if verbose:
                print('Player {} wins'.format(winner))
            if winner == 1:
                p1_wins += 1
            print('*** Game {} done ***'.format(j+1))
        
        print('Player 1 wins {} of {} games ({}%).\nPlayer 1 started {}% of the time'.format(p1_wins, G, p1_wins/G*100, p1_start/G*100))

        """
        TODO: 
        - Create board + environment

        Default policy = behaviour policy = target policy - neural net

        Neural net:
        - input is a board state, output is a probability distribution over all possble moves (from the input state)
        - to create an intelligent target policy that can be used without MCTS

        RL Procedure:
        1. Making moves in the actual game, each game is an episode
        2. Making moves in simulated game (search moves), each simulated game is a search game
        3. Update target policy with supervised learning, training cases are from visit counts of arcs in the MC tree (updates happen
            at the end of an actual game)
        """