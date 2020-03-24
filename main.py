import numpy as np

from GlobalConstants import *
from Game import *
from Environment import Environment
from MCTS import MCTS
import matplotlib.pyplot as plt

if __name__ == '__main__':
    Menu = {
        'T': 'Testspace',
        'M': 'MCTS',
    }['M']

    if Menu == 'Testspace':
        print('Welcome to testspace')

        env = Environment()
        s = env.generate_initial_state()
        states = [s.copy()]
        s[0,1] = (1,0)
        states.append(s.copy())
        print(env.check_game_done(s))   
        s[1,0] = (0,1)
        states.append(s.copy())
        print(env.check_game_done(s))   
        s[1,1] = (1,0)
        states.append(s.copy())
        print(env.check_game_done(s))   
        s[2,1] = (0,1)
        states.append(s.copy())
        print(env.check_game_done(s))   
        s[1,2] = (1,0)
        states.append(s.copy())
        print(env.check_game_done(s))   
        s[2,2] = (0,1)
        states.append(s.copy())
        print(env.check_game_done(s))   
        s[1,3] = (1,0)
        states.append(s.copy())
        print(env.check_game_done(s))   
        s[3,3] = (0,1)
        states.append(s.copy())
        print(env.check_game_done(s))   
        s[2,3] = (1,0)
        states.append(s.copy())
        print(env.check_game_done(s))   
        s[2,0] = (0,1)
        states.append(s.copy())
        print(env.check_game_done(s))   
        s[3,2] = (1,0)
        states.append(s.copy())
        print(env.check_game_done(s))   
        
        env.visualize(states, 500)
        # env = Environment()
        # s = env.generate_initial_state()
        # # Nothing has really happened, so begin with p2
        # player = (0,1)
        # env.game.print_board(s)
        # actions = env.get_possible_actions_from_state(s)
        # i = 0
        # states_in_game = []
        # while not env.check_game_done(s, player):
        #     print('****************')
        #     player = ((i+1)%2, i%2)
        #     actions = env.get_possible_actions_from_state(s)
        #     action = actions[0]
        #     print('Player {} does action {}'.format(player, action))
        #     s = env.generate_child_state_from_action(s, action, player)
        #     states_in_game.append(s)
        #     i += 1
        #     #env.draw_game(s)
        # print('Player {} won'.format((i+1)%2+1))
        # env.visualize(states_in_game, 1000)

    elif Menu == 'MCTS':
        print('Welcome to MCTS')

        p1_wins = 0
        p1_start = 0
        for j in range(G):
            env = Environment()
            mcts = MCTS(env)
            states_in_game = []
            state = env.generate_initial_state()
            states_in_game.append(state)
            player_number = P
            # Player number is (1,0) for P1 and (0,1) for P2
            p1_start += player_number[0]
            while not env.check_game_done(state):
                possible_actions = env.get_possible_actions_from_state(state)
                # Do M simulations
                best_action = mcts.simulate(player_number, M, state)

                # TODO: Get D back from MCTS (See TODO in MCTS). Save to RBUF

                # Do the action, get next state
                state = env.generate_child_state_from_action(state, best_action, player_number, True)
                states_in_game.append(state)
                # Next players turn
                player_number = (player_number[1], player_number[0])
            # Winner was the last player to make a move (one before player_number)
            winner = (player_number[1], player_number[0])
            if verbose:
                print('Player {} wins'.format(winner))
            if winner == (1,0):
                p1_wins += 1
            print('*** Game {} done ***'.format(j+1))
            if visualize:
                env.visualize(states_in_game, 500)

            # TODO: Train anet on random minibatch of cases from RBUF
            # TODO: Save anet's parameters if save-condition
        
        print('Player 1 wins {} of {} games ({}%).\nPlayer 1 started {}% of the time'.format(p1_wins, G, p1_wins/G*100, p1_start/G*100))

        """
        TODO: 
        **** P1 only wins 30% no matter who starts. Could be easier to go NW to SE because of my default action-pickers???

        NOTE: files to add
        - neural network(s)
        - TOPP
        - Have environment (and Hex)
        - Have MCTS

        - Run games project 2-style - requires clean GCs, working env
        - Add NN to rollouts (ANET)
        - Add target policy update after each actual game

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