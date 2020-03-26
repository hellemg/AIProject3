import numpy as np
import time
from GlobalConstants import *
from Game import *
from Environment import Environment
from MCTS import MCTS
from NeuralNet import NeuralNet
from utils import test_time

import matplotlib.pyplot as plt

if __name__ == '__main__':
    Menu = {
        'Test': 'Testspace',
        'M': 'MCTS',
        'T': 'TOPP',
    }['T']

    if Menu == 'Testspace':
        print('Welcome to testspace')

        env = Environment()
        nn = NeuralNet()
        mcts = MCTS(env, nn, 0.5)
        
        state = env.generate_initial_state()
        actions = env.get_possible_actions_from_state(state)



        # env = Environment()
        # s = env.generate_initial_state()
        # print(env.game.cell_neighbours[0])
        # print(env.game.cell_neighbours[1])
        # print(env.game.cell_neighbours[3])
        # print(env.game.cell_neighbours[8])
        # # Nothing has really happened, so begin with p2
        # p1 = 1
        # p2 = -1
        # env.draw_game(s)
        # plt.show()
        # i = 0
        # states_in_game = []
        # while not env.check_game_done(s):
        #     print('****************')
        #     player ^= (p1 ^ p2)
        #     actions = env.get_possible_actions_from_state(s)
        #     print('possible actions:', actions)
        #     action = actions[0]
        #     print('Player {} does action {}'.format(player, action))
        #     s = env.generate_child_state_from_action(s, action, player)
        #     states_in_game.append(s)
        #     i += 1
        #     env.game.print_board(s)
        # print('Player {} won'.format((i+1) % 2+1))
        # env.visualize(states_in_game, 750)

    elif Menu == 'MCTS':
        print('Welcome to MCTS')

        # Rounds to save parameters for ANET
        save_interval = int(np.floor(G/(num_caches-1)))
        # TODO: Save parameters for starting ANET (round 0, no training has occured)
        # TODO: Clear RBUF
        # TODO: Train (and switch to ANET) when RBUF is large enough
        # -- When RBUF has reached size so that train_buf_size*rbuf_size = batch_size, train and switch to ANET
        ane = random_leaf_eval_fraction
        p1_wins = 0
        p1_start = 0
        neural_net = NeuralNet()

        # List of training-data
        rbuf_X = []
        # List of target-data
        rbuf_y = []
        # Batch size for training
        batch_size = 128

        # Save model before training
        neural_net.save_params(0)
        for j in range(G):
            env = Environment()
            mcts = MCTS(env, neural_net, ane)
            print('...using {}% ANET evaluation'.format(
                np.round((1-ane)*100, 3)))
            states_in_game = []
            state = env.generate_initial_state()
            states_in_game.append(state)
            player_number = P
            # Player add 1 if player_number is 1 (P1 starts)
            p1_start += player_number + 1 and 1
            while not env.check_game_done(state):
                possible_actions = env.get_possible_actions_from_state(state)
                # Do M simulations
                best_action, D = mcts.simulate(player_number, M, state)

                # Add tuple of training example-data and target to RBUF
                features = np.append(state, player_number).reshape(len(state)+1)
                rbuf_X.append(features)
                rbuf_y.append(D)

                # Do the action, get next state
                state = env.generate_child_state_from_action(
                    state, best_action, player_number, verbose)
                states_in_game.append(state)
                # Next players turn
                player_number ^= (p1 ^ p2)
            # Winner was the last player to make a move (one before player_number)
            winner = player_number ^ (p1 ^ p2)
            if verbose:
                print('Player {} wins'.format(winner))
            if winner == 1:
                p1_wins += 1
            print('*** Game {} done ***'.format(j+1))
            if visualize:
                env.visualize(states_in_game, 500)

            # Decay anet_fraction
            ane *= random_leaf_eval_decay

            # TODO: Train anet on random minibatch of cases from RBUF (in method)
            # Train when batch_size is large enough, reset rbuf
            if len(rbuf_X)//batch_size >= 1:
                neural_net.train_on_rbuf(np.array(rbuf_X), np.array(rbuf_y), batch_size)
                rbuf_X = []
                rbuf_y = []

            # j begins at 0, so add 1
            if (j+1) % save_interval == 0:
                neural_net.save_params(j+1)

        print('Player 1 wins {} of {} games ({}%).\nPlayer 1 started {}% of the time'.format(
            p1_wins, G, p1_wins/G*100, p1_start/G*100))

    elif Menu == 'TOPP':
        print('******* WELCOME TO THE TOURNAMENT *******')

        agents = []

        # TODO: Rank agents after how many total games they win

        for i in range(num_caches):
            # TODO: Create agent for each saved parameters-file. Need Agent-class
            # New suggestion: give ANETS states and get back action-probabilities. Create agents that have choose_action_method
            print('...TODO: create agent')
            # a = 'agent'+str(i+1)
            a = NeuralNet()
            a.load_params(i+1)
            a.anet._name = 'ANET'+str(i+1)
            agents.append(a)

        for i in range(len(agents)):
            # Each agent plays against all agents after it in the list `agents`
            first_agent = agents[i]
            for j in range(i+1, len(agents)):
                second_agent = agents[j]
                print('TOURNAMENT BETWEEN: ', first_agent, second_agent)
                print('...play {} games'.format(num_games))
                first_agent_wins = 2
                print('{} won {} of {} games ({}%).'.format(
                    first_agent, first_agent_wins, num_games, np.round(first_agent_wins/num_games*100, 2)))

        """

        - DONE: Run games project 2-style - requires clean GCs, working env
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
