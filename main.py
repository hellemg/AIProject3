import numpy as np
import time
from GlobalConstants import *
from Game import *
from Environment import Environment
from MCTS import MCTS
from NeuralNet import NeuralNet
from utils import test_time
from TOPP import TOPP

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
                features = np.append(
                    state, player_number)
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
            if (j+1) % 10 == 0:
                neural_net.train_on_rbuf(
                    np.array(rbuf_X), np.array(rbuf_y), batch_size)
                rbuf_X = []
                rbuf_y = []

            # j begins at 0, so add 1
            if (j+1) % save_interval == 0:
                neural_net.save_params((j+1)*10)


        print('Player 1 wins {} of {} games ({}%).\nPlayer 1 started {}% of the time'.format(
            p1_wins, G, p1_wins/G*100, p1_start/G*100))

    elif Menu == 'TOPP':
        print('******* WELCOME TO THE TOURNAMENT *******')

        agents = []

        # NOTE: i: adam, lr = 0.001, 20 epochs
        # NOTE: i*10: adam, lr = 0.001, 50 epochs
        for i in [0, 125, 250]:  # np.linspace(0, G, num_caches, dtype=int):
            print('...fetching agent ', i)
            a = NeuralNet()
            a.load_params(i)
            a.anet._name = 'ANET_'+str(i)
            agents.append(a)

        topp = TOPP(agents)
        topp.tournament()

        """
        TODO:
        - DONE: Run games project 2-style - requires clean GCs, working env
        - DONE: Add NN to rollouts (ANET)
        - Add target policy update after each actual game (train NN)
        - Make list of architectual choices to try and train the network on
            - Send as input to NN, not from GC (just when testing this, use GC else)
            - Save each M trained anets with different names for different architectures
            - Decide how to measure success
            - Save success-measurement with each agent
            - Run small test to ensure it works
            - Run large test overnight: 5x5 board, 4 ANETS, minimum 200 episodes (Try 1000 simulations)
            - Log all results
            - Choose one architecture

        # NOTE: questions
        - time: batch size of 64 or 32, fyll opp fra starten i rbuf, tren etter hvert spill
        - anets are insecure to begin with (too large search-space to simulate to the end), but get more secure
            as the game gets closer to an end
        - ALTERNATE STARTING PLAYER WHEN PLAYING GAMES TO TRAIN ANETS?
        """
