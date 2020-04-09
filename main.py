import numpy as np
import time
from GlobalConstants import *
from Game import *
from Environment import Environment
from MCTS import MCTS
from NeuralNet import NeuralNet
from utils import test_time
from TOPP import TOPP
#from Client_side import BasicClientActor

import matplotlib.pyplot as plt

if __name__ == '__main__':
    Menu = {
        'Test': 'Testspace',
        'M': 'MCTS',
        'T': 'TOPP',
    }['T']

    if Menu == 'Testspace':
        print('Welcome to testspace')

        # bca = BasicClientActor()
        # bca.connect_to_server()

    elif Menu == 'MCTS':
        print('Welcome to MCTS')

        # Rounds to save parameters for ANET
        save_interval = int(np.floor(G/(num_caches-1)))
        # Only use random leaf evaluation before the first training has happened
        ane = 1
        p1_wins = 0
        p1_start = 0
        neural_net = NeuralNet(grid_size**2+1)

        # List of training-data, rbuf_size x features
        rbuf_X = np.empty((1000, input_shape), dtype=np.ndarray)
        # List of target-data
        rbuf_y = np.empty((1000, grid_size*grid_size), dtype=np.ndarray)
        # Counter for position in rbuf
        i = 0
        # Batch size for training
        batch_size = 64
        # Flag indicating if the neural net should train
        train = False

        # Save model before training
        neural_net.save_params(save_path+str(0))
        for j in range(G):
            start_time = time.time()

            env = Environment(grid_size)
            mcts = MCTS(env, neural_net, ane)
            print('...using {}% ANET evaluation'.format(
                np.round((1-ane)*100, 3)))
            states_in_game = []
            state = env.generate_initial_state()
            states_in_game.append(state)
            # Initiate starting player for each game (should always be 1)
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
                rbuf_X[i % 1000] = features
                rbuf_y[i % 1000] = D
                # Increase counter
                i += 1

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

            # Do not train until the rbuf has filled up to batch size
            # After the rbuf has filled to batch size the first time, train after every game
            if i >= batch_size and train == False:
                # Turn on leaf evaluation with anet (one time)
                print('...evaluating leaf nodes with ANET')
                ane = random_leaf_eval_fraction
                # Turn on training of anet
                print(
                    '...turning on training, there are now {} examples to train on'.format(i))
                train = True

            # Train the neural net
            if train:
                filled_rows_lenght = rbuf_X[(
                    rbuf_X != np.array(None)).any(axis=1)].shape[0]
                random_rows = np.random.choice(
                    filled_rows_lenght, batch_size, replace=False)
                # Get the same rows from X and y
                train_X = rbuf_X[random_rows].astype(float)
                train_y = rbuf_y[random_rows].astype(float)
                neural_net.train_on_rbuf(train_X, train_y, batch_size)
                # Decay anet_fraction
                ane *= random_leaf_eval_decay

            # j begins at 0, so add 1
            if (j+1) % save_interval == 0:
                neural_net.save_params(save_path+str(j+1))

            print('...time for this game-run: {}'.format(time.time()-start_time))

        print('Player 1 wins {} of {} games ({}%).\nPlayer 1 started {}% of the time'.format(
            p1_wins, G, p1_wins/G*100, p1_start/G*100))

    elif Menu == 'TOPP':
        print('******* WELCOME TO THE TOURNAMENT *******')

        agents = []
        env = Environment(grid_size)
        state = env.generate_initial_state()
        features = np.append(state, P).reshape((1, len(state)+1))
        agent_numbers = np.linspace(0, G, num_caches, dtype=int)

        # NOTE: i: adam, lr = 0.001, 20 epochs
        # NOTE: i*10: adam, lr = 0.001, 50 epochs
        # np.linspace(0, G, num_caches, dtype=int):
        for i in [0, 50, 100, 150, 200]:
            print('...fetching agent ', i)
            a = NeuralNet(input_shape)
            a.load_params(load_path+str(i))
            a.anet._name = 'ANET_'+str(i)
            agents.append(a)

        #     print(a.anet(features))

        # p2_moves = np.array([1, 4, 5])
        # p1_moves = np.array([2, 3, 6])
        # p2_moves = np.array([2, 6])
        # p1_moves = np.array([4, 7])

        # state[p2_moves] = -1
        # state[p1_moves] = 1
        # # 6 is win for p1
        # state = np.array([0,  1, -1,  1,  1, -1,  0,  1, -1,  1,
        #                   1, -1, -1, -1, -1,  1,  1,  1, -1, -1,  1,  0,  1, -1, -1])
        # features = np.append(state, P).reshape((1, len(state)+1))
        # env.draw_game(state)
        # plt.show()

        # for i in range(1):
        #     for a in agents:
        #         print('.....', a.anet._name)
        #         print(a.anet(features))
        #         print(a.default_policy([], state, 1))
        # input()
        topp = TOPP(agents)
        topp.tournament()
        topp.display_results()
        #topp.several_tournaments()

        #topp.tournament()
        #topp.display_results()

        """
        TODO:
        - Change state in TOPP as done in BCA, so GC can have odd number of games
        - Do through project description (not OTH)
        - Go through pivotal parameters
        - Go through deliverables
        - Connect to OTH-server and test
            - Print-test everything you have in BCA
            - Smartest agent
            - Stupidest agent
            - One agent in between
        """
