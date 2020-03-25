import numpy as np

from GlobalConstants import *
from Game import *
from Environment import Environment
from MCTS import MCTS
from NeuralNet import NeuralNet

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
        neural_net = NeuralNet()
        state = env.generate_initial_state()
        state[0,2] = (1,0)
        state[1,1] = (1,0)
        state[1,0] = (0,1)
        state[1,2] = (0,1)
        print(state)
        env.draw_game(state)
        plt.show()
        # Wins are (2,0) and (2,1)

        X = np.array([0, 0, 0, 0, 1, 0,
                      0, 1, 1, 0, 0, 1,
                      0, 0, 0, 0, 0, 0,
                      1, 0]).reshape(1, 20)
        neural_net.load_params(0)
        pred1 = neural_net.anet.predict(X)
        neural_net.load_params(125)
        pred2 = neural_net.anet.predict(X)
        neural_net.load_params(250)
        pred3 = neural_net.anet.predict(X)
        print(np.round(pred1,3))
        print(np.round(pred2,3))
        print(np.round(pred3,3))
        print(np.sum(pred1))
        print(np.sum(pred2))
        print(np.sum(pred3))

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

        # Rounds to save parameters for ANET
        save_interval = int(np.floor(G/(num_caches-1)))

        # TODO: Save parameters for starting ANET (round 0, no training has occured)
        # TODO: Clear RBUF
        eps = epsilon
        ane = random_leaf_eval_fraction
        p1_wins = 0
        p1_start = 0
        neural_net = NeuralNet()
        # Save model before training
        neural_net.save_params(0)
        for j in range(G):
            env = Environment()
            mcts = MCTS(env, neural_net, eps, ane)
            # RBUF with room for one example per game-move
            rbuf_X = []
            rbuf_y = []
            print('...using {}% ANET evaluation'.format(
                np.round((1-ane)*100, 3)))
            states_in_game = []
            state = env.generate_initial_state()
            states_in_game.append(state)
            player_number = P
            # Player number is (1,0) for P1 and (0,1) for P2
            p1_start += player_number[0]
            while not env.check_game_done(state):
                possible_actions = env.get_possible_actions_from_state(state)
                # Do M simulations
                best_action, D = mcts.simulate(player_number, M, state)

                # Add tuple of training example-data and target to RBUF
                features = neural_net.convert_to_nn_input(state, player_number)
                rbuf_X.append(features.reshape(20))
                rbuf_y.append(D)

                # Do the action, get next state
                state = env.generate_child_state_from_action(
                    state, best_action, player_number, verbose)
                states_in_game.append(state)
                # Next players turn
                player_number = (player_number[1], player_number[0])
            # Winner was the last player to make a move (one before player_number)
            winner = (player_number[1], player_number[0])
            if verbose:
                print('Player {} wins'.format(winner))
            if winner == (1, 0):
                p1_wins += 1
            print('*** Game {} done ***'.format(j+1))
            if visualize:
                env.visualize(states_in_game, 500)

            # Decay epsilon and anet_fraction
            #eps *= epsilon_decay
            ane *= random_leaf_eval_decay

            # TODO: Train anet on random minibatch of cases from RBUF (in method)
            neural_net.train_on_rbuf(np.array(rbuf_X), np.array(rbuf_y))

            # j begins at 0, so add 1
            if (j+1) % save_interval == 0:
                neural_net.save_params(j+1)

        print('Player 1 wins {} of {} games ({}%).\nPlayer 1 started {}% of the time'.format(
            p1_wins, G, p1_wins/G*100, p1_start/G*100))

    elif Menu == 'TOPP':
        print('******* WELCOME TO THE TOURNAMENT *******')

        agents = []

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
