import math
from BasicClientActorAbs import BasicClientActorAbs
from Environment import Environment
from NeuralNet import NeuralNet
import numpy as np

from GlobalConstants import grid_size as BCA_grid_size


class BasicClientActor(BasicClientActorAbs):

    def __init__(self, agent, env, IP_address=None, verbose=False):
        self.series_id = -1
        self.agent = agent
        self.env = env
        BasicClientActorAbs.__init__(self, IP_address, verbose=verbose)

    def handle_get_action(self, state):
        """
        Here you will use the neural net that you trained using MCTS to select a move for your actor on the current board.
        Remember to use the correct player_number for YOUR actor! The default action is to select a random empty cell
        on the board. This should be modified.
        :param state: The current board in the form (1 or 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), where
        1 or 2 indicates the number of the current player.  If you are player 2 in the current series, for example,
        then you will see a 2 here throughout the entire series, whereas player 1 will see a 1.
        :return: Your actor's selected action as a tuple (row, column)
        """
        # Get the board from the state
        board = np.array(state[1:])
        # Change 2s to -1s
        board = np.where(board == 2, -1, board)

        # Fix board
        my_player = self.series_id
        # Change to -1
        if my_player == 2:
            my_player = -1

        flip = False

        # If the other person starts and I am 1, then I have to invert and flip and become -1
        if self.starting_player != self.series_id and self.series_id == 1:
            board = self.env.flip_state(board)
            my_player = -1
            flip = True
        # If I start and I am -1, then I have to invert and flip the board and become 1
        if self.starting_player == self.series_id and self.series_id == 2:
            board = env.flip_state(board)
            my_player = 1
            flip = True

        # Find best action for my player, but with state and my_player as features
        possible_actions = self.env.get_possible_actions_from_state(board)
        best_action_index = self.agent.default_policy(possible_actions, board, my_player)

        # Convert index to row and column
        if flip:
            return (best_action_index % BCA_grid_size, best_action_index//BCA_grid_size)
        else:
            return (best_action_index//BCA_grid_size,
                    best_action_index % BCA_grid_size)

    def handle_series_start(self, unique_id, series_id, player_map, num_games, game_params):
        """
        Set the player_number of our actor, so that we can tell our MCTS which actor we are.
        :param unique_id - integer identifier for the player within the whole tournament database
        :param series_id - (1 or 2) indicating which player this will be for the ENTIRE series
        :param player_map - a list of tuples: (unique-id series-id) for all players in a series
        :param num_games - number of games to be played in the series
        :param game_params - important game parameters.  For Hex = list with one item = board size (e.g. 5)
        :return

        """
        self.series_id = series_id
        #############################
        #
        #
        # YOUR CODE (if you have anything else) HERE
        #
        #
        ##############################

    def handle_game_start(self, start_player):
        """
        :param start_player: The starting player number (1 or 2) for this particular game.
        :return
        """
        self.starting_player = start_player
        #############################
        #
        #
        # YOUR CODE (if you have anything else) HERE
        #
        #
        ##############################

    def handle_game_over(self, winner, end_state):
        """
        Here you can decide how to handle what happens when a game finishes. The default action is to print the winner and
        the end state.
        :param winner: Winner ID (1 or 2)
        :param end_state: Final state of the board.
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        ##############################
        print("Game over, winner was: {} (you are {}):".format(winner, self.series_id))
        #print('Winner: ' + str(winner))
        #print('End state: ' + str(end_state))

    def handle_series_over(self, stats):
        """
        Here you can handle the series end in any way you want; the initial handling just prints the stats.
        :param stats: The actor statistics for a series = list of tuples [(unique_id, series_id, wins, losses)...]
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("Series ended, these are the stats:")
        print(str(stats))

    def handle_tournament_over(self, score):
        """
        Here you can decide to do something when a tournament ends. The default action is to print the received score.
        :param score: The actor score for the tournament
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("Tournament over. Your score was: " + str(score))

    def handle_illegal_action(self, state, illegal_action):
        """
        Here you can handle what happens if you get an illegal action message. The default is to print the state and the
        illegal action.
        :param state: The state
        :param action: The illegal action
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("An illegal action was attempted:")
        print('State: ' + str(state))
        print('Action: ' + str(illegal_action))


if __name__ == '__main__':
    env = Environment(BCA_grid_size)
    agent = NeuralNet(BCA_grid_size**2+1)
    agent.load_params('./rung_long_OTH2/6_100_5000_11_100')

    bsa = BasicClientActor(agent, env)
    bsa.connect_to_server()
