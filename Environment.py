from Game import Nim, Ledge
from Board import Board
from GlobalConstants import *


class Environment:
    def __init__(self):
        # Game type: nim or ledge
        self.game = Board(grid_size)
  
    def generate_initial_state(self):
        return self.game.get_initial_state()

    def generate_child_state_from_action(self, state, action, player=(1,0), verbose=False):
        """
        :param state: board, either ndarray (ledge) or int (nim)
        :param action: tuple with action to do
        """
        return self.game.get_state_from_state_action(state, *action, player, verbose)

    def check_game_done(self, state, player):
        """
        :param state: board, either ndarray (ledge) or int (nim)

        :returns: boolean, True if the game is done
        """
        return self.game.check_game_done(state, player)

    def get_possible_actions_from_state(self, state):
        """
        :param state: board, either ndarray (ledge) or int (nim)

        :returns: list of possible actions
        """
        return self.game.get_possible_actions_from_state(state)

    def get_environment_value(self, player_num):
        """
        :param player_num: int, 1 for P1 and 2 for P2

        :returns: 1 if P1 won, -1 if P2 won
        """
        if player_num == 1:
            return 1
        elif player_num == 2:
            return -1

    def draw_game(self, state):
        self.game.display_board_graph(state)