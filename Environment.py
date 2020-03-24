from Game import Nim, Ledge
from Board import Board
from GlobalConstants import *
import matplotlib.animation
import matplotlib.pyplot as plt


class Environment:
    def __init__(self):
        # Game type: nim or ledge
        self.game = Board(grid_size)
  
    def generate_initial_state(self):
        return self.game.get_initial_state()

    def generate_child_state_from_action(self, state, action, player, verbose=False):
        """
        :param state: board, either ndarray (ledge) or int (nim)
        :param action: tuple with action to do
        """
        return self.game.get_state_from_state_action(state, *action, player, verbose)

    def check_game_done(self, state):
        """
        :param state: board, either ndarray (ledge) or int (nim)

        :returns: boolean, True if the game is done
        """
        return self.game.check_game_done(state)

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
        if player_num == (1,0):
            return 1
        elif player_num == (0,1):
            return -1

    def draw_game(self, state):
        # Visualize a single state
        self.game.display_board_graph(state)

    def visualize(self, states, frame_delay):
        """
        Visualize a game given a list of states
        :param actions: A list of states (grids)
        """
        def act_and_visualize(i):
            self.game.display_board_graph(states[i])

        fig = plt.gcf()
        ani = matplotlib.animation.FuncAnimation(fig, act_and_visualize, frames=(len(states)), interval=frame_delay, repeat=False)
        plt.show()
