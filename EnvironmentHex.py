from Board import *
from GlobalConstants import *
import time
import matplotlib.animation
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, board_type, board_size, open_cells):
        self.game = Board(board_type,board_size, open_cells)

    def get_current_state(self):
        """
        Get string representing state of the game
        
        :returns: string, state of board
        """
        return self.game.get_board_state()


    def get_legal_actions(self):
        """
        Get the legal actions for the game

        :returns: tuple of ints - (row, column, direction)
        """
        return self.game.find_all_legal_moves()

    def emply_action(self, action):
        """
        Employs the action on the game

        :type action: tuple of ints
        :param action: peg to move and direction to move it
        """
        if blk:
            self.debug(action)
        self.game.make_jump(*action)

    def get_environment_status(self):
        """
        Gets the reward from each game status

        :returns: int, reward
        """
        game_status = self.game.get_game_status()
        if game_status == 'play':
            return 0
        elif game_status == 'win':
            return 10
        elif game_status == 'loose':
            return -1

    def get_environment_values(self):
        """
        Gets the pieces left on the board

        :returns: int, pegs left
        """
        return self.game.get_number_of_pieces()

    def display_env(self):
        # Draw board graphics
        self.game.display_board_graph()

    def display_game(self):
        # Terminal drawing of board
        self.game.display_board()


    """
    Visualize a game given a list of actions
    :param actions: A list of actions (tuples)
    """
    @staticmethod
    def visualize(actions, board_type, board_size, open_cells, frame_delay):
        def act_and_visualize(i):
            if i != 0:
                action = actions[i-1]
                game.make_jump(*action)
                print("Game status:", game.get_game_status())
                print("Remaining moves:", game.find_all_legal_moves())
            game.display_board_graph()

        print("Actions:", actions)
        game = Board(board_type, board_size, open_cells)
        fig = plt.gcf()
        ani = matplotlib.animation.FuncAnimation(fig, act_and_visualize, frames=(len(actions)+1), interval=frame_delay, repeat=False)
        plt.show()

    
    def debug(self, a):
        time.sleep(blk_time)