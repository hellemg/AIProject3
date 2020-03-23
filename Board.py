import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from BoardCell import *

FILLED_VALUE = '*'
EMPTY_VALUE = 'o'


class Board:
    def __init__(self, board_type, grid_size, open_cells):
        self.number_of_pieces = None
        self.board_type = board_type
        self.grid_size = grid_size
        self.grid = np.empty((grid_size, grid_size), dtype=BoardCell)
        self.initialize_board()
        self.remove_pieces(open_cells)
        self.last_moved_from_coords = (-1,-1)
        self.last_moved_to_coords = (-1,-1)

    # TODO: Create only one Board-class, no subclasses
    # TODO: Represent cell-value numerically, __str__-method to display stars and os

    def get_last_moved_coords(self):
        return self.last_moved_coords

    def initialize_board(self):
        for i in range(self.grid_size):  # row
            for j in range(self.grid_size):  # column
                # For triangle
                if (self.board_type == 'D' or (self.grid_size - j) > i):
                    self.grid[i, j] = BoardCell(
                        (i, j), neighbour_list=self.get_neighbour_list(i, j))
                else:
                    self.grid[i, j] = -1
        self.number_of_pieces = self.grid_size*self.grid_size

    def get_neighbour_list(self, row, column):
        neighbour_list = [None, None, None, None, None, None]
        # List of coordinates for the neighbours of cell [row, column]
        if (row > 0):
            neighbour_list[0] = [row-1, column]
            if (column < self.grid_size - 1):
                neighbour_list[1] = [row-1, column+1]
        if (column > 0):
            neighbour_list[3] = [row, column-1]
        if (column < self.grid_size - 1):
            neighbour_list[2] = [row, column+1]
        if (row < self.grid_size - 1):
            neighbour_list[5] = [row+1, column]
            if (column > 0):
                neighbour_list[4] = [row+1, column-1]
        return np.array(neighbour_list)

    def get_board_state(self):
        """
        :returns: string, state of board
        """
        state = ''
        for i in range(self.grid_size):
            for row_value in self.grid[i]:
                if str(row_value) == '*':
                    state += '1'
                elif str(row_value) == 'o':
                    state += '0'
        return state

    def get_number_of_pieces(self):
        """
        :returns: int - number of pieces on the board
        """
        num_pieces = 0
        for row in self.grid:
            row = row[row != -1]
            for value in row:
                if value.get_value() == '*':
                    num_pieces += 1
        return num_pieces

    def get_game_status(self):
        all_legal_moves = self.find_all_legal_moves()
        num_legal_moves = len(all_legal_moves)
        num_pieces_on_board = self.get_number_of_pieces()
        if num_pieces_on_board == 1:
            #print('YOU HAVE WON! :D')
            return 'win'
        elif num_legal_moves == 0:
            #print('You lost :(')
            return 'loose'
        else:
            # print('Pieces left on board:', num_pieces_on_board)
            # print('-- number of legal moves:', num_legal_moves)
            # print('-- legal moves:', all_legal_moves)
            return 'play'

    def get_grid_size(self):
        return self.grid_size

    def display_board(self):
        for i in range(self.grid_size):
            print(self.grid[i])#[self.grid[i] != -1])

    def find_all_legal_moves(self):
        """
        :returns: list of all legal moves - tuples of (row, column, direction)
        """
        all_legal_moves = []
        for i in range(self.grid_size):
            row = self.grid[i][self.grid[i] != -1]
            legal_row = []
            for j in range(len(row)):
                legal_moves = self.get_legal_move_for_piece(i, j)
                all_legal_moves += legal_moves
        return all_legal_moves

    def get_legal_move_for_piece(self, row, column):
        """

        :returns: list of legal move for piece in position [row][column] - tuples of (row, column, direction)
        """
        # Assumes 'row' and 'column' have legal board-values
        legal_moves = []
        if self.grid[row, column].get_value() == EMPTY_VALUE:
            return legal_moves
        # Check all possible neighbours
        for i in range(6):
            neighbour_coords = self.grid[row, column].neighbour_list[i]
            # Can only jump over non-empty neighbours
            if neighbour_coords is not None:
                neighbour = self.grid[neighbour_coords[0], neighbour_coords[1]]
                # Triangle
                if neighbour != -1:
                    if neighbour.get_value() != EMPTY_VALUE:
                        neighbours_neighbour_coords = self.grid[neighbour_coords[0],
                                                        neighbour_coords[1]].neighbour_list[i]
                        if neighbours_neighbour_coords is not None:
                            if str(self.grid[neighbours_neighbour_coords[0], neighbours_neighbour_coords[1]]) == EMPTY_VALUE:
                                # Legal to move in direction i
                                legal_moves.append((row, column, i))
        return legal_moves

    def make_jump(self, from_row, from_column, direction):
        """
        Jump peg on position (from_row, from_column) two steps in 'direction'
        Remove piece between old position and new position

        :type from_row: int
        :param from_row: row of peg to be moved

        :type from_column: int
        :param from_column: column of peg to be moved

        :type direction: int
        :param direction: index in neighbour-list - represents direction in which to jump
        """
        # Get coordinates for new position and overjumped cell
        overjumped_row, overjumped_column = self.grid[from_row,
                                                      from_column].neighbour_list[direction]
        to_row, to_column = self.grid[overjumped_row,
                                      overjumped_column].neighbour_list[direction]
        # Move piece
        self.grid[to_row, to_column].set_value(self.grid[from_row,
                                                         from_column].get_value())
        # Remove piece from old position
        self.grid[from_row, from_column].set_value(EMPTY_VALUE)
        # Remove overjumped piece
        self.remove_pieces([[overjumped_row, overjumped_column]])
        self.last_moved_from_coords = (from_row, from_column)
        self.last_moved_to_coords = (to_row, to_column)

    def remove_pieces(self, coordinates):
        for [row, column] in coordinates:
            # For Triangle board
            if (self.grid[row, column] != -1):
                if self.grid[row, column].get_value() != EMPTY_VALUE:
                    self.grid[row, column].set_value(EMPTY_VALUE)
                    self.number_of_pieces -= 1

    def display_board_graph(self):
        # node network
        G = nx.Graph()
        node_colors = []
        # Add nodes to graph
        for r in range(self.grid_size):
            x = -r
            for c in range(self.grid_size):
                y = self.grid_size - r - c
                pos = (x, y)
                board_cell = self.grid[r, c]
                if board_cell != -1:
                    G.add_node(board_cell.get_coordinates(), pos=pos)
                    # if board_cell.get_coordinates() == self.last_moved_from_coords:
                    #     node_colors.append('black')
                    if board_cell.get_coordinates() == self.last_moved_to_coords:
                        node_colors.append('lightskyblue')
                    elif board_cell.get_value() != FILLED_VALUE:
                        node_colors.append('black')
                    else:
                        node_colors.append('lightblue')
                    x += 1

        # Add edges between neighbours
        for r in self.grid:
            for cell in r:
                if cell != -1:
                    for n in cell.get_neighbour_list():
                        if n is not None:
                            neighbour_cell = self.grid[n[0], n[1]]
                            if neighbour_cell != -1:
                                G.add_edge(cell.get_coordinates(),
                                           neighbour_cell.get_coordinates())

        # Plot
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, node_color=node_colors, node_size=3000,
                with_labels=True, font_weight='bold')
        #plt.show()
