import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Board:
    def __init__(self, grid_size):
        # FIXED
        self.grid_size = grid_size
        # Array holding edge-representation of grid
        self.cell_neighbours = np.empty(grid_size*grid_size, dtype=list)
        # self.cell_neighbours = np.empty((grid_size, grid_size), dtype=list)
        for i in range(grid_size*grid_size):
            # for j in range(grid_size):
            self.cell_neighbours[i] = self.get_neighbour_list(i)

    def get_initial_state(self):
        # FIXED
        # Values of each cell will be 0.
        return np.zeros(self.grid_size*self.grid_size, dtype=int)

    def get_possible_actions_from_state(self, grid):
        # FIXED
        """
        Finds all possible actions from a given state (action = empty boardcell)
        grid: ndarray, grid in some state

        :returns: ndarray of ints, indices where a piece can be put
        """
        return np.where(grid == 0)[0]

    def get_state_from_state_action(self, grid, ind, player, verbose):
        # FIXED
        """
        grid: ndarray, grid in some state
        row: int, row to place piece on
        column: int, column to place piece on
        player: tuple, (1,0) for p1, (0,1) for p2. Also is the new value
        """
        temp_grid = grid.copy()
        # Fill grid-cell with appropriate tuple
        temp_grid[ind] = player
        if verbose:
            print('Player {} places piece on ({})'.format(
                player, ind))
        return temp_grid

    def check_game_done(self, grid):
        # FIXED
        """
        grid: ndarray, grid in some state

        returns: boolean, True if grid is winning state
        """
        # P1: path across rows (northeast to southwest), P2 path spanning columns (northwest to southeast)
        """
        Strategies:
        1. check north (east or west, depending on player)
        -- NE: (0, c), c is any column (top of matrix)
        -- NW: (r, 0), r is any row (left of matrix)
        2. find first value that corresponds to player
        3. breadth-first search or similar to see if there is a path to the other side
        4. GOAL: reach board-cell with coordinates south(west or east)
        -- SW: (grid_size-1, c), c is any column (bottom of matrix)
        -- SE: (r, grid_size-1), r is any row (right of matrix)
        """
        # Check for P1 - Upper right to lower left
        to_visit = []
        for i in range(self.grid_size):
            # Check upper right edge (0,1,2) for boardcells with 1-value
            if grid[i] == 1:
                to_visit.append(i)
        if self.check_path(grid, to_visit, 1):
            return True
        # Check for P2 - Upper left to lower right
        to_visit = []
        for i in range(self.grid_size):
            # Check upper left edge (0,3,6) for boardcells with -1-value
            if grid[i*self.grid_size] == -1:
                to_visit.append(i*self.grid_size)
        if self.check_path(grid, to_visit, -1):
            return True
        return False

    def check_path(self, grid, to_visit, player):
        # FIXED
        
        """
        :param grid: ndarray, grid in some state
        :param to_visit: list of ints, coordinates on grid to visit
        :param player: int, 1 for P1, -1 for P2
        """
        # Go through all to_visit-coordinates until none left
        visited_cells = []
        while to_visit:
            current_ind = to_visit.pop()
            # Check if current_cell is on the other side for P1 (row max)
            if player == 1 and current_ind//self.grid_size == self.grid_size-1:
                return True
            # Check if current_cell is on the other side for P2 (column max)
            if player == -1 and current_ind % self.grid_size == self.grid_size-1:
                return True
            # Find neighbouring cells to check for a path of `player`-cells
            neighbours = self.cell_neighbours[current_ind]
            for n_coords in neighbours:
                # Dont go outside grid, dont go back to earlier visited cells
                if (n_coords is not None) and (n_coords not in visited_cells):
                    # Piece continuing the trail, add to to_visit
                    if grid[n_coords] == player:
                        to_visit.append(n_coords)
            # Make sure not to go back to current cell
            visited_cells.append(current_ind)
        # Did not find a path
        return False

    def get_neighbour_list(self, ind):  # row, column):
        # FIXED
        # Input: ind - index in 1d array holding board
        neighbour_list = [None, None, None, None, None, None]
        row = ind // self.grid_size
        column = ind % self.grid_size
        # List of coordinates for the neighbours of cell [row, column]
        if (row > 0):
            neighbour_list[0] = ind-self.grid_size
            if (column < self.grid_size - 1):
                neighbour_list[1] = ind-self.grid_size + 1
        if (column > 0):
            neighbour_list[3] = ind - 1
        if (column < self.grid_size - 1):
            neighbour_list[2] = ind + 1
        if (row < self.grid_size - 1):
            neighbour_list[5] = ind + self.grid_size
            if (column > 0):
                neighbour_list[4] = ind + self.grid_size - 1
        return neighbour_list

    def print_board(self, grid):
        # FIXED
        print('-------------------')
        for i in range(self.grid_size):
            print(grid[i*self.grid_size:i*self.grid_size+self.grid_size])

    def display_board_graph(self, grid):
        # FIXED
        # node network
        G = nx.Graph()
        node_colors = []
        # Add nodes to graph
        for r in range(self.grid_size):
            x = -r
            y = -r
            for c in range(self.grid_size):
                pos = (x, y)
                board_cell = grid[r*self.grid_size + c]
                G.add_node((r*self.grid_size + c), pos=pos)
                if board_cell == 0:
                    node_colors.append('lightgrey')
                elif board_cell == 1:
                    # Player 1
                    node_colors.append('lightskyblue')
                elif board_cell == -1:
                    # Player 2
                    node_colors.append('magenta')
                else:
                    ValueError('Unknown value in boardcell ({},{}): {}'.format(
                        x, y, board_cell))
                x += 1
                y -= 1

        # Add edges between neighbours
        for i in range(self.grid_size*self.grid_size):
            for n in self.cell_neighbours[i]:
                if n is not None:
                    # Add edge between current node and neighbour node
                    G.add_edge((i), (n))
        # Plot
        pos = nx.get_node_attributes(G, 'pos')

        # TODO: Fix node size so it works with both grid_size = 3 and grid_size = 10

        nx.draw(G, pos, node_color=node_colors, node_size=3000,
                with_labels=True, font_weight='bold')

        # TODO: Comment out plt.show() when animating

        # plt.show()


# class Board:
#     def __init__(self, K):
#         # K as defined in GC
#         self.K = K

#     def get_initial_state(self):
#         return 75

#     def get_state_from_state_action(self, total_pieces, num_pieces, player, verbose):
#         """
#         :param num_pieces: int, pieces getting picked up

#         :returns: int, pieces left on the board
#         """
#         if verbose:
#             print('Player {} selects {} stones. Remaining stones = {}'.format(
#                 player, num_pieces, total_pieces-num_pieces))
#         return total_pieces-num_pieces

#     def check_game_done(self, total_pieces):
#         """
#         :param total_pieces: int, pieces on the board

#         returns: boolean, if all pieces have been picked up
#         """
#         return total_pieces == 0

#     def get_possible_actions_from_state(self, total_pieces):
#         """
#         :param total_pieces: int, pieces on the board

#         :returns: list tuples with possible actions to take
#         """
#         # returns list of possible number of pieces to pick up
#         max_pieces = np.minimum(self.K, total_pieces)+1
#         return [(i,) for i in range(1, max_pieces)]
