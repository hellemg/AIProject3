import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Board:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        # Array holding edge-representation of grid
        self.cell_neighbours = np.empty(grid_size*grid_size, dtype=list)
        # self.cell_neighbours = np.empty((grid_size, grid_size), dtype=list)
        for i in range(grid_size*grid_size):
            # for j in range(grid_size):
            self.cell_neighbours[i] = self.get_neighbour_list(i)

    def flip_board(self, board):
        new_board = self.get_initial_state()
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                ind = r*self.grid_size + c
                inverse_ind = c*self.grid_size + r
                new_board[inverse_ind] = board[ind]*(-1)
        return new_board

    def get_initial_state(self):
        # Values of each cell will be 0.
        return np.zeros(self.grid_size*self.grid_size, dtype=int)

    def get_possible_actions_from_state(self, grid):
        """
        Finds all possible actions from a given state (action = empty boardcell)
        grid: ndarray, grid in some state

        :returns: ndarray of ints, indices where a piece can be put
        """
        return np.where(grid == 0)[0]

    def get_state_from_state_action(self, grid, ind, player, verbose):
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

    def get_neighbour_list(self, ind):
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
        print('-------------------')
        for i in range(self.grid_size):
            print(grid[i*self.grid_size:i*self.grid_size+self.grid_size])

    def display_board_graph(self, grid):
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
        labels = False
        if self.grid_size >= 9:
            node_size = 500
        elif self.grid_size >= 7:
            node_size = 1000
        elif self.grid_size >= 5:
            node_size = 1500
        else:
            node_size = 3000
            labels = True
        nx.draw(G, pos, node_color=node_colors, node_size=node_size, with_labels=labels)