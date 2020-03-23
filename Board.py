import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Board:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        # Array holding edge-representation of grid
        self.cell_neighbours = np.empty((grid_size, grid_size), dtype=list)
        for i in range(grid_size):
            for j in range(grid_size):
                self.cell_neighbours[i, j] = self.get_neighbour_list(i, j)

    def get_initial_state(self):
        # Values of each cell will be tuples
        grid = np.empty((self.grid_size, self.grid_size), dtype=tuple)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                grid[i, j] = (0, 0)
        return grid

    def get_neighbour_list(self, row, column):
        neighbour_list = [None, None, None, None, None, None]
        # List of coordinates for the neighbours of cell [row, column]
        if (row > 0):
            neighbour_list[0] = (row-1, column)
            if (column < self.grid_size - 1):
                neighbour_list[1] = (row-1, column+1)
        if (column > 0):
            neighbour_list[3] = (row, column-1)
        if (column < self.grid_size - 1):
            neighbour_list[2] = (row, column+1)
        if (row < self.grid_size - 1):
            neighbour_list[5] = (row+1, column)
            if (column > 0):
                neighbour_list[4] = (row+1, column-1)
        return neighbour_list

    def get_grid_size(self):
        return self.grid_size

    def display_board(self, grid):
        print('-------------------')
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                print(grid[i][j], end='')
            print('\n')

    def get_possible_actions_from_state(self, grid):
        """
        Finds all possible actions from a given state (empty boardcells)
        grid: ndarray, grid in some state

        :returns: list of tuples (row, column)
        """
        possible_actions = []
        rows, columns = grid.shape
        for r in range(rows):
            for c in range(columns):
                if grid[r][c] == (0, 0):
                    possible_actions.append((r, c))
        return possible_actions

    def get_state_from_state_action(self, grid, row, column, player, verbose):
        """
        grid: ndarray, grid in some state
        row: int, row to place piece on
        column: int, column to place piece on
        player: tuple, (1,0) for p1, (0,1) for p2. Also is the new value
        """
        print('welcome to state from sa')
        temp_grid = grid.copy()
        # Fill grid-cell with appropriate tuple
        temp_grid[row][column] = player
        if verbose:
            print('Player {} places piece on ({},{})'.format(
                player[1]+1, row, column))
        return temp_grid

    def check_game_done(self, grid, player):
        """
        grid: ndarray, grid in some state
        player: tuple, (1,0) for p1, (0,1) for p2
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
        # Start implementing for p1 to see how it goes
        visited_cells = []
        to_visit = []
        for c in range(self.grid_size):
            # Check for boardcells with (1,0)-value to initiate to_visit
            if grid[0][c] == player:
                # Should visit the boardcell
                to_visit.append((0, c))
        print(to_visit)
        # Go through all to_visit-coordinates until none left
        while to_visit:
            current_coords = to_visit.pop()
            print('current coords:', current_coords)
            # Check if current_cell is on the other side (SW)
            if current_coords[0] == self.grid_size-1:
                return True

            neighbours = self.cell_neighbours[current_coords]
            print('neighbours:', neighbours)
            for n_coords in neighbours:
                input()
                # Dont go outside grid, dont go back to earlier visited cells
                if (n_coords is not None) and (n_coords not in visited_cells):
                    # Piece continuing the trail, add to to_visit
                    print('...checking ', n_coords)
                    if grid[n_coords] == player:
                        print('...values is {}, adding {} to to_visit'.format(grid[n_coords], n_coords))
                        to_visit.append(n_coords)
            # Make sure not to go back to current cell
            print('...adding current coords to visited cells', current_coords)
            visited_cells.append(current_coords)
        # Did not find a path
        return False

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
                board_cell = grid[r, c]
                G.add_node((r, c), pos=pos)
                if board_cell == (0, 0):
                    node_colors.append('black')
                elif board_cell == (1, 0):
                    # Player 1
                    node_colors.append('lightskyblue')
                elif board_cell == (0, 1):
                    # Player 2
                    node_colors.append('magenta')
                else:
                    ValueError('Unknown value in boardcell ({},{}): {}'.format(
                        x, y, board_cell))
                x += 1
                y -= 1

        # Add edges between neighbours
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for n in self.cell_neighbours[i, j]:
                    if n is not None:
                        # Add edge between current node and neighbour node
                        G.add_edge((i, j), (n[0], n[1]))

        # Plot
        pos = nx.get_node_attributes(G, 'pos')

        # TODO: Fix node size so it works with both grid_size = 3 and grid_size = 10

        nx.draw(G, pos, node_color=node_colors, node_size=3000,
                with_labels=True, font_weight='bold')

        # TODO: Comment out plt.show() when animating

        plt.show()
