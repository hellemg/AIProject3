import numpy as np
import matplotlib.pyplot as plt

from Environment import Environment
from GlobalConstants import P, p1, p2, num_games, visualize, grid_size

class TOPP:
    def __init__(self, players: [], policy: str):
        """
        :param players: list of neural_nets

        M different neural nets are to play against each other
        G games between each agent, giving M*(M-1)/2 series

        M agents:
        - Each agent is a neural net and has a name, indicating when it was trained

        S series:
        - Alternate between who starts each game, send corresponding int to game
        - Get winner from game, add +1 to winning player

        G games:
        - Each game gets the int for the starting player ( P1: 1., P2: -1)
        - Each game reports which player won

        """
        # Keep all players
        self.players = players
        self.scores = {}
        self.policy = policy
        # Initiate dictionary, no one has won anything yet
        for nn in players:
            self.scores[nn.anet._name] = 0

    def play_one_game(self, starting_player: int, player_one, player_two):
        """
        starting player: 1 or -1
        player_one: NeuralNet (P1)
        player_two: NeuralNet (P2, -1)

        :returns: int, 1 or -1: winner of the game
        """
        env = Environment(grid_size)
        state = env.generate_initial_state()
        states_in_game = []
        current_player = starting_player
        while not env.check_game_done(state):
            # Fix board
            my_player = current_player
            # If the other person starts and I am 1, then I have to invert and flip and become -1
            if starting_player != current_player and current_player == 1:
                state = env.flip_state(state)
                my_player = -1
            # If I start and I am -1, then I have to invert and flip the board and become 1
            if starting_player == current_player and current_player == -1:
                state = env.flip_state(state)
                my_player = 1
            # Get possible actions
            possible_actions = env.get_possible_actions_from_state(
                state)
            # Find best action for current player, but with state and my_player as features
            if self.policy == 'default':
                best_action = {1: player_one.default_policy(possible_actions, state, my_player),
                            -1: player_two.default_policy(possible_actions, state, my_player)}[current_player]
            elif self.policy == 'best':
                best_action = {1: player_one.best_action(possible_actions, state, my_player),
                            -1: player_two.best_action(possible_actions, state, my_player)}[current_player]
            else:
                raise ValueError('Unknown policy in TOPP: {}'.format(self.policy))
            # Do the action, get next state
            state = env.generate_child_state_from_action(
                state, best_action, my_player)
            # Flip the state back if necessary
            if my_player != current_player:
                state = env.flip_state(state)
            # Add the state to list for visualization
            states_in_game.append(state)
            # Next players turn
            current_player ^= (p1 ^ p2)
        # Winner was the last player to make a move (one before player)
        winner = current_player ^ (p1 ^ p2)
        if visualize:
            env.visualize(states_in_game, 500)
        return winner

    def play_one_serie(self, player_one, player_two, verbose):
        """
        player_one: NeuralNet, P1
        player_two: NeuralNet, P2 (-1)
        """
        # Player that has been trained to start always starts the series
        starting_player = P
        player_names = {1: player_one.anet._name, -1: player_two.anet._name}
        for i in range(num_games):
            if verbose:
                print('*** PLAYOFF BETWEEN {} AS Player 1 AND {} AS Player -1'.format(player_names[1], player_names[-1]))
            # Play one game
            winner = self.play_one_game(
                starting_player, player_one, player_two)
            # Update score for the winner
            winner_name = player_names[winner]
            self.scores[winner_name] += 1
            if verbose:
                print('...{} wins game '.format(winner_name, i+1))

            # Alternate between who is starting player in self.play_one_game(starting_player: int, player_one, player_two)
            starting_player ^= (p1 ^ p2)

    def tournament(self, verbose = False):
        """
        Goes through the list of players and ensures they play one series against each other
        P1 is the i-player, P2, is the j-player
        """
        num = 0
        for i in range(len(self.players)):
            for j in range(i+1, len(self.players)):
                self.play_one_serie(self.players[i], self.players[j], verbose)
                num += 1

        print('played {} series of {} games'.format(num, num_games))

    def display_results(self):
        print('Final scores:\n{}'.format(self.scores))
        plt.bar(self.scores.keys(), self.scores.values())
        plt.show()

    def several_tournaments(self, num_tournaments = 9):
        no_rows = np.ceil(np.sqrt(num_tournaments))
        no_cols = np.ceil(num_tournaments / no_rows)
        fig = plt.figure(figsize=(no_rows, no_cols))
        for t in range(num_tournaments):
            self.tournament()
            plt.subplot(no_rows, no_cols, t + 1)
            plt.bar(self.scores.keys(), self.scores.values())
            self.reset_scores()
        plt.show()

    def reset_scores(self):
        self.scores = {}
        # Initiate dictionary, no one has won anything yet
        for nn in self.players:
            self.scores[nn.anet._name] = 0