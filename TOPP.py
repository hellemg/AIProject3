import numpy as np
import matplotlib.pyplot as plt

from Environment import Environment
from GlobalConstants import P, p1, p2, num_games, visualize, grid_size


class TOPP:
    def __init__(self, players: []):
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
        # Initiate dictionary, no one has won anything yet
        for nn in players:
            self.scores[nn.anet._name] = 0

    def play_one_game(self, starting_player: int, player_one, player_two):
        """
        starting player: 1 or -1
        player_one: NeuralNet
        player_two: NeuralNet

        :returns: int, 1 or -1: winner of the game
        """
        env = Environment(grid_size)
        state = env.generate_initial_state()
        states_in_game = []
        current_player = starting_player
        while not env.check_game_done(state):
            # Get possible actions
            possible_actions = env.get_possible_actions_from_state(
                state)
            # Find best action for current player
            best_action = {1: player_one.default_policy(possible_actions, state, current_player),
                           -1: player_two.default_policy(possible_actions, state, current_player)}[current_player]
            # Do the action, get next state
            state = env.generate_child_state_from_action(
                state, best_action, current_player, True)
            states_in_game.append(state)
            # Next players turn
            current_player ^= (p1 ^ p2)
        # Winner was the last player to make a move (one before player)
        winner = current_player ^ (p1 ^ p2)
        if visualize:
            env.visualize(states_in_game, 500)
        return winner

    def play_one_serie(self, player_one, player_two):
        """
        player_one: NeuralNet
        player_two: NeuralNet
        """
        # Player that has been trained to start always starts the series
        starting_player = P
        player_names = {1: player_one.anet._name, -1: player_two.anet._name}
        for i in range(num_games):
            print('*** PLAYOFF BETWEEN {} AS Player 1 AND {} AS Player -1'.format(player_names[1], player_names[-1]))
            # Play one game
            winner = self.play_one_game(
                starting_player, player_one, player_two)
            # Update score for the winner
            winner_name = player_names[winner]
            self.scores[winner_name] += 1
            print('...{} wins game '.format(winner_name, i+1))

            # Alternate between who is starting player in self.play_one_game(starting_player: int, player_one, player_two)
            starting_player ^= (p1 ^ p2)

    def tournament(self):
        num = 0
        for i in range(len(self.players)):
            for j in range(i+1, len(self.players)):
                self.play_one_serie(self.players[i], self.players[j])
                print('Scores after this series:\n{}'.format(self.scores))
                num += 1

        print('played {} series of {} games'.format(num, num_games))
        self.display_results()

    def display_results(self):
        plt.bar(self.scores.keys(), self.scores.values())
        plt.show()
