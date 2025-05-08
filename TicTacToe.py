#Dewey Schoenfelder
import numpy as np
import torch

class TicTacToe:
    # Environment
    def __init__(self, player1, player2):
        self.players = {1: player1, 2: player2}
        # reward for each outcome of the game (tie, player1 wins, player2 wins)
        self._reward = {0: 0.33, 1: 1, 2: -3}

    def play(self, num_games=1, visualize=False):
        """ Experience store - play several full games """
        draw_count = 0
        won_count = 0
        lost_count = 0
        total_count = 0
        transitions_1 = []
        transitions_2 = []

        for _ in range(num_games):
            turn = 1
            state2d = np.zeros((3, 3), dtype=np.int64)
            state = (state2d, turn)
            current_player = self.players[turn]
            action1 = current_player.get_random_action()
            next_state, reward = self.play_turn(state, action1)

            if visualize:
                self.visualize_state(next_state, turn)

            for i in range(9):
                current_player = self.players[turn]
                (state2d, turn) = next_state
                action2 = current_player.get_epsilon_greedy_action(next_state)
                next_state2, reward2 = self.play_turn(next_state, action2)

                if visualize:
                    self.visualize_state(next_state2, turn)

                if reward2 == -3:
                    lost_count += 1
                    transitions_1.append((state, action1, next_state2, reward2))
                    transitions_2.append((next_state, action2, next_state2, -1 * reward2))
                else:
                    transitions_1.append((state, action1, next_state2, reward2))

                (state2d, turn) = next_state2

                if turn == 0:
                    if visualize:
                        print('----------game ended2-------------')
                    break

                current_player = self.players[turn]
                action3 = current_player.get_epsilon_greedy_action(next_state2)
                next_state3, reward3 = self.play_turn(next_state2, action3)

                if visualize:
                    self.visualize_state(next_state3, turn)

                if reward3 == 1:
                    won_count += 1
                    transitions_1.append((next_state2, action3, next_state3, reward3))
                    transitions_2.append((next_state, action2, next_state3, -1 * reward3))
                else:
                    transitions_2.append((next_state, action2, next_state3, reward2))

                (state2d, turn) = next_state3
                if turn == 0:
                    if reward3 == 0.33:
                        draw_count += 1
                    if visualize:
                        print('----------game ended-------------')
                    break

                next_state = next_state3
                reward = reward3
                state = next_state2
                action1 = action3

                if turn == 0:
                    if visualize:
                        print('----------game ended-------------')
                    break

        return transitions_1, transitions_2, won_count, draw_count, lost_count, num_games

    def play_turn(self, state, action):
        state2d, turn = state
        next_state2d = state2d.copy()
        next_turn = turn % 2 + 1
        ax, ay = torch.div(action, 3, rounding_mode='trunc'), action % 3
        next_state2d[ax, ay] = turn

        mask = next_state2d == turn
        if (
            (mask[0, 0] and mask[1, 1] and mask[2, 2]) or
            (mask[0, 2] and mask[1, 1] and mask[2, 0]) or
            (mask[0, 0] and mask[0, 1] and mask[0, 2]) or
            (mask[1, 0] and mask[1, 1] and mask[1, 2]) or
            (mask[2, 0] and mask[2, 1] and mask[2, 2]) or
            (mask[0, 0] and mask[1, 0] and mask[2, 0]) or
            (mask[0, 1] and mask[1, 1] and mask[2, 1]) or
            (mask[0, 2] and mask[1, 2] and mask[2, 2])
        ):
            next_state = (next_state2d, 0)
            return next_state, self._reward[turn]

        if (next_state2d != 0).all():
            next_state = (next_state2d, 0)
            return next_state, self._reward[0]

        next_state = (next_state2d, next_turn)
        return next_state, self._reward[0]

    @staticmethod
    def visualize_state(next_state, turn):
        next_state2d, next_turn = next_state
        print(f"player {turn}'s turn:")
        if (next_state2d == 0).all() and turn == 0:
            print("[invalid state]\n\n")
        else:
            print(
                str(next_state2d)
                .replace("[[", "")
                .replace(" [", "")
                .replace("]]", "")
                .replace("]", "")
                .replace("0", ".")
                .replace("1", "O")
                .replace("2", "X")
                + "\n\n"
            )
