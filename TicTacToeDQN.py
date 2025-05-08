#Dewey Schoenfelder
import sys
import os
import numpy as np
import torch
from TicTacToe import TicTacToe
from Agent import Agent
import tqdm

def main():
    np.random.seed(3)
    torch.manual_seed(1)

    player1 = Agent(epsilon=0.0, learning_rate=0.003, player_id=1)
    player2 = Agent(epsilon=0.0, learning_rate=0.003, player_id=2)

    USE_TRAINED_MODELS = True

    if USE_TRAINED_MODELS:
        checkpoint_data1 = torch.load('checkpoint/tictactoe_player1_model.pt')
        player1.qmodel.load_state_dict(checkpoint_data1['state_dict'])
        player1._optimizer.load_state_dict(checkpoint_data1['optimizer'])

        checkpoint_data2 = torch.load('checkpoint/tictactoe_player2_model.pt')
        player2.qmodel.load_state_dict(checkpoint_data2['state_dict'])
        player2._optimizer.load_state_dict(checkpoint_data2['optimizer'])

        player1.epsilon = 0.0
        player2.epsilon = 0.0

        game = TicTacToe(player1, player2)
        _, _, won_count, draw_count, lost_count, total_count = game.play(num_games=100, visualize=True)
        print("percentage won =", won_count, "percentage lost=", lost_count, "percentage draw=", draw_count, "total count=", total_count)
        print("------------")
        return

    total_number_of_games = 1000000
    number_of_games_per_batch = 200

    game = TicTacToe(player1, player2)
    min_loss = np.inf
    range_ = tqdm.trange(total_number_of_games // number_of_games_per_batch)

    for i in range_:
        transitions1, transitions2, _, _, _, _ = game.play(num_games=number_of_games_per_batch)
        np.random.shuffle(transitions1)
        np.random.shuffle(transitions2)

        loss = player1.do_Qlearning_on_agent_model(transitions1)
        loss1 = player2.do_Qlearning_on_agent_model(transitions2)

        if loss < min_loss and loss < 0.01:
            min_loss = loss

        range_.set_postfix(loss=loss, min_loss=min_loss)

        print("----------saving models-----------------")
        checkpoint_data1 = {
            'state_dict': player1.qmodel.state_dict(),
            'optimizer': player1._optimizer.state_dict()
        }
        ckpt_path1 = os.path.join("checkpoint/tictactoe_player1_model.pt")
        torch.save(checkpoint_data1, ckpt_path1)

        checkpoint_data2 = {
            'state_dict': player2.qmodel.state_dict(),
            'optimizer': player2._optimizer.state_dict()
        }
        ckpt_path2 = os.path.join("checkpoint/tictactoe_player2_model.pt")
        torch.save(checkpoint_data2, ckpt_path2)

if __name__ == "__main__":
    sys.exit(int(main() or 0))
