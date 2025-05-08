#Dewey Schoenfelder
import torch
import numpy as np
from QModel import QModel

class Agent:
    def __init__(self, epsilon=0.2, learning_rate=0.01, gamma=0.9, player_id=1):
        self.player_id = player_id
        self.qmodel = QModel()
        self.learning_rate = learning_rate
        self._optimizer = torch.optim.Adam(self.qmodel.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon

    def _get_valid_random_action(self, state2d):
        empty_cells = np.argwhere(state2d == 0)
        if len(empty_cells) == 0:
            return 0
        idx = np.random.choice(len(empty_cells))
        ax, ay = empty_cells[idx]
        return ax * 3 + ay

    def get_random_action(self):
        return np.random.randint(0, 9)

    def get_Q_action(self, state):
        state2d, turn = state
        turns = torch.tensor([turn], dtype=torch.int64)
        states2d = torch.tensor([state2d], dtype=torch.int64)
        mask = (state2d == 0).astype(np.float32).flatten()
        mask2 = (state2d != 0).astype(np.float32).flatten() * -1000

        with torch.no_grad():
            qvalues = self.qmodel(states2d, turns)[0]
            masked_qactions = qvalues * torch.tensor(mask) + torch.tensor(mask2)
            action = torch.argmax(masked_qactions).item()
            ax, ay = divmod(action, 3)
            if state2d[ax, ay] != 0:
                action = self._get_valid_random_action(state2d)
            return action

    def get_epsilon_greedy_action(self, state):
        state2d, _ = state
        if np.random.rand() < self.epsilon:
            return self._get_valid_random_action(state2d)
        return self.get_Q_action(state)

    def do_Qlearning_on_agent_model(self, state_action_nstate_rewards):
        states, actions, next_states, rewards = zip(*state_action_nstate_rewards)
        states2d, turns = zip(*states)
        next_states2d, next_turns = zip(*next_states)

        turns = torch.tensor(turns, dtype=torch.int64)
        next_turns = torch.tensor(next_turns, dtype=torch.int64)
        states2d = torch.tensor(states2d, dtype=torch.int64)
        next_states2d = torch.tensor(next_states2d, dtype=torch.int64)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        with torch.no_grad():
            mask = (next_turns > 0).float()
            next_qvalues = self.qmodel(next_states2d, next_turns)
            expected_q = rewards + self.gamma * torch.max(next_qvalues, dim=1)[0] * mask

        current_q = self.qmodel(states2d, turns)
        q_for_actions = current_q.gather(1, actions.unsqueeze(1)).squeeze()

        loss = torch.nn.functional.smooth_l1_loss(q_for_actions, expected_q)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss.item()