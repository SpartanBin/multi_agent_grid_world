from typing import Optional, NamedTuple

import numpy as np
import torch


class RolloutBufferSamples(NamedTuple):
    state_features: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class RolloutBuffer:
    """
    """

    def __init__(
        self,
        agents_num: int,
        buffer_size: int,
        state_features_dim: int,
        height: int,
        width: int,
        gae_lambda: float = 1,
        gamma: float = 0.99,
    ):
        '''
        :param agents_num:
        :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
        :param gamma: Discount factor
        '''
        self.agents_num = agents_num
        self.buffer_size = buffer_size
        self.state_features_dim = state_features_dim
        self.height = height
        self.width = width
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.pos = 0
        self.full = False
        self.loc, self.weight, self.actions, self.rewards, self.advantages = None, None, None, None, None
        self.returns, self.dones, self.values, self.log_probs = None, None, None, None
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.state_features = torch.zeros(
            (self.buffer_size, self.agents_num, self.state_features_dim, self.height, self.width), dtype=torch.float32)
        self.actions = torch.zeros((self.buffer_size, self.agents_num), dtype=torch.int32)
        self.rewards = torch.zeros((self.buffer_size, self.agents_num), dtype=torch.float32)
        self.returns = torch.zeros((self.buffer_size, self.agents_num), dtype=torch.float32)
        self.dones = torch.zeros((self.buffer_size, self.agents_num), dtype=torch.float32)
        self.values = torch.zeros((self.buffer_size, self.agents_num), dtype=torch.float32)
        self.log_probs = torch.zeros((self.buffer_size, self.agents_num), dtype=torch.float32)
        self.advantages = torch.zeros((self.buffer_size, self.agents_num), dtype=torch.float32)
        self.generator_ready = False
        self.pos = 0
        self.full = False

    def add(
        self, state_features: torch.Tensor, actions: torch.Tensor, reward: torch.Tensor,
        done: torch.Tensor, values: torch.Tensor, log_probs: torch.Tensor,
    ) -> None:
        """

        :param state_features:
        :param actions: Action
        :param reward:
        :param values: estimated value of the current state
            following the current policy.
        :param log_probs: log probability of the action
            following the current policy.
        """

        self.state_features[self.pos] = state_features.clone()
        self.actions[self.pos] = actions.clone()
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = values.clone()
        self.log_probs[self.pos] = log_probs.clone()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantage(self, last_values: torch.Tensor, done: torch.Tensor) -> None:
        """
        Post-processing step: compute the returns (sum of discounted rewards)
        and GAE advantage.
        Adapted from Stable-Baselines PPO2.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
        where R is the discounted reward with value bootstrap,
        set ``gae_lambda=1.0`` during initialization.

        :param last_values: shape = (1, self.vehicle_num)
        :param done:

        """

        last_values = last_values.clone()

        last_gae_lam = torch.zeros(self.agents_num, dtype=torch.float32)
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - done
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    def prepare_for_iteration(self):

        index0 = []
        index1 = []
        for i in range(self.agents_num):
            notdone_index = torch.where(self.dones[:, i] == 0)[0]
            if len(notdone_index) > 0:
                use_index = notdone_index.clone().tolist()
                first_notdone_index = notdone_index[0]
                if first_notdone_index > 0:
                    use_index.append((first_notdone_index - 1).item())
                for index_ in range(len(notdone_index)):
                    if index_ > 0:
                        if notdone_index[index_] - notdone_index[index_ - 1] > 1:
                            use_index.append((notdone_index[index_] - 1).item())
                index0 += use_index
                index1 += [i] * len(use_index)
        index = (index0, index1)

        self.state_features = self.state_features[index]
        self.actions = self.actions[index]
        self.values = self.values[index]
        self.log_probs = self.log_probs[index]
        self.advantages = self.advantages[index]
        self.returns = self.returns[index]

    def copy_or_not(self, tensor: torch.Tensor, copy: bool = True) -> torch.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param tensor:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return:
        """
        if copy:
            return tensor.clone()
        return tensor

    def _get_samples(self, batch_inds: np.ndarray):
        data = (
            self.state_features[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds],
            self.log_probs[batch_inds],
            self.advantages[batch_inds],
            self.returns[batch_inds],
        )
        return RolloutBufferSamples(*tuple(map(self.copy_or_not, data)))

    def get(self, batch_size: Optional[int] = None):

        assert self.full, 'Must fill the container if you want to sample from container'
        buffer_size = len(self.actions)
        indices = np.random.permutation(buffer_size)

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = buffer_size

        start_idx = 0
        while start_idx < buffer_size:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size