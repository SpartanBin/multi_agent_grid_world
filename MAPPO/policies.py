import sys
import os
from functools import partial
from typing import Tuple, Union

import numpy as np
from torch.distributions import Categorical

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)

from MAPPO.model import *


class Extractor(nn.Module):

    def __init__(self, in_channels: int):
        '''
        :param in_channels:
        :return:
        '''
        super(Extractor, self).__init__()

        self.policy_net = body_model(in_channels)
        self.value_net = body_model(in_channels)

    def forward(self, features: torch.Tensor):
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        policy_output = self.policy_net(features)
        value_output = self.value_net(features)
        return policy_output, value_output


class CategoricalDistribution():
    """
    Categorical distribution for discrete actions.

    :param action_dim: Number of discrete actions
    """

    def __init__(self, action_dim: int):
        super(CategoricalDistribution, self).__init__()
        self.distribution = None
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits of the Categorical distribution.
        You can then get probabilities using a softmax.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """
        action_logits = nn.Linear(latent_dim, self.action_dim)
        return action_logits

    def proba_distribution(self, action_logits: torch.Tensor) -> "CategoricalDistribution":
        self.distribution = Categorical(logits=action_logits)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions)

    def all_probs(self):
        return self.distribution.probs

    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy()

    def sample(self) -> torch.Tensor:
        return self.distribution.sample()

    def mode(self) -> torch.Tensor:
        return torch.argmax(self.distribution.probs, dim=1)

    def get_actions(self) -> torch.Tensor:
        """
        Return actions according to the probability distribution.
        :return:
        """
        return self.sample()

    def actions_from_params(self, action_logits: torch.Tensor) -> torch.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions()

    def log_prob_from_params(self, action_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class ActorCriticPolicy(nn.Module):

    def __init__(
        self,
        ortho_init: bool,
        in_channels: int,
        output_dim: int,
        action_dim: int,
        learning_rate: Union[int, float],
    ):
        '''
        :param ortho_init:
        param names, key values are allowed param values
        :param output_dim: type of item in iteration must be int object
        :param action_dim:
        :param learning_rate:
        :return:
        '''
        super(ActorCriticPolicy, self).__init__()

        self.ortho_init = ortho_init

        self.extractor = Extractor(
            in_channels=in_channels,
        )
        self.value_net = nn.Linear(output_dim, 1)
        # Action distribution
        self.action_distribution = CategoricalDistribution(action_dim=action_dim)
        self.action_net = self.action_distribution.proba_distribution_net(latent_dim=output_dim)

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, eps=1e-5)

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def output_action_distribution(self, latent_pi: torch.Tensor):
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        return self.action_distribution.proba_distribution(action_logits=mean_actions)

    def forward(
            self,
            input_features: torch.Tensor,
            actions: Union[torch.Tensor, None] = None):
        """
        Forward pass in all the networks (actor and critic)

        :param input_features:
        :param actions:
        :return: action, value and log probability of the action
        """
        latent_pi, latent_vf = self.extractor(input_features)
        distribution = self.output_action_distribution(latent_pi)
        if latent_vf is None:
            latent_vf = latent_pi
        value = self.value_net(latent_vf)
        if actions is None:
            return distribution, value, None
        else:
            log_prob = distribution.log_prob(actions)
            entropy = distribution.entropy()
            return value, log_prob, entropy


class multi_agent_ACP():

    def __init__(
            self,
            ortho_init: bool,
            in_channels: int,
            output_dim: int,
            action_dim: int,
            learning_rate: Union[int, float]):
        '''
        :param conv_params: with input layer params
        :param output_dim: without input dim
        :return:
        '''

        self.ACP = ActorCriticPolicy(
            ortho_init=ortho_init,
            in_channels=in_channels,
            output_dim=output_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
        )

    @staticmethod
    def cal_conv_output_shape(input_shape, kernel_size, padding, dilation, stride):
        if type(input_shape) == tuple:
            row = int((input_shape[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
            col = int((input_shape[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
            return (row, col)
        else:
            return int((input_shape + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

    def to(self, param):
        self.ACP.to(param)
        return self

    def train(self):
        self.ACP.train()
        return self

    def eval(self):
        self.ACP.eval()
        return self

    def state_dict(self):
        return self.ACP.state_dict()

    def load_state_dict(self, ACP_params):
        self.ACP.load_state_dict(ACP_params)
        return self

    def optimize(self, loss: torch.Tensor, max_grad_norm):

        self.ACP.optimizer.zero_grad()
        loss.backward()
        # # Clip grad norm
        # torch.nn.utils.clip_grad_norm_(self.ACP.parameters(), max_grad_norm)
        self.ACP.optimizer.step()

    def forward(
            self,
            state_features: torch.Tensor,
            actions: Union[torch.Tensor, None] = None):
        """
        Forward pass in all the networks (actor and critic)

        :param state_features:
        :return: action, value and log probability of the action
        """

        if actions is None:
            distributions, values, _ = self.ACP(state_features, None)
            return distributions, values, None
        else:
            values, log_probs, entropys = self.ACP(state_features, actions)
            return values, log_probs, entropys