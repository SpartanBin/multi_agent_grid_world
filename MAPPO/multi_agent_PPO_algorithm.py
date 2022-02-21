import sys
import os
from typing import Optional, Union
from copy import deepcopy
import random
import pickle

import numpy as np
import torch
from torch.nn import functional as F

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)

from environment import rl_env
from MAPPO import buffers, policies


class multi_agent_PPO:

    def __init__(
        self,
        obstacles_index,
        height: int,
        width: int,
        agents_num: int,
        reward_type: str,
        file_name: str,
        ortho_init: bool,
        learning_rate: Union[float, int] = 3e-4,
        buffer_size: int = 2048,
        batch_size: Optional[int] = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        seed: Union[int, None] = 400,
        device: Union[torch.device, str] = "cpu",
    ):
        '''

        :param obstacles_index:
        :param height:
        :param width:
        :param agents_num:
        :param reward_type:
        :param file_name:
        :param ortho_init:
        :param learning_rate:
        :param buffer_size:
        :param batch_size:
        :param n_epochs:
        :param gamma:
        :param gae_lambda:
        :param clip_range:
        :param clip_range_vf:
        :param ent_coef:
        :param vf_coef:
        :param max_grad_norm:
        :param target_kl:
        :param seed:
        :param device:
        :return:
        '''

        if seed is not None:
            random.seed(seed)
            # Seed numpy RNG
            np.random.seed(seed)
            # seed the RNG for all devices (both CPU and CUDA)
            torch.manual_seed(seed)
        if device == 'cuda':
            # Deterministic operations for CuDNN, it may impact performances
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.num_timesteps = 0
        self.episode = 0
        self.all_past_time = []
        self.past_time_100 = []
        self.mean_100_past_time = 200
        self.old_mean_100_past_time = float('inf')
        self.need_saving_params = True

        self.env = rl_env(
            obstacles_index=obstacles_index,
            height=height,
            width=width,
            agents_num=agents_num,
            reward_type=reward_type,
            seed=seed,
        )

        self.device = torch.device(device)

        self.init_learning_rate = learning_rate

        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.ortho_init = ortho_init
        self.action_dim = 5

        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl
        self.file_name = file_name

        self.rollout_buffer = buffers.RolloutBuffer(
            agents_num=self.env.agents_num,
            buffer_size=self.buffer_size,
            state_features_dim=4,
            height=self.env.height,
            width=self.env.width,
            gae_lambda=self.gae_lambda,
            gamma=self.gamma,
        )
        self.policy = policies.multi_agent_ACP(
            ortho_init=self.ortho_init,
            in_channels=4,
            output_dim=32,
            action_dim=self.action_dim,
            learning_rate=self.init_learning_rate,
        ).to(torch.float32).to(self.device).eval()

        self.old_params = self.policy.to('cpu').state_dict()
        self.policy = self.policy.to(self.device)

    def feature_engeering(self, agents_num, agents_positions_, target_positions, obstacles, device):

        working_id = np.where(agents_positions_[:, 0] != -1)[0]
        self.working_id = working_id

        # agents positions
        agents_positions_ = tuple(np.array(agents_positions_[working_id], dtype=int).T.tolist())
        all_agents_positions = np.zeros((self.env.height, self.env.width), dtype=np.float32)
        all_agents_positions[agents_positions_] = 1
        idiv_agents_positions = np.zeros((agents_num, self.env.height, self.env.width), dtype=np.float32)
        agents_positions_ = tuple([list(working_id)] + list(agents_positions_))
        idiv_agents_positions[agents_positions_] = 1

        # reshape and concat
        idiv_agents_positions = idiv_agents_positions[:, np.newaxis, :, :]
        all_agents_positions = all_agents_positions[np.newaxis, np.newaxis, :, :]
        all_agents_positions = np.repeat(all_agents_positions, repeats=agents_num, axis=0)
        target_positions = target_positions[np.newaxis, np.newaxis, :, :]
        target_positions = np.repeat(target_positions, repeats=agents_num, axis=0)
        obstacles = obstacles[np.newaxis, np.newaxis, :, :]
        obstacles = np.repeat(obstacles, repeats=agents_num, axis=0)
        state_features = np.concatenate(
            (idiv_agents_positions, all_agents_positions, target_positions, obstacles), axis=1)

        state_features = torch.from_numpy(state_features).to(torch.float32).to(device)

        return state_features

    def make_one_step_forward_for_env_by_ac_probs(self, env, distributions):

        ac_probs = distributions.all_probs()

        actions, new_obs, reward, done = env.step_by_action_probs(ac_probs=ac_probs)

        actions = torch.tensor(actions).view((1, -1)).to(self.device)

        log_probs = distributions.log_prob(actions)

        return actions, log_probs, new_obs, reward, done

    def collect_rollouts(self):
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        need_test = False
        timestep = 0
        self.rollout_buffer.reset()
        self.policy.eval()

        while timestep < self.buffer_size:

            with torch.no_grad():
                # Convert to pytorch tensor
                agents_positions, target_positions, obstacles = self._last_obs
                state_features = self.feature_engeering(
                    agents_num=self.env.agents_num,
                    agents_positions_=agents_positions,
                    target_positions=target_positions,
                    obstacles=obstacles,
                    device=self.device,
                )

                distributions, values, _ = self.policy.forward(
                    state_features=state_features,
                    actions=None,
                )
            state_features = state_features.cpu()
            values = values.flatten().cpu()

            actions, log_probs, new_obs, reward, done = self.make_one_step_forward_for_env_by_ac_probs(
                env=self.env,
                distributions=distributions,
            )
            actions = actions[0].cpu()
            log_probs = log_probs[0].cpu()
            reward = torch.from_numpy(reward).to(torch.float32)
            done = torch.from_numpy(done).to(torch.float32)

            timestep += 1
            self.num_timesteps += 1

            self.rollout_buffer.add(
                state_features=state_features,
                actions=actions,
                reward=reward,
                done=self._last_done,
                values=values,
                log_probs=log_probs,
            )

            if self.env.done or self.env.past_time >= 4000:
                self.episode += 1
                self.all_past_time.append(self.env.past_time)
                self.past_time_100.append(self.env.past_time)
                new_obs = self.env.reset()

                if len(self.past_time_100) > 100:
                    self.past_time_100.pop(0)
                    self.mean_100_past_time = np.mean(self.past_time_100)
                    print(self.mean_100_past_time)
                    if self.mean_100_past_time < self.old_mean_100_past_time:
                        self.old_mean_100_past_time = self.mean_100_past_time
                        self.need_saving_params = True

            self._last_obs = new_obs
            self._last_done = done

        with torch.no_grad():
            # Compute value for the last timestep
            agents_positions, target_positions, obstacles = self._last_obs
            state_features = self.feature_engeering(
                agents_num=self.env.agents_num,
                agents_positions_=agents_positions,
                target_positions=target_positions,
                obstacles=obstacles,
                device=self.device,
            )
            _, values, _ = self.policy.forward(
                state_features=state_features,
                actions=None,
            )
        values = values.flatten().cpu()
        self.rollout_buffer.compute_returns_and_advantage(last_values=values, done=done)
        self.rollout_buffer.prepare_for_iteration()

        return need_test

    def cal_learning_rate(self):
        learning_rate = self.init_learning_rate * self.mean_100_past_time / 50
        return learning_rate

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        self.policy.train()
        # Update optimizer learning rate
        for param_group in self.policy.ACP.optimizer.param_groups:
            param_group["lr"] = self.cal_learning_rate()

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):

                values, log_prob, entropy = self.policy.forward(
                    state_features=rollout_data.state_features.to(self.device),
                    actions=rollout_data.actions.to(self.device),
                )

                # Normalize advantage
                advantages = rollout_data.advantages.to(self.device)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # flatten data
                values = torch.flatten(values)
                log_prob = torch.flatten(log_prob)
                entropy = torch.flatten(entropy)
                old_log_prob = torch.flatten(torch.as_tensor(
                    rollout_data.old_log_prob, dtype=torch.float32, device=self.device))
                advantages = torch.flatten(torch.as_tensor(
                    advantages, dtype=torch.float32, device=self.device))
                old_values = rollout_data.old_values.to(self.device)
                returns = torch.flatten(torch.as_tensor(
                    rollout_data.returns, dtype=torch.float32, device=self.device))

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = - torch.min(policy_loss_1, policy_loss_2).mean()

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    old_values = torch.flatten(torch.as_tensor(old_values, dtype=torch.float32, device=self.device))
                    values_pred = old_values + torch.clamp(
                        values - old_values, - self.clip_range_vf, self.clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(returns, values_pred)

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = - torch.mean(- log_prob)
                else:
                    entropy_loss = - torch.mean(entropy)

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Optimization step
                self.policy.optimize(
                    loss=loss,
                    max_grad_norm=self.max_grad_norm,
                )
                approx_kl_divs.append(torch.mean(old_log_prob - log_prob).detach().cpu().numpy().copy())

            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                print(f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}")
                break

    def test(self, test_episode_times: int):
        self.policy.eval()
        env = deepcopy(self.env)
        new_obs = env.reset()
        past_time_list = []
        all_states = []
        for _ in range(test_episode_times):
            states = []
            while not env.done and env.past_time <= 2000:
                with torch.no_grad():
                    states.append(new_obs)
                    agents_positions, target_positions, obstacles = new_obs
                    state_features = self.feature_engeering(
                        agents_num=env.agents_num,
                        agents_positions_=agents_positions,
                        target_positions=target_positions,
                        obstacles=obstacles,
                        device=self.device,
                    )
                    distributions, _, _ = self.policy.forward(
                        state_features=state_features,
                    )
                _, _, new_obs, _, _ = self.make_one_step_forward_for_env_by_ac_probs(
                    env=env,
                    distributions=distributions,
                )
            # print(env.past_time)
            states.append(new_obs)
            all_states.append(states)
            past_time_list.append(env.past_time)
            new_obs = env.reset()
        print(np.mean(past_time_list))
        return all_states

    def planning_on_large_env(self, envs):
        self.policy.eval()
        envs_over_time = {}
        for key in envs.keys():
            env = envs[key]
            new_obs = env.agents_positions.copy(), env.target_positions.copy(), env.obstacles.copy()
            while not env.done:
                with torch.no_grad():
                    agents_positions, target_positions, obstacles = new_obs
                    state_features = self.feature_engeering(
                        agents_num=env.agents_num,
                        agents_positions_=agents_positions,
                        target_positions=target_positions,
                        obstacles=obstacles,
                        device=self.device,
                    )
                    distributions, _, _ = self.policy.forward(
                        state_features=state_features,
                    )
                _, _, new_obs, _, _ = self.make_one_step_forward_for_env_by_ac_probs(
                    env=env,
                    distributions=distributions,
                )
            agents_move_steps = env.agents_move_steps
            agents_ids = env.agents_ids
            for i, agents_id in enumerate(agents_ids):
                envs_over_time[agents_id] = agents_move_steps[i]
        return envs_over_time

    def save(self):
        if self.need_saving_params:
            params = self.policy.to('cpu').state_dict()
            self.old_params = params
            self.need_saving_params = False
        else:
            params = self.old_params
        with open(project_path + '/{}.pickle'.format(self.file_name), 'wb') as file:
            pickle.dump((params, self.all_past_time), file)
        self.policy = self.policy.to(self.device)

    def learn(self, training_times: int, test_episode_times: int,):

        self.num_timesteps = 0
        self.episode = 0
        self.all_past_time = []
        self.past_time_100 = []
        self.mean_100_past_time = 200
        self.old_mean_100_past_time = float('inf')
        self.need_saving_params = True

        self.old_params = self.policy.to('cpu').state_dict()
        self.policy = self.policy.to(self.device)

        self._last_obs = self.env.reset()
        self._last_done = torch.tensor([False] * self.env.agents_num, dtype=torch.float32)

        train_session = 0
        test_session = 0
        self.best_train_session = train_session
        while train_session < training_times:

            # if train_session % 1 == 0:
            #     # ------------------------------------------test--------------------------------------------------
            #     test_session += 1
            #     self.test(test_episode_times=test_episode_times)
            #
            #     # print('''
            #     # **------------------------------------------------------------------------------------------**
            #     # **------------------------------------------------------------------------------------------**
            #     # {}th test: seed is {},
            #     # **------------------------------------------------------------------------------------------**
            #     # **------------------------------------------------------------------------------------------**
            #     # '''.format(test_session, self.seed))
            #     # self.save()
            #     # ------------------------------------------------------------------------------------------------

            need_test = self.collect_rollouts()
            self.save()
            self.train()
            train_session += 1
            print('training successful in {}th training session; working_id: {}! '.format(
                train_session, self.working_id))

        self.save()

    def load_params(self, file_path: str = project_path + '/MAPPO_params.pickle'):
        with open(file_path, 'rb') as file:
            params = pickle.load(file)[0]
        self.policy = self.policy.to('cpu')
        self.policy.load_state_dict(params)
        self.policy = self.policy.to(self.device)