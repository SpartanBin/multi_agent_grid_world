'''
main program for MAPPO
'''

import sys
import os

import numpy as np

project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_path)

from MAPPO.multi_agent_PPO_algorithm import multi_agent_PPO
from environment import large_agents_planning_env


def test_dijkstra_path_on_large_env(test_times):
    '''
    Test using dijkstra path to assign targets and planning path to the target for agents only.
    :param test_times: testing times
    :return:
    '''
    large_env = large_agents_planning_env()
    mean_over_steps = []
    max_over_steps = []
    for timestep in range(test_times):
        large_env.reset()
        agents_move_steps = large_env.assigning_targets_and_planning()
        print('{} has been completed! '.format(timestep))
        mean_over_steps.append(np.mean(agents_move_steps))
        max_over_steps.append(np.max(agents_move_steps))
    return mean_over_steps, max_over_steps


def test_dijkstra_path_with_marl_on_large_env(test_times, marl):
    '''
    Test using dijkstra path with MARL to assign targets and planning path to the target for agents.
    :param test_times: testing times
    :param marl: instantiated class multi_agent_PPO
    :return:
    '''
    large_env = large_agents_planning_env()
    mean_over_steps = []
    max_over_steps = []
    for timestep in range(test_times):
        large_env.reset()
        envs, agents_move_steps = large_env.assigning_targets_and_planning_for_next_control()
        envs_over_time = marl.planning_on_large_env(envs)
        for agents_id in envs_over_time.keys():
            agents_move_steps[agents_id] = envs_over_time[agents_id]
        print('{} has been completed! '.format(timestep))
        mean_over_steps.append(np.mean(agents_move_steps))
        max_over_steps.append(np.max(agents_move_steps))
    return mean_over_steps, max_over_steps


if __name__ == '__main__':

    ortho_init = True
    learning_rate = 1e-4
    buffer_size = 2048
    batch_size = 256
    n_epochs = 10
    gamma = 0.99  # In OpenAI Five, when set this to 0.99, policy performs best
    gae_lambda = 0.95
    clip_range = 0.2
    clip_range_vf = None
    ent_coef = 0.0  # In OpenAI Five, when set this to 0.01, policy performs best
    vf_coef = 0.5
    max_grad_norm = 0.5
    target_kl = None
    device = 'cpu'

    # # for 30 * 30
    # obstacles_index = (
    #     [4, 8, 14, 21, 25, 4, 8, 15, 21, 25],
    #     [4, 8, 12, 8, 4, 25, 21, 17, 21, 25],
    # )

    # for 10 * 10
    obstacles_index = (
        [5],
        [5],
    )

    height = 10
    width = 10
    agents_num = 10
    reward_type = 'reverse_distance'  # 'default', 'reverse_distance', 'cooperation'
    file_name = 'env{}_agents{}_block{}_reward_type_{}_planning'.format(
        height, agents_num, len(obstacles_index[0]), reward_type)

    # init MARL class
    marl = multi_agent_PPO(
        obstacles_index=obstacles_index,
        height=height,
        width=width,
        agents_num=agents_num,
        reward_type=reward_type,
        file_name=file_name,
        ortho_init=ortho_init,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        clip_range_vf=clip_range_vf,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        target_kl=target_kl,
        seed=400,
        device=device,
    )

    # # training MARL
    # training_times = 200
    # marl.learn(
    #     training_times=training_times,
    #     test_episode_times=100,
    # )

    # load MARL params
    file_path = project_path + '/results/env10_agents10_block1_reward_type_reverse_distance_planning.pickle'
    marl.load_params(file_path=file_path)

    import pickle

    # test the performance of marl on env
    all_states = marl.test(10)
    with open('all_states.pickle', 'wb') as file:
        pickle.dump(all_states, file)

    # # test the performance of dijkstra path with MARL on large env
    # mean_over_steps, max_over_steps = test_dijkstra_path_with_marl_on_large_env(1000, marl)
    # with open('30_30_dijkstra_path_with_marl_finished_time.pickle', 'wb') as file:
    #     pickle.dump((mean_over_steps, max_over_steps), file)

    # # test the performance of dijkstra path on large env
    # mean_over_steps, max_over_steps = test_dijkstra_path_on_large_env(1000)
    # with open('30_30_dijkstra_path_finished_time.pickle', 'wb') as file:
    #     pickle.dump((mean_over_steps, max_over_steps), file)