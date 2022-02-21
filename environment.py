'''
reinforcement learning simulation environment
'''

import random
from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
import torch
import networkx as nx


class rl_env:
    '''
    Use for training MARL
    '''

    def __init__(
        self,
        obstacles_index,
        height: int = 8,
        width: int = 8,
        agents_num: int = 4,
        reward_type: str = 'default',
        seed: Union[int, None] = 400,
    ):
        '''

        :param obstacles_index: fixed position of obstacles, format like ([5, 7], [6, 2])
        :param height: the number of row for env
        :param width: the number of column for env
        :param agents_num: the number of agent(agents) in this simulation environment
        :param reward_type: allowed values are 'default', 'reverse_distance', 'cooperation'
        :param seed: number of random seeds, None means do not set random seed.
        :return:
        '''

        if seed is not None:
            random.seed(seed)
            # Seed numpy RNG
            np.random.seed(seed)
            # seed the RNG for all devices (both CPU and CUDA)
            torch.manual_seed(seed)

        self.height = height
        self.width = width
        self.past_time = 0  # past steps for env
        self.agents_num = agents_num
        self.reward_type = reward_type
        self.agents_positions = []  # positions for agentss on current step
        for i in range(self.agents_num):
            self.agents_positions.append([-1, -1])  # row and col
        self.agents_positions = np.array(self.agents_positions, dtype=int)
        self.target_positions = np.zeros((width, height), dtype=int)  # targets' positions on current step
        self.agents_move_steps = np.zeros(self.agents_num, dtype=int)  # per agents cost steps to finish the episode
        self.still_working_agents = [True] * self.agents_num  # still working agentss on current step
        self.done = True  # env has finished or not

        # Set up obstacles.
        self.obstacles = np.zeros((width, height), dtype=int)  # the fixed positions for obstacles
        self.obstacles[obstacles_index] = 1

        self.grid = self.obstacles.copy()  # positions for obstacles and agentss on current step

        self.G = nx.Graph()  # adjacent matrix
        # creat adjacent matrix
        self.G.add_nodes_from(list(range(self.height * self.width)))
        for row in range(self.height):
            for col in range(self.width):
                if self.obstacles[row, col] == 0:
                    up_node = (row - 1, col)
                    down_node = (row + 1, col)
                    left_node = (row, col - 1)
                    right_node = (row, col + 1)
                    for node in (up_node, down_node, left_node, right_node):
                        if 0 <= node[0] < self.height and 0 <= node[1] < self.width and self.obstacles[node] == 0:
                            starting_node_id, ending_node_id = row * self.width + col, node[0] * self.width + node[1]
                            self.G.add_edge(starting_node_id, ending_node_id)

    def find_dijkstra_path(self, starting_node, ending_node) -> list:
        '''
        :param starting_node: the location of the starting node, format like (2, 3)
        :param ending_node: the location of the ending node, format like (4, 6)
        :return: dijkstra path from starting_node to ending_node
        '''
        starting_node_id = starting_node[0] * self.width + starting_node[1]
        ending_node_id = ending_node[0] * self.width + ending_node[1]
        path = nx.dijkstra_path(self.G, source=starting_node_id, target=ending_node_id)
        return path

    def path_to_actions(self, path) -> list:
        '''
        :param path: path from some node to other node
        :return: path from some node to other node needs executing actions
        '''
        actions = []
        for i in range(len(path)):
            if i > 0:
                ending_node_id = path[i]
                starting_node_id = path[i - 1]
                d = ending_node_id - starting_node_id
                if d < 0:
                    if d == -1:
                        ac = 1
                    else:
                        ac = 0
                else:
                    if d == 1:
                        ac = 2
                    else:
                        ac = 3
                actions.append(ac)
        return actions

    def update_grid(self):
        '''
        update self.grid, needs executing this method when some agents has moved.
        '''
        self.grid = self.obstacles.copy()
        agents_positions = tuple(self.agents_positions[np.where(self.agents_positions != -1)].reshape((-1, 2)).T.tolist())
        self.grid[agents_positions] = 1
        done = np.array([False] * self.agents_num)
        if len(agents_positions[0]) <= 1:
            self.done = True
            done[:] = True
        else:
            self.done = False
            done[np.where(self.agents_positions[:, 0] == -1)] = True
        return done, agents_positions

    def check_arriving(self, position) -> bool:
        '''
        check input position is target position or not
        :param position: path from some node to other node
        :return:
        '''
        row, col = position
        value = self.target_positions[row, col]
        if value == 0:
            return False
        else:
            return True

    def reset(self):
        '''
        reset the agents(agents) position and target position
        :return: state of env
        '''

        self.done = False

        # # reset fixed agents' starting positions
        # self.agents_positions[:, 0] = self.height - 1
        # self.agents_positions[:, 1] = np.array(range(self.agents_num))
        # reset random agents' starting positions
        empty_grid_position = np.array(np.where(self.grid == 0), dtype=int).T.tolist()  # the positions without obstacles
        self.agents_positions = np.array(random.sample(empty_grid_position, self.agents_num), dtype=int)
        self.agents_move_steps = np.zeros(self.agents_num, dtype=int)

        self.update_grid()

        # reset target position
        empty_grid_position = np.array(np.where(self.grid == 0), dtype=int).T.tolist()  # the positions without agentss and obstacles
        target_positions = np.array(random.sample(empty_grid_position, self.agents_num), dtype=int).T.tolist()
        self.target_positions = np.zeros((self.width, self.height), dtype=int)
        self.target_positions[tuple(target_positions)] = 1

        self.past_time = 0

        return (self.agents_positions.copy(), self.target_positions.copy(), self.obstacles.copy())

    def return_allowed_action(self, agents_position):
        '''
        return actions which can be executed on input position
        :param agents_position: an agents position
        :return: path from some node to other node needs executing actions
        '''

        row_position, col_position = agents_position
        if row_position == -1:  # if agents_position[0] == -1, means this agents has finished its mission
            return {4}
        ac_allowed = {0, 1, 2, 3, 4}  # 0-up, 1-left, 2-right, 3-down, 4-immobility

        if row_position == 0:
            ac_allowed -= {0}
        else:
            if self.grid[row_position - 1, col_position] == 1:
                ac_allowed -= {0}
        if row_position == self.height - 1:
            ac_allowed -= {3}
        else:
            if self.grid[row_position + 1, col_position] == 1:
                ac_allowed -= {3}
        if col_position == 0:
            ac_allowed -= {1}
        else:
            if self.grid[row_position, col_position - 1] == 1:
                ac_allowed -= {1}
        if col_position == self.width - 1:
            ac_allowed -= {2}
        else:
            if self.grid[row_position, col_position + 1] == 1:
                ac_allowed -= {2}

        return ac_allowed

    def excute_action(self, action, position):
        '''
        input a position and an action, return the next position when executing action from input position
        :param action: allowed values are in (0, 1, 2, 3, 4,)
        :param position: some position
        :return: next position when executing action from input position
        '''
        row_position, col_position = position
        if action == 0:
            row_position -= 1
        elif action == 1:
            col_position -= 1
        elif action == 2:
            col_position += 1
        elif action == 3:
            row_position += 1
        return [row_position, col_position]

    def cal_distance(self, position, target_positions):
        '''
        calculate the distances from one position to many target positions respectively
        :param position:
        :param target_positions:
        :return: distance
        '''
        return ((target_positions[:, 0] - position[0]) ** 2 + (target_positions[:, 1] - position[1]) ** 2) ** 0.5

    def reward_func(self, reward_type: str, last_t_working_id: list):
        '''
        calculate rewards from current step when executing some action
        :param reward_type: allowed values in ['default', 'reverse_distance', 'cooperation']

            'default': reward = -1 if done for per agentss or reward = 10
            'reverse_distance': based 'default' reward, and + the reciprocal of the distance to the nearest target position
            'cooperation': based 'reverse_distance' reward, and + 'team spirit' (https://arxiv.org/abs/1912.06680)

        :param last_t_working_id: still working agentss' id on current step
        :return: rewards for per agentss
        '''

        reward = np.array([-1.1] * self.agents_num, dtype=np.float32)
        working_id = np.where(self.agents_positions[:, 0] != -1)[0]
        noworking_id = np.where(self.agents_positions[:, 0] == -1)[0]
        if len(noworking_id) > 0:
            reward[noworking_id] += 10

        if reward_type == 'reverse_distance' or reward_type == 'cooperation':

            target_positions = np.array(np.where(self.target_positions == 1)).T
            working_positions = self.agents_positions[working_id]

            if len(working_positions) > 0:
                scores = []
                for position in working_positions:
                    inverse_distance = 1 / self.cal_distance(position, target_positions)
                    score = np.max(inverse_distance)
                    scores.append(score.copy())
                scores = np.array(scores, dtype=np.float32)
                reward[working_id] += scores

        if reward_type == 'cooperation':

            reward_ = reward
            reward = np.zeros_like(reward_, dtype=np.float32)

            a, b = 0.6, 0.4
            for i in range(self.agents_num):
                use_index = deepcopy(last_t_working_id)
                try:
                    use_index.remove(i)
                except:
                    pass
                if len(use_index) == 0:
                    reward[i] = reward_[i]
                else:
                    reward[i] = reward_[i] * a + reward_[use_index].mean() * b

        return reward

    def step_by_action_probs(self, ac_probs, reward_type: str = None):
        '''
        Env receives all agents' action probability, sampling from it, and make one timestep forward
        :param ac_probs: dict, key allowed in list(range(self.vehicle_num)), key value are torch.tensor like
        [0.2012, 0.1987, 0.2000, 0.2001, 0.2000]
        :param reward_type:
        :return:
        '''

        assert self.done == False, 'You need run reset firstly! '

        if reward_type is None:
            reward_type = self.reward_type

        last_t_working_id = []
        if reward_type == 'cooperation':
            last_t_working_id = np.where(self.agents_positions[:, 0] != -1)[0].tolist()

        actions = [5] * self.agents_num
        next_positions = []
        choose_move_order = list(range(self.agents_num))
        for i in choose_move_order:
            if self.agents_positions[i, 0] != -1:
                ac_allowed = list(self.return_allowed_action(self.agents_positions[i]))
                ac = ac_allowed[torch.multinomial(ac_probs[i][ac_allowed], num_samples=1).item()]
                next_position = self.excute_action(ac, self.agents_positions[i])
                while next_position in next_positions:
                    ac = ac_allowed[torch.multinomial(ac_probs[i][ac_allowed], num_samples=1).item()]
                    next_position = self.excute_action(ac, self.agents_positions[i])
                next_positions.append(next_position)
                if self.check_arriving(position=next_position):
                    self.agents_positions[i, :] = -1
                    self.target_positions[next_position[0], next_position[1]] = 0
                else:
                    self.agents_positions[i] = next_position
                self.agents_move_steps[i] += 1
            else:
                ac = 4
            actions[i] = ac

        done, agents_positions = self.update_grid()
        if self.done and len(agents_positions[0]) == 1:
            starting_node = np.array(agents_positions, dtype=int).T[0]
            ending_node = np.array(np.where(self.target_positions == 1)).T[0]
            final_path_length = len(self.find_dijkstra_path(
                starting_node=starting_node,
                ending_node=ending_node,
            )) - 1
            final_agents_id = np.where(self.agents_positions[:, 0] != -1)[0][0]
            self.agents_move_steps[final_agents_id] += final_path_length
            self.past_time += final_path_length

        reward = self.reward_func(reward_type, last_t_working_id)

        self.past_time += 1

        return actions, (self.agents_positions.copy(), self.target_positions.copy(), self.obstacles.copy()), reward, done


class segment_rl_env(rl_env):
    '''
    Use for large_agents_planning_env's MARL agentss planning
    '''

    def __init__(self, agents_positions: np.ndarray, target_positions: np.ndarray, agents_ids, agents_move_steps):
        '''

        :param agents_positions: agents positions at this segment region
        :param target_positions: target positions at this segment region
        :param agents_ids: agents ids at this segment region
        :param agents_move_steps:
        :return:
        '''

        still_working_id = np.where(agents_positions[:, 0] != -1)[0]
        agents_positions = agents_positions[still_working_id]
        agents_positions %= 10
        self.agents_ids = agents_ids[still_working_id]
        super(segment_rl_env, self).__init__(
            height=10,
            width=10,
            agents_num=len(agents_positions),
            obstacles_index=([5], [5],),
            seed=None,
        )
        self.reset(
            agents_positions=agents_positions,
            target_positions=target_positions,
            agents_move_steps=agents_move_steps[still_working_id],
        )

    def reset(self, agents_positions: np.ndarray, target_positions: np.ndarray, agents_move_steps):
        '''

        :return:
        '''

        self.done = False

        self.agents_positions = agents_positions

        self.update_grid()

        self.target_positions = target_positions

        self.past_time = 0

        self.agents_move_steps = agents_move_steps


class large_agents_planning_env:
    '''
    30 * 30 large env assigning targets and planning by (dijkstra path) and (dijkstra path with MARL)
    '''

    def __init__(
            self, seed: int = 400,
    ):
        '''

        :param seed:
        :return:
        '''

        random.seed(seed)
        # Seed numpy RNG
        np.random.seed(seed)
        # seed the RNG for all devices (both CPU and CUDA)
        torch.manual_seed(seed)

        self.past_time = 0
        self.agents_positions = []
        for i in range(20):
            self.agents_positions.append([-1, -1])  # row and col
        self.agents_positions = np.array(self.agents_positions, dtype=int)
        self.agents_move_steps = np.zeros(20, dtype=int)
        self.target_positions = np.zeros((30, 30), dtype=int)
        self.done = True

        # Set up 9 obstacles.
        self.obstacles = np.zeros((30, 30), dtype=int)
        obstacles_index = (
            [5, 15, 25, 5, 15, 25, 5, 15, 25],
            [5, 5, 5, 15, 15, 15, 25, 25, 25],
        )
        self.obstacles[obstacles_index] = 1

        self.grid = self.obstacles.copy()

        # grid id (node id, position id) = position[0] * 30 + position[1]
        self.grid_id = np.array(list(range(30 * 30)), dtype=int).reshape((30, 30))
        # segment region, dict key is segment region id
        self.segment_part = {
            0: set(self.grid_id[: 10, : 10].flatten().tolist()),
            1: set(self.grid_id[: 10, 10: 20].flatten().tolist()),
            2: set(self.grid_id[: 10, 20:].flatten().tolist()),
            3: set(self.grid_id[10: 20, : 10].flatten().tolist()),
            4: set(self.grid_id[10: 20, 10: 20].flatten().tolist()),
            5: set(self.grid_id[10: 20, 20:].flatten().tolist()),
            6: set(self.grid_id[20:, : 10].flatten().tolist()),
            7: set(self.grid_id[20:, 10: 20].flatten().tolist()),
            8: set(self.grid_id[20:, 20:].flatten().tolist()),
        }

    def update_grid(self):
        self.grid = self.obstacles.copy()
        agents_positions = tuple(self.agents_positions[np.where(self.agents_positions != -1)].reshape((-1, 2)).T.tolist())
        self.grid[agents_positions] = 1
        if len(agents_positions[0]) == 0:
            self.done = True
        else:
            self.done = False
        return self.done

    def reset(self):
        '''
        reset the agents(agents) position and target position
        :return:
        '''

        self.done = False

        # reset fixed agents starting position
        self.agents_positions[:, 0] = 29
        self.agents_positions[:, 1] = np.array(range(20)) + 5
        self.agents_move_steps = np.zeros(20, dtype=int)

        self.update_grid()

        # reset target position
        reset_target_position = True
        while reset_target_position:
            reset_target_position = False
            empty_grid_position = np.array(np.where(self.grid == 0), dtype=int).T.tolist()
            target_positions = np.array(random.sample(empty_grid_position, 20), dtype=int)
            target_positions = target_positions.T.tolist()
            self.target_positions = np.zeros((30, 30), dtype=int)
            self.target_positions[tuple(target_positions)] = 1
            target_positions = np.array(np.where(self.target_positions == 1), dtype=int).T
            segment_target_num = {}
            self.target_segment = []
            for target_position in target_positions:
                target_position_id = target_position[0] * 30 + target_position[1]
                for segment_id in self.segment_part.keys():
                    if target_position_id in self.segment_part[segment_id]:
                        self.target_segment.append(segment_id)
                        if segment_id not in segment_target_num.keys():
                            segment_target_num[segment_id] = 1
                        else:
                            segment_target_num[segment_id] += 1
                        if segment_target_num[segment_id] > 10:
                            reset_target_position = True
                        break

        self.past_time = 0

        return (self.agents_positions.copy(), self.target_positions.copy(), self.obstacles.copy())

    def return_allowed_move(self, agents_position):

        row_position, col_position = agents_position
        if row_position == -1:
            return {4}
        move_allowed = {0, 1, 2, 3, 4}  # 0-up, 1-left, 2-right, 3-down, 4-immobility

        if row_position == 0:
            move_allowed -= {0}
        else:
            if self.grid[row_position - 1, col_position] == 1:
                move_allowed -= {0}
        if row_position == 29:
            move_allowed -= {3}
        else:
            if self.grid[row_position + 1, col_position] == 1:
                move_allowed -= {3}
        if col_position == 0:
            move_allowed -= {1}
        else:
            if self.grid[row_position, col_position - 1] == 1:
                move_allowed -= {1}
        if col_position == 29:
            move_allowed -= {2}
        else:
            if self.grid[row_position, col_position + 1] == 1:
                move_allowed -= {2}

        return move_allowed

    def excute_move(self, move, agents_position):
        row_position, col_position = agents_position
        if move == 0:
            row_position -= 1
        elif move == 1:
            col_position -= 1
        elif move == 2:
            col_position += 1
        elif move == 3:
            row_position += 1
        return [row_position, col_position]

    def cal_distance(self, position, target_position):
        return ((target_position[0] - position[0]) ** 2 + (target_position[1] - position[1]) ** 2) ** 0.5

    def decide_move_order(self, origin, destination):

        move_allowed = list(self.return_allowed_move(origin))
        next_positions_distance = []
        for move in move_allowed:
            next_position = self.excute_move(move, origin)
            distance = self.cal_distance(next_position, destination)
            next_positions_distance.append(distance)
        acs_ = pd.DataFrame({'move': move_allowed, 'distance': next_positions_distance})
        acs_.sort_values(by='distance', ascending=True, inplace=True, ignore_index=True)
        move4_index = acs_.loc[(acs_['move'] == 4)].index[0]
        move_order = acs_['move'].values[: move4_index].tolist()
        move_prob = acs_['move'].values[move4_index:].tolist()

        return move_order, move_prob

    def check_arriving_at_segment(self, agents_id):
        agents_position = self.agents_positions[agents_id]
        agents_position_id = agents_position[0] * 30 + agents_position[1]
        agents_target_segment = self.agents_targets_segment[agents_id]
        if agents_position_id in self.segment_part[agents_target_segment]:
            try:
                self.not_ready_to_rl_planning[agents_target_segment].remove(agents_id)
            except:
                pass
        return agents_target_segment

    def cal_distances(self, target_position, agents_positions):
        return ((agents_positions[:, 0] - target_position[0]) ** 2 + (
                agents_positions[:, 1] - target_position[1]) ** 2) ** 0.5

    def planning(self, agents_targets, target_positions):
        choose_move_order = list(range(20))
        next_positions = []
        for i in choose_move_order:
            if self.agents_positions[i, 0] != -1:
                target_position = target_positions[agents_targets[i]].tolist()
                move_order, move_prob = self.decide_move_order(self.agents_positions[i], target_position)
                next_position = []
                chosen_ac_is_valid = False
                if len(move_order) > 0:
                    move = move_order.pop(0)
                    next_position = self.excute_move(move, self.agents_positions[i])
                    while (next_position in next_positions) and len(move_order) > 0:
                        move = move_order.pop(0)
                        next_position = self.excute_move(move, self.agents_positions[i])
                    if next_position not in next_positions:
                        chosen_ac_is_valid = True
                if not chosen_ac_is_valid:
                    move = random.sample(move_prob, 1)[0]
                    next_position = self.excute_move(move, self.agents_positions[i])
                    while next_position in next_positions:
                        move = random.sample(move_prob, 1)[0]
                        next_position = self.excute_move(move, self.agents_positions[i])
                next_positions.append(next_position)
                if next_position == target_position:
                    self.agents_positions[i, :] = -1
                    self.target_positions[next_position[0], next_position[1]] = 0
                else:
                    self.agents_positions[i] = next_position
                self.agents_move_steps[i] += 1

        done = self.update_grid()

        self.past_time += 1

        return done

    def planning_for_next_control(self, agents_targets, target_positions):
        choose_move_order = list(range(20))
        next_positions = []
        env = {}
        for i in choose_move_order:
            if self.agents_positions[i, 0] != -1:
                target_position = target_positions[agents_targets[i]].tolist()
                move_order, move_prob = self.decide_move_order(self.agents_positions[i], target_position)
                next_position = []
                chosen_ac_is_valid = False
                if len(move_order) > 0:
                    move = move_order.pop(0)
                    next_position = self.excute_move(move, self.agents_positions[i])
                    while (next_position in next_positions) and len(move_order) > 0:
                        move = move_order.pop(0)
                        next_position = self.excute_move(move, self.agents_positions[i])
                    if next_position not in next_positions:
                        chosen_ac_is_valid = True
                if not chosen_ac_is_valid:
                    move = random.sample(move_prob, 1)[0]
                    next_position = self.excute_move(move, self.agents_positions[i])
                    while next_position in next_positions:
                        move = random.sample(move_prob, 1)[0]
                        next_position = self.excute_move(move, self.agents_positions[i])
                next_positions.append(next_position)
                if next_position == target_position:
                    self.agents_positions[i, :] = -1
                    self.target_positions[next_position[0], next_position[1]] = 0
                else:
                    self.agents_positions[i] = next_position
                agents_target_segment = self.check_arriving_at_segment(i)
                if len(self.not_ready_to_rl_planning[agents_target_segment]) == 0:
                    cur_part_agents_id = self.segment_need_contain_agents_id[agents_target_segment]
                    cur_part_agents_positions = self.agents_positions[cur_part_agents_id].copy()
                    self.agents_positions[cur_part_agents_id] = -1
                    if agents_target_segment == 0:
                        cur_part_target = self.target_positions[: 10, : 10].copy()
                        self.target_positions[: 10, : 10] = 0
                    elif agents_target_segment == 1:
                        cur_part_target = self.target_positions[: 10, 10: 20].copy()
                        self.target_positions[: 10, 10: 20] = 0
                    elif agents_target_segment == 2:
                        cur_part_target = self.target_positions[: 10, 20:].copy()
                        self.target_positions[: 10, 20:] = 0
                    elif agents_target_segment == 3:
                        cur_part_target = self.target_positions[10: 20, : 10].copy()
                        self.target_positions[10: 20, : 10] = 0
                    elif agents_target_segment == 4:
                        cur_part_target = self.target_positions[10: 20, 10: 20].copy()
                        self.target_positions[10: 20, 10: 20] = 0
                    elif agents_target_segment == 5:
                        cur_part_target = self.target_positions[10: 20, 20:].copy()
                        self.target_positions[10: 20, 20:] = 0
                    elif agents_target_segment == 6:
                        cur_part_target = self.target_positions[20:, : 10].copy()
                        self.target_positions[20:, : 10] = 0
                    elif agents_target_segment == 7:
                        cur_part_target = self.target_positions[20:, 10: 20].copy()
                        self.target_positions[20:, 10: 20] = 0
                    else:  # elif agents_target_segment == 8
                        cur_part_target = self.target_positions[20:, 20:].copy()
                        self.target_positions[20:, 20:] = 0
                    env[agents_target_segment] = segment_rl_env(
                        agents_positions=cur_part_agents_positions,
                        target_positions=cur_part_target,
                        agents_ids=np.array(cur_part_agents_id, dtype=int),
                        agents_move_steps=self.agents_move_steps[cur_part_agents_id],
                    )
                self.agents_move_steps[i] += 1

        done = self.update_grid()

        self.past_time += 1

        return done, env

    def assigning_targets_and_planning(self):
        '''

        :return:
        '''

        agents_targets = np.zeros(20, dtype=int)
        agents_targets[:] = 20
        target_positions = np.array(np.where(self.target_positions == 1), dtype=int).T
        for i, target_position in enumerate(target_positions):
            target_distances = pd.DataFrame({'agents_id': list(range(20))})
            target_distances['distances'] = self.cal_distances(target_position, self.agents_positions)
            random_bool = [True, False]
            target_distances.sort_values(
                by=['distances', 'agents_id'],
                ascending=[True, random.sample(random_bool, 1)[0]],
                inplace=True,
                ignore_index=True,
            )
            for index in range(20):
                agents_id = target_distances.loc[index, 'agents_id']
                if agents_targets[agents_id] == 20:
                    agents_targets[agents_id] = i
                    break

        done = False
        while not done:
            done = self.planning(agents_targets, target_positions)

        return self.agents_move_steps

    def assigning_targets_and_planning_for_next_control(self):
        '''

        :return:
        '''

        agents_targets = np.zeros(20, dtype=int)
        agents_targets[:] = 20
        target_positions = np.array(np.where(self.target_positions == 1), dtype=int).T
        for i, target_position in enumerate(target_positions):
            target_distances = pd.DataFrame({'agents_id': list(range(20))})
            target_distances['distances'] = self.cal_distances(target_position, self.agents_positions)
            random_bool = [True, False]
            target_distances.sort_values(
                by=['distances', 'agents_id'],
                ascending=[True, random.sample(random_bool, 1)[0]],
                inplace=True,
                ignore_index=True,
            )
            for index in range(20):
                agents_id = target_distances.loc[index, 'agents_id']
                if agents_targets[agents_id] == 20:
                    agents_targets[agents_id] = i
                    break

        self.agents_targets_segment = np.array(self.target_segment, dtype=int)[agents_targets]
        self.not_ready_to_rl_planning = {}
        for i, segment_id in enumerate(self.agents_targets_segment):
            if segment_id not in self.not_ready_to_rl_planning.keys():
                self.not_ready_to_rl_planning[segment_id] = [i]
            else:
                self.not_ready_to_rl_planning[segment_id].append(i)
        self.segment_need_contain_agents_id = deepcopy(self.not_ready_to_rl_planning)

        done = False
        envs = {}
        while not done:
            done, env = self.planning_for_next_control(agents_targets, target_positions)
            if len(env.keys()) > 0:
                for key in env.keys():
                    envs[key] = env[key]

        return envs, self.agents_move_steps


if __name__ == '__main__':

    # large_env = large_agents_planning_env()
    # large_env.reset()
    # envs, agents_move_steps = large_env.assignment_target_by_distance()
    # print(envs.keys())
    # for key in envs.keys():
    #     print(envs[key].agents_positions)
    #     print(envs[key].target_positions)

    from itertools import count
    import pickle
    ac_prob = [torch.tensor([0.2] * 5)] * 10
    env = rl_env(
        obstacles_index=([5], [5],),
        height=10,
        width=10,
        agents_num=10,
    )
    env.reset()
    past_times = []
    for _ in count():
        _, _, _, _ = env.step_by_action_probs(ac_prob)
        if env.done:
            past_times.append(env.past_time)
            env.reset()
        if len(past_times) > 8000:
            break
    with open('random.pickle', 'wb') as file:
        pickle.dump(past_times, file)