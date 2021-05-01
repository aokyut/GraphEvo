import torch
# from collections import namedtuple
from typing import List, NamedTuple, Optional
import random
from config import Config
import queue
import numpy as np


class Transition(NamedTuple):
    state: torch.Tensor  # [1, node_size, state_size]
    action: Optional[torch.Tensor]  # [1, node_size, action_size]
    next_state: torch.Tensor  # [1, node_size, state_size]
    reward: torch.Tensor  # [1, 1, 1]
    done: torch.Tensor  # [1, 1, 1]


class GraphBatch(NamedTuple):
    adj_mat: torch.Tensor       # [batch_size, seq_size, node_size, node_size]
    state: torch.Tensor         # [batch_size, seq_size, node_size, state_size]
    bi_state: torch.Tensor      # [batch_size, burn_in_size, node_size, state_size]
    bi_adj: torch               # [batch_size, burn_in_size, node_size, node_size]
    action: torch.Tensor        # [batch_size, seq_size, node_size, action_size]
    next_state: torch.Tensor    # [batch_size, seq_size, node_size, state_size]
    reward: torch.Tensor        # [batch_size, seq_size, 1, 1]
    done: torch.Tensor          # [batch_size, seq_size, 1, 1] 0 or 1


class GraphFeatureData(NamedTuple):
    """
    一つのエピソードを保存するクラス
    グラフの隣接行列と１エピソード分のTransitionを保持し
    sampleでバッチを出力するクラス。
    """
    adj_mat: torch.Tensor
    transitions: List[Transition]

    def __len__(self):
        return len(self.transitions)

    def ok(self):
        return int(self.__len__() > (Config.burn_in + Config.seq_in))

    def sample(self, batch_size: int):
        repeat_size = min(batch_size, self.__len__())
        transitions_list = []
        for i in range(Config.batch_size):
            start_index = random.randint(0, self.__len__() - Config.pick_out_range)
            transitions_burn_in = Transition(*zip(*self.transitions[start_index: start_index + Config.burn_in]))
            transitions_seq_in = Transition(*zip(*self.transitions[start_index + Config.burn_in: start_index + Config.pick_out_range]))
            transitions_list.append(
                GraphBatch(
                    adj_mat=self.adj_mat.repeat(self.Config.seq_in, 1, 1).unsqueeze(),
                    state=torch.cat(transitions_seq_in.state).unsqueeze(),
                    bi_adj=self.adj_mat.repeat(self.Config.burn_in, 1, 1).unsqueeze(),
                    bi_state=torch.cat(transitions_burn_in.state).unsqueeze(),
                    action=torch.cat(transitions_seq_in.action).unsqueeze(),
                    next_state=torch.cat(transitions_seq_in.next_state).unsqueeze(),
                    reward=torch.cat(transitions_seq_in.reward).unsqueeze(),
                    done=torch.cat(transitions_seq_in.done).unsqueeze()
                )
            )
        transitions_list = random.sample(self.transitions, repeat_size)
        transitions_batch = GraphBatch(*zip(*transitions_list))
        return GraphBatch(
            adj_mat=torch.cat(transitions_batch.adj_mat),
            state=torch.cat(transitions_batch.state),
            bi_adj=torch.cat(transitions_batch.bi_adj),
            bi_state=torch.cat(transitions_batch.bi_state),
            next_state=torch.cat(transitions_batch.next_state),
            action=torch.cat(transitions_batch.action),
            reward=torch.cat(transitions_batch.reward),
            done=torch.cat(transitions_batch.done),
        )
        # shape[B, S, N, F]


class GraphDataset():
    def __init__(self, train_func, log_reward):
        self.hoge = "fuga"
        self.memory: List[GraphFeatureData] = []
        self.batch_size = Config.batch_size
        self.bundle_size = Config.bundle_size
        self.train_func = train_func
        self.log_reward = log_reward
        self.size = Config.dataset_eps_size
        self.index = 0
        self.step_width = Config.dataset_eps_size // 2

    def __len__(self):
        return len(self.memory)

    def get_batchs(self):
        return [self.sample(self.batch_size) for i in range(self.bundle_size)]

    def sample(self, batch_size: int):  # type: ignore
        # サイズが足りないシークエンスを選ばないようにする。
        choice_flag = np.array([f_data.ok() for f_data in self.memory])
        p_dist = choice_flag / choice_flag.sum()
        return np.random.choice(self.memory, p=p_dist)

    def push(self, data: GraphFeatureData):
        if len(self.memory) < self.size:
            while len(self.memory) < self.size:
                self.memory.append(data)
        else:
            self.memory[self.index] = data
            self.index = (self.index + 1) % self.size
            if (self.index % Config.learn_freq) == 0:
                self.train_func(self.get_batchs())

    def _push(self, data: GraphFeatureData, num: int):
        learn = False
        for i in range(num):
            self.memory[self.index] = data
            if (self.index + 1 == self.size):
                self.step_width = (self.step_width + 1) // 2
            self.index = (self.index + 1) % self.size
            learn |= self.index % Config.learn_freq == 0
        if (learn):
            self.train_func(self.get_batchs())


class Agent:
    def __init__(self, adj_mat: torch.Tensor,
                       dataset: GraphDataset):
        self.adj_mat = adj_mat
        self.node_size = adj_mat.shape[0]
        self.memory: List[Transition] = []
        self.last_state = None
        self.last_action = None
        self.acum_reward = 0
        self.dataset = dataset
        self.state_queue = queue.Queue()
        self.action_queue = queue.Queue()
        self.reward_queue = []
        self.multi_step_reward = 0
        self.gamma = Config.gamma
        self.n_step = Config.n_step

    def __str__(self) -> str:
        return "Adjacency Matrix\n" + str(self.adj_mat) + "\n" + \
            "Size Matrix\n" + self.size_mat + "\n" + \
            "Connect Matrix\n" + self.connect_mat + "\n" + \
            "Type Matrix\n" + self.type_mat

    def reset(self):
        self.state_queue = queue.Queue()
        self.action_queue = queue.Queue()
        self.reward_queue = []
        self.memory: List[Transition] = []
        self.acum_reward = 0
        self.multi_step_reward = 0

    def push_data(self, state: torch.Tensor,  # [node_size, state_size]
                        action: torch.Tensor,  # [node_size, acion_size]
                        reward: float,
                        done: bool):
        self.acum_reward += reward

        if self.state_queue.qsize() < self.n_step:
            self.state_queue.put(state)
            self.action_queue.put(action)
            self.reward_queue.append(reward)
            self.multi_step_reward = self.gamma * self.multi_step_reward + reward
            return

        self.reward_queue.append(reward)
        self.multi_step_reward = self.gamma * (self.multi_step_reward - self.gamma ** (self.n_step - 1) * self.reward_queue[0]) + reward
        self.reward_queue = self.reward_queue[-Config.n_step:]

        if done is False:
            self.memory.append(Transition(
                self.state_queue.get().reshape(1, -1, Config.state_size),
                self.action_queue.get().reshape(1, -1, Config.action_size),
                state.reshape(1, -1, Config.state_size),
                torch.Tensor([self.multi_step_reward]).reshape(1, 1, 1),
                torch.Tensor([1]).reshape(1, 1, 1)
            ))
        else:
            self.memory.append(Transition(
                self.state_queue.get().reshape(1, -1, Config.state_size),
                self.action_queue.get().reshape(1, -1, Config.action_size),
                state.reshape(1, -1, Config.state_size),
                torch.Tensor([self.multi_step_reward]).reshape(1, 1, 1),
                torch.Tensor([0]).reshape(1, 1, 1)
            ))

        if done is True:
            self.episode_end()
            return

        self.state_queue.put(state)
        self.action_queue.put(action)

    def episode_end(self):
        set_data: GraphFeatureData = GraphFeatureData(adj_mat=self.adj_mat,
            transitions=self.memory)
        self.dataset.push(set_data)
        self.dataset.log_reward(self.acum_reward)
        self.reset()
