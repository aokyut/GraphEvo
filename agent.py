import torch
# from collections import namedtuple
from typing import List, NamedTuple, Optional
import random
from config import Config
import queue


class Transition(NamedTuple):
    state: torch.Tensor  # [1, node_size, state_size]
    action: Optional[torch.Tensor]  # [1, node_size, action_size]
    next_state: torch.Tensor  # [1, node_size, state_size]
    reward: torch.Tensor  # [1, 1, 1]
    done: torch.Tensor  # [1, 1, 1]


class GraphBatch(NamedTuple):
    adj_mat: torch.Tensor  # [batch_size, node_size, node_size]
    # node_feature: torch.Tensor  # [batch_size, node_size, feature_size]
    state: torch.Tensor  # [batch_size, node_size, state_size + feature_size]
    action: torch.Tensor  # [batch_size, node_size, action_size]
    next_state: torch.Tensor  # [batch_size, node_size, state_size]
    reward: torch.Tensor  # [batch_size, 1, 1]
    done: torch.Tensor  # [batch_size, 1, 1] 0 or 1


class GraphFeatureData(NamedTuple):
    adj_mat: torch.Tensor
    transitions: List[Transition]

    def __len__(self):
        return len(self.transitions)

    def sample(self, batch_size: int):
        repeat_size = min(batch_size, self.__len__())
        transitions_list = random.sample(self.transitions, repeat_size)
        transitions_batch = Transition(*zip(*transitions_list))
        return GraphBatch(
            adj_mat=self.adj_mat.repeat(repeat_size, 1, 1),
            state=torch.cat(transitions_batch.state),
            action=torch.cat(transitions_batch.action),
            next_state=torch.cat(transitions_batch.next_state),
            reward=torch.cat(transitions_batch.reward),
            done=torch.cat(transitions_batch.done)
        )


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
        return random.choice(self.memory).sample(batch_size)

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
            self.multi_step_reward = self.gamma ** (len(self.reward_queue) - 1) * reward
            return

        self.reward_queue.append(reward)
        self.multi_step_reward = (self.multi_step_reward - self.reward_queue[0]) / self.gamma + reward * self.gamma ** (self.n_step - 1)
        self.reward_queue = self.reward_queue[-Config.n_step:]

        if done is False:
            self.memory.append(Transition(
                state=self.state_queue.get().reshape(1, self.node_size, -1),
                action=self.action_queue.get().reshape(1, self.node_size, -1),
                next_state=state.reshape(1, -1, Config.state_size),
                reward=torch.Tensor([self.multi_step_reward]).reshape(1, 1, 1),
                done=torch.Tensor([1]).reshape(1, 1, 1)
            ))
        else:
            self.memory.append(Transition(
                state=self.state_queue.get().reshape(1, -1, Config.state_size),
                action=self.action_queue.get().reshape(1, -1, Config.action_size),
                next_state=state.reshape(1, -1, Config.state_size),
                reward=torch.Tensor([self.multi_step_reward]).reshape(1, 1, 1),
                done=torch.Tensor([0]).reshape(1, 1, 1)
            ))

        if done is True:
            for i in range(Config.n_step):
                self.multi_step_reward = (self.multi_step_reward - self.reward_queue[0]) / self.gamma
                self.reward_queue = self.reward_queue[1:] if len(self.reward_queue) > 1 else self.reward_queue
                self.memory.append(Transition(
                    state=self.state_queue.get().reshape(1, -1, Config.state_size),
                    action=self.action_queue.get().reshape(1, -1, Config.action_size),
                    next_state=state.reshape(1, -1, Config.state_size),
                    reward=torch.Tensor([self.multi_step_reward]).reshape(1, 1, 1),
                    done=torch.Tensor([0]).reshape(1, 1, 1)
                ))
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
