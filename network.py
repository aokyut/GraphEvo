import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
import math

EPS = 1e-8
LOG_STD_MAX = 2
LOG_STD_MIN = -10
CONST_GAUSSIAN = math.log(math.pi * 2)


class GraphLinear(nn.Module):
    def __init__(self, in_size, out_size, activate=None, in_global_size=0, out_global_size=Config.global_size):
        super().__init__()
        self.activate = activate
        self.use_global = False if in_global_size == 0 else True
        self.global_size = in_global_size
        self.linear = nn.Linear(in_size * 3 + in_global_size, out_size)
        self.u_linear = nn.Linear(in_size * 3 + in_global_size, out_global_size)

    def forward(self, adj_mat, state, u=None):
        x_in = []
        # print(self.global_size)
        if self.use_global:
            shape = list(state.shape)  # (B, N, F)
            shape[-1] = u.shape[-1]  # (B, N, U)
            x_in.append(u.unsqueeze(-2).expand(shape))
        x_in.append(torch.matmul(self._preprocess_adj(adj_mat), state))
        x_in.append(torch.matmul(self._preprocess_adj(torch.transpose(adj_mat, -2, -1)), state))
        x_in.append(state)

        l_in = torch.cat(x_in, dim=-1)
        vertex_feature = self.linear(l_in)
        global_feature = self.u_linear(torch.mean(l_in, dim=-2))

        if self.activate is None:
            return vertex_feature, global_feature
        else:
            return self.activate(vertex_feature), self.activate(global_feature)

    def _preprocess_adj(self, adj):
        return F.normalize(adj, p=1, dim=-1)


class GraphQNetwork(nn.Module):
    def __init__(self, in_size=Config.network_in_size,
                       out_size=Config.network_out_size,
                       hidden_layers=Config.hidden_layers):
        super().__init__()
        layers = [
            GraphLinear(in_size=in_size, out_size=64, activate=nn.LeakyReLU()),
            GraphLinear(in_size=64, out_size=32, activate=nn.LeakyReLU(), in_global_size=Config.global_size),
            GraphLinear(in_size=32, out_size=16, activate=nn.LeakyReLU(), in_global_size=Config.global_size),
            GraphLinear(in_size=16, out_size=3, in_global_size=Config.global_size)
        ]

        self.network = nn.ModuleList(layers)

    def get_q(self, adj_mat, x):
        u = None
        for layer in self.network:
            if isinstance(layer, GraphLinear):
                x, u = layer(adj_mat, x, u)
            else:
                x = layer(x)
        return x

    def forward(self, adj_mat, x):
        # adj_mat = adj_mat.squeeze()
        # x = x.squeeze()
        u = None
        for layer in self.network:
            if isinstance(layer, GraphLinear):
                x, u = layer(adj_mat, x, u)
            else:
                x = layer(x)

        # x = 2 * x.tanh()
        action_probs = F.softmax(x, dim=-1)
        shape = torch.Size(action_probs.shape)
        actions = torch.multinomial(action_probs.reshape(-1, shape[-1]), 1, True).reshape(shape[:-1])

        return actions.float() - 1


class GraphQ(nn.Module):
    def __init__(self, in_size=Config.network_in_size,
                 out_size=Config.network_out_size):
        super().__init__()
        self.q1 = GraphQNetwork(
        )
        self.q2 = GraphQNetwork(
        )
        self.q1_tar = GraphQNetwork(
        )
        self.q2_tar = GraphQNetwork(
        )

        self.copy_param()

    def copy_param(self):
        self.q1_tar.load_state_dict(self.q1.state_dict())
        self.q2_tar.load_state_dict(self.q2.state_dict())

    def update_target(self, rho):
        for q_p, q_tar_p in zip(self.q1.parameters(), self.q1_tar.parameters()):
            q_tar_p.data = rho * q_tar_p.data + (1 - rho) * q_p.data
        for q_p, q_tar_p in zip(self.q2.parameters(), self.q2_tar.parameters()):
            q_tar_p.data = rho * q_tar_p.data + (1 - rho) * q_p.data

    def compute_q_target(self, adj_mat, state, gamma, reward, done):
        target_q1, target_q2 = self.q_function_tar(adj_mat, state)

        value = torch.min(target_q1, target_q2)

        return (reward.squeeze(-1) + gamma * done.squeeze(-1) * value.max(dim=-1).values).detach()

    def q_function(self, adj_mat, state):
        q1, q2 = self.q1.get_q(adj_mat, state), self.q2.get_q(adj_mat, state)
        return q1.squeeze(), q2.squeeze()

    def q_function_tar(self, adj_mat, state):
        q1, q2 = self.q1_tar.get_q(adj_mat, state), self.q2_tar.get_q(adj_mat, state)
        return q1.squeeze(), q2.squeeze()

    def q_value(self, adj_mat, state, action):
        q1, q2 = self.q1.get_q(adj_mat, state), self.q2.get_q(adj_mat, state)
        index = (action + 1).round().long()
        q1 = torch.gather(q1, dim=2, index=index)
        q2 = torch.gather(q2, dim=2, index=index)
        return q1.squeeze(), q2.squeeze()
