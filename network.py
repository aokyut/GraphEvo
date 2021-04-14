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
    def __init__(self, in_size, out_size, activate=None):
        super().__init__()
        self.linear_in = nn.Linear(
            in_features=in_size,
            out_features=out_size
        )
        self.linear_out = nn.Linear(
            in_features=in_size,
            out_features=out_size
        )
        self.linear_loop = nn.Linear(
            in_features=in_size,
            out_features=out_size
        )
        self.activate = activate

    def forward(self, adj_mat, state):
        # グラフの順方向への畳み込み
        l_rel_in = self.linear_in(torch.matmul(adj_mat, state))
        # グラフの逆方向への畳み込み
        l_rel_out = self.linear_out(torch.matmul(torch.transpose(adj_mat, -2, -1), state))
        # グラフの自己ノードの畳み込み
        l_self = self.linear_loop(state)

        if self.activate is None:
            return l_rel_in + l_rel_out + l_self
        else:
            return self.activate(l_rel_in + l_rel_out + l_self)


class GraphGaussianPolicy(nn.Module):
    def __init__(self,
            in_size=Config.network_in_size,
            out_size=Config.network_out_size,
            hidden_layers=Config.hidden_layers):
        super().__init__()

        layers = [
            GraphLinear(in_size, hidden_layers[0], nn.LeakyReLU())
        ]

        for i in range(1, len(hidden_layers)):
            layers.append(
                GraphLinear(hidden_layers[i - 1], hidden_layers[i], nn.LeakyReLU())
            )

        layers.append(GraphLinear(hidden_layers[-1], out_size))
        self.pre_network = nn.ModuleList(layers)

    def forward(self, adj_mat, x):
        # adj_mat = adj_mat.squeeze()
        # x = x.squeeze()
        for layer in self.pre_network:
            x = layer(adj_mat, x)

        action_probs = F.softmax(x.tanh(), dim=-1)
        shape = torch.Size(action_probs.shape)
        actions = torch.multinomial(action_probs.reshape(-1, shape[-1]), 1, True).reshape(shape[:-1])

        return actions.float() - 1

    def pi_and_log_prob_pi(self, adj_mat, x):

        for layer in self.pre_network:
            x = layer(adj_mat, x)

        action_probs = F.softmax(x.tanh(), dim=-1)

        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return action_probs, log_action_probs


class GraphQNetwork(nn.Module):
    def __init__(self, in_size=Config.network_in_size,
                       out_size=Config.network_out_size,
                       hidden_layers=Config.hidden_layers):
        super().__init__()
        layers = [GraphLinear(in_size, hidden_layers[0], nn.LeakyReLU())]
        for i in range(1, len(hidden_layers)):
            layers.append(
                GraphLinear(hidden_layers[i - 1], hidden_layers[i])
            )
        layers.append(GraphLinear(hidden_layers[-1], out_size))

        self.network = nn.ModuleList(layers)

    def forward(self, adj_mat, x):
        for layer in self.network:
            x = layer(adj_mat, x)
        return x


class GraphSAC(nn.Module):
    def __init__(self, in_size=Config.network_in_size,
                 out_size=Config.network_out_size):
        super().__init__()
        self.policy = GraphGaussianPolicy()

        # self.q = GraphValueFunction(
        #     Config.network_in_size + Config.network_out_size
        # )

        self.q1 = GraphQNetwork(
        )
        self.q2 = GraphQNetwork(
        )
        self.q1_tar = GraphQNetwork(
        )
        self.q2_tar = GraphQNetwork(
        )

        # self.q_tar = GraphValueFunction(
        #     Config.network_in_size + Config.network_out_size
        # )

        self.copy_param()
        self.alpha = torch.autograd.Variable(torch.tensor(Config.alpha), requires_grad=True)

    def copy_param(self):
        # self.v_tar.load_state_dict(self.v.state_dict())
        # self.q_tar.load_state_dict(self.q.state_dict())
        self.q1_tar.load_state_dict(self.q1.state_dict())
        self.q2_tar.load_state_dict(self.q2.state_dict())

    def get_action(self, adj_mat, x):
        return self.policy(adj_mat, x)

    def update_target(self, rho):
        # for v_p, v_tar_p in zip(self.v.parameters(), self.v_tar.parameters()):
        #     v_tar_p.data = rho * v_tar_p.data + (1 - rho) * v_p.data
        for q_p, q_tar_p in zip(self.q1.parameters(), self.q1_tar.parameters()):
            q_tar_p.data = rho * q_tar_p.data + (1 - rho) * q_p.data
        for q_p, q_tar_p in zip(self.q2.parameters(), self.q2_tar.parameters()):
            q_tar_p.data = rho * q_tar_p.data + (1 - rho) * q_p.data

    # def compute_v_target(self, adj_mat, state):
    #     pi, log_prob_pi = self.policy.pi_and_log_prob_pi(adj_mat, state)
    #     q1, q2 = self.q1(adj_mat, torch.cat([state, pi], dim=2)), self.q2(adj_mat, torch.cat([state, pi], dim=2))
    #     q = torch.min(q1, q2).squeeze()
    #     return (q - self.call_alpha() * log_prob_pi.squeeze()).detach()

    def compute_q_target(self, adj_mat, state, gamma, reward, done):
        action_probs, log_prob_pi = self.policy.pi_and_log_prob_pi(adj_mat, state)
        target_q1, target_q2 = self.q_function(adj_mat, state)

        value = (action_probs * (torch.min(target_q1, target_q2) - self.call_alpha() * log_prob_pi)).sum(dim=2)
        return (reward.squeeze(-1) + gamma * done.squeeze(-1) * value.squeeze()).detach()

    def q_function(self, adj_mat, state):
        q1, q2 = self.q1(adj_mat, state), self.q2(adj_mat, state)
        return q1.squeeze(), q2.squeeze()

    def q_value(self, adj_mat, state, action):
        q1, q2 = self.q1(adj_mat, state), self.q2(adj_mat, state)
        index = (action + 1).round().long()
        q1 = torch.gather(q1, dim=2, index=index)
        q2 = torch.gather(q2, dim=2, index=index)
        return q1.squeeze(), q2.squeeze()

    def q_function_entropy(self, adj_mat, state):
        action_probs, log_prob_pi = self.policy.pi_and_log_prob_pi(adj_mat, state)
        # q_value = self.q_tar(adj_mat, torch.cat([state, pi], 2)).squeeze()
        target_q1, target_q2 = self.q_function(adj_mat, state)

        H = - torch.sum(action_probs * log_prob_pi, dim=2)

        return torch.sum(torch.min(target_q1, target_q2) * action_probs, dim=-1), H.squeeze()

    def compute_alpha_target(self, adj_mat, state):
        action_probs, log_prob_pi = self.policy.pi_and_log_prob_pi(adj_mat, state)
        entropy = - torch.sum(action_probs * log_prob_pi, dim=2)
        return self.call_alpha() * (entropy - Config.target_entropy)

    def call_alpha(self):
        return torch.exp(self.alpha)
