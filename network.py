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
    def __init__(self, in_size, out_size, activate=True):
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

        if self.activate:
            return F.leaky_relu(l_rel_in + l_rel_out + l_self)
        else:
            return l_rel_in + l_rel_out + l_self


class GraphGaussianPolicy(nn.Module):
    def __init__(self,
            in_size=Config.network_in_size,
            out_size=Config.network_out_size,
            hidden_layers=Config.hidden_layers):
        super().__init__()

        layers = [
            GraphLinear(in_size, hidden_layers[0])
        ]

        for i in range(1, len(hidden_layers)):
            layers.append(
                GraphLinear(hidden_layers[i - 1], hidden_layers[i])
            )

        self.pre_network = nn.ModuleList(layers)
        self.mu = GraphLinear(hidden_layers[-1], out_size, activate=False)
        self.log_sig = GraphLinear(hidden_layers[-1], out_size, activate=False)

    def forward(self, adj_mat, x):
        # adj_mat = adj_mat.squeeze()
        # x = x.squeeze()
        for layer in self.pre_network:
            x = layer(adj_mat, x)
        # print(x.shape)
        mu = self.mu(adj_mat, x)
        log_sig = torch.tanh(self.log_sig(adj_mat, x))
        log_sig = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_sig + 1)

        return torch.tanh(mu + torch.randn_like(mu) * log_sig.exp())

    def pi_and_log_prob_pi(self, adj_mat, x):
        for layer in self.pre_network:
            x = layer(adj_mat, x)
        mu = self.mu(adj_mat, x)
        log_sig = torch.tanh(self.log_sig(adj_mat, x))
        log_sig = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_sig + 1)
        pi = torch.tanh(mu + torch.randn_like(mu) * log_sig.exp())
        # compute prob pi
        # print("pi", pi.shape)
        # print(mu.shape)
        # print(log_sig.shape)
        # print("mu", mu)
        # print("log_sig", log_sig)

        log_prob_pi = torch.sum(-0.5 * (((pi - torch.tanh(mu)) / (log_sig.exp() + EPS))**2 + 2 * log_sig + CONST_GAUSSIAN), 2)
        log_prob_pi = log_prob_pi - torch.log((1 - pi ** 2) + EPS).sum(2)
        # print("log_prob_pi", log_prob_pi)
        # print("pi", pi)
        return pi, log_prob_pi


class GraphValueFunction(nn.Module):
    def __init__(self, in_size, hidden_layers=Config.hidden_layers):
        super().__init__()
        layers = [GraphLinear(in_size, hidden_layers[0])]
        for i in range(1, len(hidden_layers)):
            layers.append(
                GraphLinear(hidden_layers[i - 1], hidden_layers[i])
            )
            # layers.append(nn.LeakyReLU())
        layers.append(GraphLinear(hidden_layers[-1], 1, activate=False))
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

        self.q1 = GraphValueFunction(
            Config.network_in_size + Config.network_out_size
        )
        self.q2 = GraphValueFunction(
            Config.network_in_size + Config.network_out_size
        )
        self.q1_tar = GraphValueFunction(
            Config.network_in_size + Config.network_out_size
        )
        self.q2_tar = GraphValueFunction(
            Config.network_in_size + Config.network_out_size
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
        # v_value = self.v_tar(adj_mat, state).squeeze()
        # done = done.squeeze(-1)
        # print("v_value", v_value.shape)
        # print("done", done.shape)
        # return (reward.squeeze(-1) + gamma * done.squeeze(-1) * self.v_tar(adj_mat, state).squeeze()).detach()
        pi, log_prob_pi = self.policy.pi_and_log_prob_pi(adj_mat, state)
        # q_value = self.q_tar(adj_mat, torch.cat([state, pi], 2)).squeeze()
        target_q1, target_q2 = self.q_function(adj_mat, state, pi)
        # print("pi", pi.shape)
        # print("log_prob_pi", log_prob_pi.shape)
        # print("q_value", q_value.shape)
        value = torch.min(target_q1, target_q2) - self.call_alpha() * log_prob_pi
        return (reward.squeeze(-1) + gamma * done.squeeze(-1) * value.squeeze()).detach()

    def q_function(self, adj_mat, state, pi):
        q1, q2 = self.q1(adj_mat, torch.cat([state, pi], 2)), self.q2(adj_mat, torch.cat([state, pi], 2))
        return q1.squeeze(), q2.squeeze()

    def q_function_entropy(self, adj_mat, state):
        pi, log_prob_pi = self.policy.pi_and_log_prob_pi(adj_mat, state)
        # q_value = self.q_tar(adj_mat, torch.cat([state, pi], 2)).squeeze()
        target_q1, target_q2 = self.q_function(adj_mat, state, pi)
        H = -log_prob_pi
        return torch.min(target_q1, target_q2), H.squeeze()

    def compute_alpha_target(self, adj_mat, state):
        _, log_prob_pi = self.policy.pi_and_log_prob_pi(adj_mat, state)
        return self.call_alpha() * (- log_prob_pi - Config.target_entropy)

    def call_alpha(self):
        return torch.exp(self.alpha)
