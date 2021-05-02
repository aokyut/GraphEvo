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
            shape = list(state.shape)  # (B, S, N, F)
            shape[-1] = u.shape[-1]  # (B, S, N, U)
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


class GraphLSTM(nn.Module):
    def __init__(self, in_size, out_size, activate=None, in_global_size=0, out_global_size=Config.global_size):
        super().__init__()
        self.activate = activate
        self.use_global = False if in_global_size == 0 else True
        self.out_global_size = out_global_size
        self.out_size = out_size
        self.u_lstm = nn.LSTM(in_size * 3 + in_global_size, out_global_size)
        self.lstm = nn.LSTM(in_size * 3 + in_global_size,
                            out_size + out_global_size)

    def forward(self, adj_mat, state, h, c, u=None):
        # local_featureとglobal_featureで一度catし入力した後splitして出力する。
        x_in = []

        shape = list(state.shape)  # (B, S, N, F)
        if self.use_global:
            shape[-1] = u.shape[-1]  # (B, S, N, U)
            x_in.append(u.unsqueeze(-2).expand(shape))
        x_in.append(torch.matmul(self._preprocess_adj(adj_mat), state))
        x_in.append(torch.matmul(self._preprocess_adj(torch.transpose(adj_mat, -2, -1)), state))
        x_in.append(state)
        if len(shape) == 2:
            l_in = torch.cat(x_in, dim=-1)
            features, (h_out, c_out) = self.lstm(l_in.unsqueeze(), (h, c))
            vertex_feature, global_feature = torch.split(features.squeeze(), [self.out_size, self.out_global_size], dim=-1)
            # vertex_feature: [N, F] , global_feature: [N, G]
        else:
            l_in = torch.cat(x_in, dim=-1)
            l_in = torch.cat(l_in.split(1, dim=0), dim=2).squeeze()
            features, (h_out, c_out) = self.lstm(l_in) if (h is None) else self.lstm(l_in, (h, c))
            features = torch.cat(features.unsqueeze().split(shape[-2], dim=-2), dim=0)
            # feature: [B, S, N, F + G]
            vertex_feature, global_feature = torch.splitt(features, [self.out_size, self.out_global_size], dim=-1)

        if self.activate is None:
            return vertex_feature, global_feature, h_out, c_out
        else:
            return self.activate(vertex_feature), self.activate(global_feature), h_out, c_out

    def _preprocess_adj(self, adj):
        return F.normalize(adj, p=1, dim=-1)


class GraphPolicy(nn.Module):
    def __init__(self,
            in_size=Config.network_in_size,
            out_size=Config.network_out_size):
        super().__init__()

        layers = [
            GraphLinear(in_size=in_size, out_size=64, activate=nn.LeakyReLU()),
            GraphLinear(in_size=64, out_size=32, activate=nn.LeakyReLU(), in_global_size=Config.global_size),
            GraphLSTM(in_size=32, out_size=Config.lstm_h_size, activate=nn.LeakyReLU(), in_global_size=Config.global_size),
            GraphLinear(in_size=Config.lstm_h_size, out_size=3, in_global_size=Config.global_size)
        ]

        self.pre_network = nn.ModuleList(layers)

    def forward(self, adj_mat, x, h=None, c=None):
        # adj_mat = adj_mat.squeeze()
        # x = x.squeeze()
        u = None
        for layer in self.pre_network:
            if isinstance(layer, GraphLSTM):
                x, u, h, c = layer(adj_mat, x, h, c, u)
            elif isinstance(layer, GraphLinear):
                x, u = layer(adj_mat, x, u)
            else:
                x = layer(x)

        action_probs = F.softmax(2 * x.tanh(), dim=-1)
        shape = torch.Size(action_probs.shape)
        actions = torch.multinomial(action_probs.reshape(-1, shape[-1]), 1, True).reshape(shape[:-1])

        return actions.float() - 1, h, c

    def pi_and_log_prob_pi(self, adj_mat, x, h=None, c=None):

        u = None
        for layer in self.pre_network:
            if isinstance(layer, GraphLSTM):
                x, u, h, c = layer(adj_mat, x, h, c, u)
            elif isinstance(layer, GraphLinear):
                x, u = layer(adj_mat, x, u)
            else:
                x = layer(x)

        action_probs = F.softmax(2 * x.tanh(), dim=-1)

        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)
        return action_probs, log_action_probs


class GraphQNetwork(nn.Module):
    def __init__(self, in_size=Config.network_in_size,
                       out_size=Config.network_out_size,
                       hidden_layers=Config.hidden_layers):
        super().__init__()
        layers = [
            GraphLinear(in_size=in_size, out_size=64, activate=nn.LeakyReLU()),
            GraphLinear(in_size=64, out_size=32, activate=nn.LeakyReLU(), in_global_size=Config.global_size),
            GraphLSTM(in_size=32, out_size=Config.lstm_h_size, activate=nn.LeakyReLU(), in_global_size=Config.global_size),
            GraphLinear(in_size=Config.lstm_h_size, out_size=3, in_global_size=Config.global_size)
        ]

        self.network = nn.ModuleList(layers)

    def forward(self, adj_mat, x, h=None, c=None):
        u = None
        for layer in self.network:
            if isinstance(layer, GraphLSTM):
                x, u, h, c = layer(adj_mat, x, h, c, u)
            elif isinstance(layer, GraphLinear):
                x, u = layer(adj_mat, x, u)
            else:
                x = layer(x)
        return x, h, c


class GraphSAC(nn.Module):
    def __init__(self, in_size=Config.network_in_size,
                 out_size=Config.network_out_size):
        super().__init__()
        self.policy = GraphPolicy()
        self.q1 = GraphQNetwork(
        )
        self.q2 = GraphQNetwork(
        )
        self.q1_tar = GraphQNetwork(
        )
        self.q2_tar = GraphQNetwork(
        )

        self.copy_param()
        self.alpha = torch.autograd.Variable(torch.tensor(Config.alpha), requires_grad=True)

    def copy_param(self):
        self.q1_tar.load_state_dict(self.q1.state_dict())
        self.q2_tar.load_state_dict(self.q2.state_dict())

    def get_action(self, adj_mat, x):
        return self.policy(adj_mat, x)

    def update_target(self, rho):
        for q_p, q_tar_p in zip(self.q1.parameters(), self.q1_tar.parameters()):
            q_tar_p.data = rho * q_tar_p.data + (1 - rho) * q_p.data
        for q_p, q_tar_p in zip(self.q2.parameters(), self.q2_tar.parameters()):
            q_tar_p.data = rho * q_tar_p.data + (1 - rho) * q_p.data

    def compute_q_target(self, adj_mat, state, gamma, reward, done):
        """
        reward: n-step target value [B, 1, 1]
        state: n-step future state [B, N, state-size]
        """
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
        target_q1, target_q2 = self.q_function(adj_mat, state)

        H = - torch.sum(action_probs * log_prob_pi, dim=2)

        return torch.sum(torch.min(target_q1, target_q2) * action_probs, dim=-1), H.squeeze()

    def compute_alpha_target(self, adj_mat, state):
        action_probs, log_prob_pi = self.policy.pi_and_log_prob_pi(adj_mat, state)
        entropy = - torch.sum(action_probs * log_prob_pi, dim=-1)
        return self.call_alpha() * (entropy - Config.target_entropy)

    def call_alpha(self):
        # return torch.exp(self.alpha).clamp(0.01, 0.5)
        return 0.245 * torch.sin(self.alpha) + 0.255

    def rescaling(self, x, epsilon=EPS):
        n = math.sqrt(abs(x) + 1) - 1
        return torch.sign(x) * n + epsilon * x

    def rescaling_inverse(x):
        return torch.sign(x) * ((x + torch.sign(x)) ** 2 - 1)
