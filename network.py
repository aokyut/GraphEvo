import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from agent import GraphBatch
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
        # print(f"adj:{adj_mat.shape}, state:{state.shape}, u:{u.shape if not u is None else u}")
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
        # print(f"adj:{adj_mat.shape}, state:{state.shape}, u:{u.shape}, h:{h.shape if not h is None else h}, c:{c.shape if not c is None else c}")

        shape = list(state.shape)  # (B, S, N, F)
        if self.use_global:
            shape[-1] = u.shape[-1]  # (B, S, N, U)
            x_in.append(u.unsqueeze(-2).expand(shape))
        x_in.append(torch.matmul(self._preprocess_adj(adj_mat), state))
        x_in.append(torch.matmul(self._preprocess_adj(torch.transpose(adj_mat, -2, -1)), state))
        x_in.append(state)
        if len(shape) == 2:
            l_in = torch.cat(x_in, dim=-1)
            features, (h_out, c_out) = self.lstm(l_in.unsqueeze(dim=0), (h, c))
            vertex_feature, global_feature = torch.split(features.squeeze(), [self.out_size, self.out_global_size], dim=-1)
            global_feature = global_feature.mean(dim=-2)
            # vertex_feature: [N, F] , global_feature: [N, G]
        else:
            l_in = torch.cat(x_in, dim=-1)
            l_in = torch.cat(l_in.split(1, dim=0), dim=2).squeeze()
            features, (h_out, c_out) = self.lstm(l_in) if (h is None) else self.lstm(l_in, (h, c))
            features = torch.cat(features.unsqueeze(0).split(shape[-2], dim=-2), dim=0)
            # feature: [B, S, N, F + G]
            vertex_feature, global_feature = torch.split(features, [self.out_size, self.out_global_size], dim=-1)
            global_feature = global_feature.mean(dim=-2)

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
            GraphLinear(in_size=in_size, out_size=32, activate=nn.LeakyReLU()),
            GraphLSTM(in_size=32, out_size=16, activate=nn.LeakyReLU(), in_global_size=Config.global_size),
            GraphLinear(in_size=16, out_size=3, in_global_size=Config.global_size)
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
            GraphLinear(in_size=in_size, out_size=32, activate=nn.LeakyReLU()),
            GraphLSTM(in_size=32, out_size=16, activate=nn.LeakyReLU(), in_global_size=Config.global_size),
            GraphLinear(in_size=16, out_size=3, in_global_size=Config.global_size)
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
        self.gamma = Config.gamma

    def copy_param(self):
        self.q1_tar.load_state_dict(self.q1.state_dict())
        self.q2_tar.load_state_dict(self.q2.state_dict())

    def update_target(self, rho):
        for q_p, q_tar_p in zip(self.q1.parameters(), self.q1_tar.parameters()):
            q_tar_p.data = rho * q_tar_p.data + (1 - rho) * q_p.data
        for q_p, q_tar_p in zip(self.q2.parameters(), self.q2_tar.parameters()):
            q_tar_p.data = rho * q_tar_p.data + (1 - rho) * q_p.data

    def compute_q_loss(self, batch: GraphBatch):
        """
        reward: n-step target value [B, 1, 1]
        state: n-step future state [B, N, state-size]
        """
        state, adj = batch.state[:, :-Config.n_step, :, :], batch.adj_mat[:, :-Config.n_step, :, :]
        # burn-in sequence
        with torch.no_grad():
            _, h, c = self.policy(batch.bi_adj, batch.bi_state, batch.h_state, batch.c_state)
            _, h_q1, c_q1 = self.q1(batch.bi_adj, batch.bi_state)
            _, h_q2, c_q2 = self.q2(batch.bi_adj, batch.bi_state)
            _, h_q1_tar, c_q1_tar = self.q1_tar(batch.bi_adj, batch.bi_state)
            _, h_q2_tar, c_q2_tar = self.q2_tar(batch.bi_adj, batch.bi_state)

        action_probs, log_prob_pi = self.policy.pi_and_log_prob_pi(batch.adj_mat[:, Config.n_step:, :, :], batch.state[:, Config.n_step:, :, :], h, c)
        q1_value_tar, _, _ = self.q1_tar(batch.adj_mat, batch.state, h_q1_tar, c_q1_tar)  # [B, seq_in + n_step, N, 3]
        q2_value_tar, _, _ = self.q2_tar(batch.adj_mat, batch.state, h_q2_tar, c_q2_tar)

        q_next = torch.min(q1_value_tar[:, Config.n_step:, :, :], q2_value_tar[:, Config.n_step:, :, :])
        value = self.gamma ** Config.n_step * (action_probs * (q_next - self.call_alpha() * log_prob_pi)).sum(dim=3)
        q_tar = (batch.reward.squeeze(-1) + batch.done.squeeze(-1) * value.squeeze()).detach()
        index = (batch.action + 1).round().long()
        q1_val = torch.gather(self.q1(adj, state, h_q1, c_q1)[0], dim=3, index=index).squeeze()
        q2_val = torch.gather(self.q2(adj, state, h_q2, c_q2)[0], dim=3, index=index).squeeze()
        # print("q1_val:", q1_val.shape, "q_tar:", q_tar.shape, "q_next:", q_next.shape, "value", value.shape)
        return F.mse_loss(q1_val, q_tar) + F.mse_loss(q2_val, q_tar)

    def compute_policy_loss(self, batch: GraphBatch):
        # burn-in phase
        with torch.no_grad():
            _, h, c = self.policy(batch.bi_adj, batch.bi_state, batch.h_state, batch.c_state)
            _, h_q1, c_q1 = self.q1(batch.bi_adj, batch.bi_state)
            _, h_q2, c_q2 = self.q2(batch.bi_adj, batch.bi_state)

        adj, state = batch.adj_mat[:, :-Config.n_step, :, :], batch.state[:, :-Config.n_step, :, :]
        action_probs, log_prob_pi = self.policy.pi_and_log_prob_pi(adj, state, h, c)
        target_q1, target_q2 = self.q1(adj, state, h_q1, c_q1)[0], self.q2(adj, state, h_q2, c_q2)[0]

        H = - torch.sum(action_probs * log_prob_pi, dim=3).mean()
        q_mean = torch.sum(torch.min(target_q1, target_q2) * action_probs, dim=-1).mean()
        policy_loss = - q_mean - H

        return policy_loss, q_mean.item(), H.item()

    def compute_alpha_loss(self, batch: GraphBatch):
        with torch.no_grad():
            _, h, c = self.policy(batch.bi_adj, batch.bi_state, batch.h_state, batch.c_state)
        adj, state = batch.adj_mat[:, :-Config.n_step, :, :], batch.state[:, :-Config.n_step, :, :]
        action_probs, log_prob_pi = self.policy.pi_and_log_prob_pi(adj, state, h, c)
        entropy = - torch.sum(action_probs * log_prob_pi, dim=-1)
        loss = self.call_alpha() * (entropy - Config.target_entropy)
        return loss.mean()

    def q_value(self, adj_mat, state, action):
        q1, q2 = self.q1(adj_mat, state), self.q2(adj_mat, state)
        index = (action + 1).round().long()
        q1 = torch.gather(q1, dim=2, index=index)
        q2 = torch.gather(q2, dim=2, index=index)
        return q1.squeeze(), q2.squeeze()

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
