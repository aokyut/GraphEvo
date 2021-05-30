import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from agent import GraphBatch
from layer import GraphLinear, LSTMLayer

EPS = 1e-8


class GraphPolicy(nn.Module):
    def __init__(self,
            in_size=Config.network_in_size,
            out_size=Config.network_out_size):
        super().__init__()

        layers = [
            LSTMLayer(in_size=in_size, out_size=64, activate=nn.LeakyReLU()),
            GraphLinear(in_size=64, out_size=32, activate=nn.LeakyReLU()),
            GraphLinear(in_size=32, out_size=16, activate=nn.LeakyReLU(), in_global_size=Config.global_size),
            GraphLinear(in_size=16, out_size=8, activate=nn.LeakyReLU(), in_global_size=Config.global_size),
            GraphLinear(in_size=8, out_size=3, in_global_size=Config.global_size)
        ]

        self.pre_network = nn.ModuleList(layers)

    def forward(self, adj_mat, x, h=None, c=None):
        # adj_mat = adj_mat.squeeze()
        # x = x.squeeze()
        u = None
        for layer in self.pre_network:
            x, u, h, c = layer(adj_mat, x, u, h, c)

        action_probs = F.softmax(2 * x.tanh(), dim=-1)
        shape = torch.Size(action_probs.shape)
        actions = torch.multinomial(action_probs.reshape(-1, shape[-1]), 1, True).reshape(shape[:-1])

        return actions.float() - 1, h, c

    def pi_and_log_prob_pi(self, adj_mat, x, h=None, c=None):

        u = None
        for layer in self.pre_network:
            x, u, h, c = layer(adj_mat, x, u, h, c)

        action_probs = F.softmax(2 * x.tanh(), dim=-1)

        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)
        return action_probs, log_action_probs


class GraphQNetwork(nn.Module):
    def __init__(self, in_size=Config.network_in_size,
                       out_size=Config.network_out_size):
        super().__init__()
        layers = [
            LSTMLayer(in_size=in_size, out_size=64, activate=nn.LeakyReLU()),
            GraphLinear(in_size=64, out_size=32, activate=nn.LeakyReLU()),
            GraphLinear(in_size=32, out_size=16, activate=nn.LeakyReLU(), in_global_size=Config.global_size),
            GraphLinear(in_size=16, out_size=8, activate=nn.LeakyReLU(), in_global_size=Config.global_size),
            GraphLinear(in_size=8, out_size=3, in_global_size=Config.global_size)
        ]

        self.network = nn.ModuleList(layers)

    def forward(self, adj_mat, x, h=None, c=None):
        u = None
        for layer in self.network:
            x, u, h, c = layer(adj_mat, x, u, h, c)
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
        q_tar = self.rescaling(batch.reward.squeeze(-1) + self.rescaling_inverse(batch.done.squeeze(-1) * value.squeeze())).squeeze().detach()
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
        return (1 + self.alpha / (torch.abs(self.alpha) + 1)) / 2

    def rescaling(self, x, epsilon=EPS):
        n = torch.sqrt(abs(x) + 1) - 1
        return torch.sign(x) * n + epsilon * x

    def rescaling_inverse(self, x):
        return torch.sign(x) * ((x + torch.sign(x)) ** 2 - 1)
