import torch
from torch import nn as nn
import torch.nn.functional as F
from config import Config


class GraphLinear(nn.Module):
    def __init__(self, in_size, out_size, activate=None, in_global_size=0, out_global_size=Config.global_size):
        super().__init__()
        self.activate = activate
        self.use_global = False if in_global_size == 0 else True
        self.global_size = in_global_size
        self.linear = nn.Linear(in_size * 3 + in_global_size, out_size)
        self.u_linear = nn.Linear(in_size * 3 + in_global_size, out_global_size)

    def forward(self, adj_mat, state, u, h, c):
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
            return vertex_feature, global_feature, h, c
        else:
            return self.activate(vertex_feature), self.activate(global_feature), h, c

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

    def forward(self, adj_mat, state, u, h, c):
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
            # 2-dimentional l_in
            l_in = torch.cat(x_in, dim=-1)
            features, (h_out, c_out) = self.lstm(l_in.unsqueeze(dim=0), (h, c))
            vertex_feature, global_feature = torch.split(features.squeeze(), [self.out_size, self.out_global_size], dim=-1)
            global_feature = global_feature.mean(dim=-2)
            # vertex_feature: [N, F] , global_feature: [N, G]
        else:
            # 4-dimentional l_in
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
            return self.activate(vertex_feature), self.activate(global_feature), self.activate(h_out), self.activate(c_out)

    def _preprocess_adj(self, adj):
        return F.normalize(adj, p=1, dim=-1)


class LSTMLayer(nn.Module):
    def __init__(self,
            in_size, out_size, activate=None):
        super().__init__()
        self.activate = activate
        self.lstm = nn.LSTM(in_size, out_size)

    def forward(self, adj_mat, x, u, h, c):
        shape = x.shape
        if len(shape) == 2:
            x, (h_out, c_out) = self.lstm(x.unsqueeze(dim=0), (h, c))
        else:
            x = torch.cat(x.split(1, dim=0), dim=2).squeeze()  # [S, B * N, F]
            x, (h_out, c_out) = self.lstm(x) if (h is None) else self.lstm(x, (h, c))
            x = torch.cat(x.unsqueeze(0).split(shape[-2], dim=-2), dim=0)

        if self.activate is None:
            return x, u, h_out, c_out
        else:
            return self.activate(x), u, self.activate(h_out), self.activate(c_out)


class LinearLayer(nn.Module):
    def __init__(self, in_size, out_size, activate=None):
        super().__init__()
        self.activate = None
        self.layer = nn.Linear(in_size, out_size)

    def forward(self, adj_mat, x, u, h, c):
        x = self.layer(x)
        if self.activate is None:
            return x, u, h, c
        else:
            return self.activate(x), u, h, c
