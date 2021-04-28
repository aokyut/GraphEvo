# import torch
from utils import Writer
from network import GraphQ
from agent import GraphBatch
from config import Config
from typing import List
import torch.optim as optim
import torch
# import torch.nn.functional as F


def train(model: GraphQ,
          batches: List[GraphBatch],
          writer: Writer,
          value_optim: optim.Optimizer):
    value_loss = None

    # with torch.autograd.detect_anomaly():
    for batch in batches:
        l: torch.Tensor = train_value(model, batch)
        if value_loss is None:
            value_loss = l
        else:
            value_loss = l + value_loss
    value_loss = value_loss / Config.bundle_size
    value_optim.zero_grad()
    value_loss.backward()
    value_optim.step()

    # torch.autograd.set_detect_anomaly(False)

    model.update_target(Config.rho)

    writer.log_train("loss/q_function_loss", value_loss)

    writer.add_train_step()
    writer.save(model)

    return value_loss


def train_value(model: GraphQ,
          batch: GraphBatch):
    q_tar = model.compute_q_target(batch.adj_mat, batch.next_state, Config.gamma, batch.reward, batch.done)

    q1_val, q2_val = model.q_value(batch.adj_mat, batch.state, batch.action)

    q_loss = 0.5 * (q_tar - q1_val).pow(2).mean() + 0.5 * (q_tar - q2_val).pow(2).mean()
    # q_loss = F.smooth_l1_loss(q_tar, q1_val).mean() + F.smooth_l1_loss(q_tar, q2_val).mean()

    value_loss = q_loss

    return value_loss
