# import torch
from utils import Writer
from network import GraphSAC
from agent import GraphBatch
from config import Config
from typing import List
import torch.optim as optim
# import torch
# import torch.nn.functional as F


def train(model: GraphSAC,
          batches: List[GraphBatch],
          writer: Writer,
          value_optim: optim.Optimizer,
          policy_optim: optim.Optimizer,
          alpha_optim: optim.Optimizer):
    value_loss = None
    v_loss = 0
    q_loss = 0

    # with torch.autograd.detect_anomaly():
    for batch in batches:
        l, q, v = train_value(model, batch)
        v_loss = v + v_loss
        q_loss = q + q_loss
        if value_loss is None:
            value_loss = l
        else:
            value_loss = l + value_loss
    value_loss = value_loss / Config.bundle_size
    v_loss /= Config.bundle_size
    q_loss /= Config.bundle_size
    value_optim.zero_grad()
    value_loss.backward()
    value_optim.step()

    policy_loss = None
    q_mean = 0
    H_mean = 0
    for batch in batches:
        p, q, h = train_policy(model, batch)
        q_mean = q + q_mean
        H_mean = h + H_mean
        if policy_loss is None:
            policy_loss = p
        else:
            policy_loss = p + policy_loss

    policy_loss = policy_loss / Config.bundle_size
    q_mean /= Config.bundle_size
    H_mean /= Config.bundle_size
    policy_optim.zero_grad()
    policy_loss.backward()
    policy_optim.step()

    alpha_loss = None
    for batch in batches:
        if alpha_loss is None:
            alpha_loss = tuning_alpha(model, batch)
        else:
            alpha_loss = tuning_alpha(model, batch) + alpha_loss

    alpha_loss = alpha_loss / Config.bundle_size
    alpha_optim.zero_grad()
    alpha_loss.backward()
    alpha_optim.step()

    # torch.autograd.set_detect_anomaly(False)

    model.update_target(Config.rho)

    writer.log_train("loss/q_function_loss", q_loss)
    # writer.log_train("loss/v_function_loss", v_loss)
    writer.log_train("loss/q_value_mean", q_mean)
    writer.log_train("loss/H_mean", H_mean)
    # writer.log_train("loss/value_loss", value_loss.item())
    writer.log_train("loss/policy_loss", policy_loss.item())
    writer.log_train("loss/alpha_loss", alpha_loss.item())
    writer.log_train("loss/alpha", model.call_alpha())

    writer.add_train_step()
    writer.save(model)

    return policy_loss, value_loss


def train_value(model: GraphSAC,
          batch: GraphBatch):
    q_tar = model.compute_q_target(batch.adj_mat, batch.next_state, Config.gamma, batch.reward, batch.done)

    q1_val, q2_val = model.q_value(batch.adj_mat, batch.state, batch.action)

    q_loss = 0.5 * (q_tar - q1_val).pow(2).mean() + 0.5 * (q_tar - q2_val).pow(2).mean()
    # q_loss = F.smooth_l1_loss(q_tar, q1_val).mean() + F.smooth_l1_loss(q_tar, q2_val).mean()

    value_loss = q_loss

    return value_loss, q_loss.item(), 0


def train_policy(model: GraphSAC,
                 batch: GraphBatch):
    q, H = model.q_function_entropy(batch.adj_mat, batch.state)
    policy_loss = - q - H * model.call_alpha()
    return policy_loss.mean(), q.mean().item(), H.mean().item()


def tuning_alpha(model: GraphSAC,
                 batch: GraphBatch):
    return model.compute_alpha_target(batch.adj_mat, batch.state).mean()
