# import torch
from utils import Writer
from network import GraphSAC
from agent import GraphBatch
from config import Config
from typing import List
import torch.optim as optim
import torch
# import torch.nn.functional as F


def train(model: GraphSAC,
          batches: List[GraphBatch],
          writer: Writer,
          value_optim: optim.Optimizer,
          policy_optim: optim.Optimizer,
          alpha_optim: optim.Optimizer):
    value_loss = None

    # with torch.autograd.detect_anomaly():
    for batch in batches:
        l: torch.Tensor = model.compute_q_loss(batch)
        if value_loss is None:
            value_loss = l
        else:
            value_loss = l + value_loss
    value_loss = value_loss / Config.bundle_size
    value_optim.zero_grad()
    value_loss.backward()
    value_optim.step()

    policy_loss = None
    q_mean = 0
    H_mean = 0
    for batch in batches:
        p, q, h = model.compute_policy_loss(batch)
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
            alpha_loss = model.compute_alpha_loss(batch)
        else:
            alpha_loss = model.compute_alpha_loss(batch) + alpha_loss

    alpha_loss = alpha_loss / Config.bundle_size
    alpha_optim.zero_grad()
    alpha_loss.backward()
    alpha_optim.step()

    # torch.autograd.set_detect_anomaly(False)

    model.update_target(Config.rho)

    writer.log_train("loss/q_function_loss", value_loss.item())
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
