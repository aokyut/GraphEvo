import torch


def entropy(a):
    return (-a * a.log()).sum()


a = torch.tensor([0.05, 0.05, 0.9])

print(entropy(a))
