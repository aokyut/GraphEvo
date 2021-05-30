import torch


def entropy(a):
    return (-a * a.log()).sum()


a = torch.tensor([0.05, 0.05, 0.9])

print(entropy(a))


def test_func(a, b, **c):
    return (a, b) + tuple(c.values())


print(test_func(a=1, c=2, b=3, d=4))
