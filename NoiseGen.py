import torch


def generate_binary_noise(*dim, p=0.5, magnitude=1):
    random_tensor = torch.rand(*dim)
    return (random_tensor < p).float() * magnitude


def generate_gaussian_noise(*dim, mean=0, std=1):
    return torch.normal(mean=mean, std=std, size=dim)
