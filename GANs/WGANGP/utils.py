import torch
import torch.nn as nn


def gradient_penalty(critic, labels, real, fake, device='cpu'):
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * epsilon + fake * (1 - epsilon)

    mixed_score = critic(interpolated_images, labels)

    # mixed_score = func(interpolated_images)
    # Computes and returns the sum of gradients of outputs with respect to the inputs.
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_score,
        grad_outputs=torch.ones_like(mixed_score),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gp = torch.mean((gradient_norm - 1) ** 2)
    return gp
