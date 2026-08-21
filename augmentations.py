# augmentations.py
import torch
import numpy as np

def mixup_batch(mix, src, alpha=0.2):
    """
    Apply mixup augmentation to a batch.

    Args:
        mix: torch.Tensor of shape [batch_size, time]
        src: torch.Tensor of shape [batch_size, n_src, time]
        alpha: float, parameter for Beta distribution

    Returns:
        mixed_mix, mixed_src: augmented batch
    """
    batch_size = mix.size(0)
    perm = torch.randperm(batch_size)
    lam = np.random.beta(alpha, alpha)
    mixed_mix = lam * mix + (1 - lam) * mix[perm]
    mixed_src = lam * src + (1 - lam) * src[perm]
    return mixed_mix, mixed_src

# Здесь можно добавить другие аугментации, например:
# def gain_augment(mix, src, gain_range=(0.7, 1.3)): ...
# def time_shift_augment(mix, src, shift_range=0.05): ...