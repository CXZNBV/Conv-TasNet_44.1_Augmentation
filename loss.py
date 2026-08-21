# loss.py
import torch

def pit_si_snr_loss(est, target):
    # est, target: [B, 2, T]
    eps = 1e-8

    def si_snr(s, t):
        t = t - t.mean(-1, keepdim=True)
        s = s - s.mean(-1, keepdim=True)
        dot = (s * t).sum(-1, keepdim=True)
        t_norm = (t * t).sum(-1, keepdim=True) + eps
        t_proj = dot * t / t_norm
        noise = s - t_proj
        return 10 * torch.log10(
            (t_proj**2).sum(-1) / ((noise**2).sum(-1) + eps) + eps
        )

    loss_perm1 = -(si_snr(est[:, 0], target[:, 0]) +
                   si_snr(est[:, 1], target[:, 1])) / 2
    loss_perm2 = -(si_snr(est[:, 0], target[:, 1]) +
                   si_snr(est[:, 1], target[:, 0])) / 2
    loss = torch.minimum(loss_perm1, loss_perm2)
    return loss.mean()