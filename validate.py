# validate.py

import torch
import contextlib

def validate_epoch(model, val_loader, device, eval_batches, sample_rate, pit_si_snr_loss):
    """
    Perform one validation epoch.
    Returns:
        v_loss: average loss
        baseline_snr: average SI-SNR of mixture
        si_snri: improvement in SI-SNR
        collected_ests: list of estimated sources (cpu tensors)
        collected_targets: list of target sources (cpu tensors)
    """
    model.eval()
    v_loss = 0.0
    baseline_snr = 0.0
    collected_ests = []
    collected_targets = []

    with torch.no_grad():
        for batch_idx, (mix, src) in enumerate(val_loader):
            mix = mix.to(device)
            src = src.to(device)

            with torch.amp.autocast('cuda') if device == 'cuda' else contextlib.nullcontext():
                est = model(mix)
                loss = pit_si_snr_loss(est, src)
                mix_expanded = mix.unsqueeze(1).expand_as(src)
                base = pit_si_snr_loss(mix_expanded, src)
                baseline_snr += base.item()

            v_loss += loss.item()

            if batch_idx < eval_batches:
                collected_ests.append(est.cpu())
                collected_targets.append(src.cpu())

    v_loss /= len(val_loader)
    baseline_snr /= len(val_loader)
    si_snri = (-v_loss) - (-baseline_snr)

    return v_loss, baseline_snr, si_snri, collected_ests, collected_targets