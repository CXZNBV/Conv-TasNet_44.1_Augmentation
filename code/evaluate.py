# evaluate.py

import torch
from metrics import compute_bss_metrics

def compute_and_format_metrics(collected_ests, collected_targets, sample_rate, metrics_interval, epoch):
    """
    Compute SDR, SIR, SAR and return formatted string.
    Only computes if epoch % metrics_interval == 0.
    """
    if epoch % metrics_interval == 0 and collected_ests:
        est_all = torch.cat(collected_ests, dim=0)
        tgt_all = torch.cat(collected_targets, dim=0)

        # BSS metrics
        metrics = compute_bss_metrics(est_all, tgt_all, permute=True)
        metrics_str = (
            f"SDR: {metrics['SDR']:.2f} dB | "
            f"SIR: {metrics['SIR']:.2f} dB | "
            f"SAR: {metrics['SAR']:.2f} dB"
        )
        return metrics_str
    return ""