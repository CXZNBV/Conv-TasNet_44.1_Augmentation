# metrics.py

import numpy as np
import mir_eval
import torch


def compute_bss_metrics(estimates, targets, permute=True):
    """
    Compute SDR, SIR, SAR with PIT.
    """
    if hasattr(estimates, 'cpu'):
        est = estimates.cpu().numpy()
        tgt = targets.cpu().numpy()
    else:
        est = np.array(estimates)
        tgt = np.array(targets)

    if est.ndim == 2:
        est = est[np.newaxis, ...]
        tgt = tgt[np.newaxis, ...]

    batch_size = est.shape[0]
    all_sdr, all_sir, all_sar = [], [], []

    for i in range(batch_size):
        ref = tgt[i]
        est_i = est[i]

        if permute:
            # Permute 0
            sdr0, sir0, sar0, _ = mir_eval.separation.bss_eval_sources(ref, est_i)
            # Permute 1. FUNCTION mir_eval.separation.bss_eval_sources WORK ONLY WITH mir_eval <0.9
            est_perm = np.array([est_i[1], est_i[0]])
            sdr1, sir1, sar1, _ = mir_eval.separation.bss_eval_sources(ref, est_perm)

            if np.mean(sdr1) > np.mean(sdr0):
                sdr, sir, sar = sdr1, sir1, sar1
            else:
                sdr, sir, sar = sdr0, sir0, sar0
        else:
            sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(ref, est_i)

        all_sdr.extend(sdr)
        all_sir.extend(sir)
        all_sar.extend(sar)

    return {
        'SDR': np.mean(all_sdr),
        'SIR': np.mean(all_sir),
        'SAR': np.mean(all_sar)
    }