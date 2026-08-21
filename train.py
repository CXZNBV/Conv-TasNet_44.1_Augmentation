# train.py
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import torch
import torch.nn as nn
import torchaudio
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
import os
import contextlib
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="mir_eval")


from config import CONFIG
from dataset import VCTK
from model import ConvTasNet
from loss import pit_si_snr_loss
from metrics import compute_bss_metrics
from validate import validate_epoch
from evaluate import compute_and_format_metrics
from augmentations import mixup_batch

# fix random seed
SEED = 711  # or any else

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # for all GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # for determinism (may slow down)
    # torch.backends.cudnn.benchmark = True

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE.upper()}")
    print(f"Seed: {SEED}")

    train_csv = CONFIG['train_csv']
    val_csv = CONFIG['val_csv']
    data_root = CONFIG.get('data_root', None)
    eval_batches = CONFIG.get('eval_batches', 10)   # from CONFIG

    print(f"Train CSV: {train_csv}")
    print(f"Val CSV: {val_csv}")

    train_set = VCTK(train_csv, segment=CONFIG['segment'], sample_rate=CONFIG['sample_rate'],
                     data_root=data_root, preload=True, augment_prob=CONFIG.get('augment_prob', 0.5),
                     gain_range=CONFIG.get('gain_range', (0.7, 1.3)), time_stretch_range=CONFIG.get('time_stretch_range', (0.9, 1.1)),
                     speed_perturb_range=CONFIG.get('speed_perturb_range', (0.9, 1.1)))
    val_set = VCTK(val_csv, segment=CONFIG['segment'], sample_rate=CONFIG['sample_rate'],
                   data_root=data_root, preload=True, augment_prob=0)

    train_loader = DataLoader(train_set, batch_size=CONFIG['batch_size'], shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=CONFIG['batch_size'], shuffle=False,
                            num_workers=0, pin_memory=True)

    model = ConvTasNet(
        N=CONFIG['n_filters'], L=CONFIG['kernel_size'], B=CONFIG['bn_chan'],
        H=CONFIG['hid_chan'], P=3, X=CONFIG['n_blocks'], R=CONFIG['n_repeats'],
        C=CONFIG['n_src'], stride=CONFIG['stride']
    ).to(DEVICE)

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {params:.1f}M")

    optimizer = Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=CONFIG['patience'], factor=0.5)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-5) # deterioration of Si-SNRi Δ 1.5 dB

    scaler = torch.amp.GradScaler('cuda') if DEVICE == 'cuda' else None

    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG['log_path']), exist_ok=True)

   # ---- Load checkpoint ----
    checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], 'best_model_26.07.2026.pth')
    start_epoch = 1
    best_val = float('inf')

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val = checkpoint.get('best_val', float('inf'))
        print(f"Resuming from epoch {start_epoch}, best_val = {best_val:.4f}")
    else:
         print("No checkpoint found.")

    def log(msg):
        print(msg)
        with open(CONFIG['log_path'], 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    def save_ckpt(epoch, metric, filename):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'si_snri': metric,
            'config': CONFIG,
            'seed': SEED,
            'best_val': best_val,
        }, os.path.join(CONFIG['checkpoint_dir'], filename))

    best_val = float('inf')

    for epoch in range(1, CONFIG['epochs'] + 1):

        # --- TRAIN ---

        model.train()
        t_loss = 0.0
        for mix, src in train_loader:
            mix = mix.to(DEVICE)
            src = src.to(DEVICE)

            # --- Apply mixup with probability 0.5 ---
            if np.random.rand() < 0.5:
                mix, src = mixup_batch(mix, src, alpha=0.2)

            optimizer.zero_grad()

            autocast_ctx = torch.amp.autocast('cuda') if DEVICE == 'cuda' else contextlib.nullcontext()
            with autocast_ctx:
                est = model(mix)
                loss = pit_si_snr_loss(est, src)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_norm'])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_norm'])
                optimizer.step()

            t_loss += loss.item()
        t_loss /= len(train_loader)

        # --- Validation ---
        v_loss, baseline_snr, si_snri, collected_ests, collected_targets = validate_epoch(
            model, val_loader, DEVICE, eval_batches, CONFIG['sample_rate'], pit_si_snr_loss
        )

        # --- Evaluate ---
        metrics_str = compute_and_format_metrics(
            collected_ests,
            collected_targets,
            CONFIG['sample_rate'],
            CONFIG['metrics_interval'],
            epoch
        )

	# --- Logging ---
        log_msg = (f"Epoch {epoch:03d}/{CONFIG['epochs']} | "
                   f"Train: {t_loss:.4f} | "
                   f"Val SI-SNR: {-v_loss:.2f} dB | "
                   f"SI-SNRi: {si_snri:.2f} dB")
        if metrics_str:
            log_msg += " | " + metrics_str
        log(log_msg)

        scheduler.step(v_loss)
        save_ckpt(epoch, si_snri, 'last_model.pth')

        if v_loss < best_val:
            best_val = v_loss
            save_ckpt(epoch, si_snri, 'best_model.pth')
            log(f"  Best model (SI-SNRi: {si_snri:.2f} dB)")

        if epoch % CONFIG['metrics_interval'] == 0:
            save_ckpt(epoch, si_snri, f'epoch_{epoch:03d}.pth')

    log("Training completed!")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()