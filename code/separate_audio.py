# separate.py
"""
Speech separation script using a trained Conv-TasNet model.
Loads the best model from a checkpoint and processes a single audio file.
If ground-truth sources are provided, computes and displays objective metrics.
"""

import os
import argparse
import torch
import torchaudio
import soundfile as sf
import numpy as np
from model import ConvTasNet
from metrics import compute_bss_metrics
from loss import pit_si_snr_loss


def load_model(model_path, device):
    """Load model checkpoint and return model, device, and sample rate."""
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    cfg = checkpoint['config']

    model = ConvTasNet(
        N=cfg['n_filters'],
        L=cfg['kernel_size'],
        B=cfg['bn_chan'],
        H=cfg['hid_chan'],
        P=3,
        X=cfg['n_blocks'],
        R=cfg['n_repeats'],
        C=cfg['n_src'],
        stride=cfg['stride'],
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded | Best SI-SNRi: {checkpoint['si_snri']:.2f} dB")
    print(f"Config: sample_rate={cfg['sample_rate']} Hz, segment={cfg['segment']} s")
    return model, device, cfg['sample_rate']


def prepare_audio(file_path, target_sr):
    """Load audio, convert to mono, resample if needed, and return waveform tensor."""
    data, sr = sf.read(file_path, dtype='float32', always_2d=True)
    waveform = torch.from_numpy(data.T)
    print(f"Input file: {os.path.basename(file_path)}")
    print(f"  Sample rate: {sr} Hz | Channels: {waveform.shape[0]} | "
          f"Duration: {waveform.shape[1]/sr:.1f} s")

    if sr != target_sr:
        print(f"  Resampling {sr} → {target_sr} Hz...")
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform.squeeze(0)  # [T]


def separate(model, waveform, device):
    """Run inference and return separated sources."""
    mixture = waveform.unsqueeze(0).to(device)  # [1, T]
    with torch.no_grad():
        sources = model(mixture)                # [1, 2, T]
    return sources.squeeze(0).cpu()             # [2, T]


def save_results(sources, output_dir, input_filename, sample_rate, normalize=True):
    """Save separated sources to WAV files."""
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_filename))[0]

    for i, source in enumerate(sources):
        if normalize:
            source = source / (source.abs().max() + 1e-8)
        out_path = os.path.join(output_dir, f"{base}_speaker_{i+1}.wav")
        sf.write(out_path, source.numpy(), sample_rate)
        print(f"  Saved: {out_path}")


def load_reference(ref_path, target_sr):
    """Load and prepare a reference (clean) source."""
    if ref_path is None:
        return None
    waveform = prepare_audio(ref_path, target_sr)
    return waveform


def compute_metrics(est_sources, ref_sources, sample_rate):
    """
    Compute SI-SNR, SDR, SIR, SAR for two sources.
    est_sources: [2, T], ref_sources: [2, T]
    Returns a dictionary with metric values.
    """
    # Ensure same length (trim to minimum)
    min_len = min(est_sources.shape[-1], ref_sources.shape[-1])
    est_sources = est_sources[:, :min_len]
    ref_sources = ref_sources[:, :min_len]

    # Add batch dimension: [1, 2, T]
    est_batch = est_sources.unsqueeze(0)
    ref_batch = ref_sources.unsqueeze(0)

    # 1. SI-SNR using PIT loss (negative SI-SNR, so we take -loss)
    loss = pit_si_snr_loss(est_batch, ref_batch)   # returns negative SI-SNR average
    si_snr = -loss.item()   # positive SI-SNR

    # 2. BSS metrics (SDR, SIR, SAR)
    bss = compute_bss_metrics(est_batch, ref_batch, permute=True)
    sdr = bss['SDR']
    sir = bss['SIR']
    sar = bss['SAR']

    return {
        'SI-SNR': si_snr,
        'SDR': sdr,
        'SIR': sir,
        'SAR': sar,
    }


def main():
    parser = argparse.ArgumentParser(description="Separate speech sources using Conv-TasNet")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to input audio file (mixture)")
    parser.add_argument("--output", "-o", type=str, default="./results/separated",
                        help="Output directory for separated files (default: ./results/separated)")
    parser.add_argument("--model", "-m", type=str, default="./checkpoints/best_model.pth",
                        help="Path to model checkpoint (default: ./checkpoints/best_model.pth)")
    parser.add_argument("--ref1", type=str, default=None,
                        help="Path to ground-truth source 1 (optional, for metrics)")
    parser.add_argument("--ref2", type=str, default=None,
                        help="Path to ground-truth source 2 (optional, for metrics)")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Disable amplitude normalization before saving")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device.upper()}")

    # Load model
    model, device, target_sr = load_model(args.model, device)

    # Prepare mixture
    waveform = prepare_audio(args.input, target_sr)

    # Separate
    print("Separating...")
    sources = separate(model, waveform, device)
    print(f"  Obtained {sources.shape[0]} sources")

    # Save results
    print("Saving results...")
    save_results(sources, args.output, args.input, target_sr, normalize=not args.no_normalize)

    # Compute and display metrics if references are provided
    if args.ref1 is not None and args.ref2 is not None:
        print("\n--- Computing objective metrics ---")
        # Load references
        ref1 = load_reference(args.ref1, target_sr)
        ref2 = load_reference(args.ref2, target_sr)
        if ref1 is None or ref2 is None:
            print("Warning: Could not load one or both reference files. Skipping metrics.")
        else:
            # Stack references: [2, T]
            refs = torch.stack([ref1, ref2], dim=0)
            # Ensure sources and refs have same length (trim to min)
            min_len = min(sources.shape[-1], refs.shape[-1])
            sources_trim = sources[:, :min_len]
            refs_trim = refs[:, :min_len]

            metrics = compute_metrics(sources_trim, refs_trim, target_sr)
            print(f"SI-SNR: {metrics['SI-SNR']:.2f} dB")
            print(f"SDR:    {metrics['SDR']:.2f} dB")
            print(f"SIR:    {metrics['SIR']:.2f} dB")
            print(f"SAR:    {metrics['SAR']:.2f} dB")
    else:
        print("\nNo reference sources provided. Skipping metric computation.")

    print("Done!")


if __name__ == "__main__":
    main()