# generate_vctk2mix.py
"""
Generate VCTK-2mix dataset at 44.1 kHz.
Uses environment variables:
    VCTK_SOURCE_DIR  – path to the original VCTK corpus (required)
    VCTK_DATA_BASE   – root folder for datasets (default: ./data)
"""
import os
import random
import csv
import numpy as np
import soundfile as sf
from pathlib import Path

# Configuration via environment variables
VCTK_SOURCE = os.environ.get('VCTK_SOURCE_DIR')
if VCTK_SOURCE is None:
    print("Error: Please set VCTK_SOURCE_DIR")
    exit(1)

BASE_DIR = Path(__file__).parent.parent.parent
DATA_BASE = os.environ.get('VCTK_DATA_BASE', str(BASE_DIR / 'data'))
OUTPUT_DIR = os.path.join(DATA_BASE, 'VCTK2mix_44k')

print(f"Original VCTK: {VCTK_SOURCE}")
print(f"Dataset will be saved to: {OUTPUT_DIR}")

CONFIG = {
    'vctk_dir': VCTK_SOURCE,
    'output_dir': OUTPUT_DIR,
    'sample_rate': 44100,
    'mic_id': 'mic1',
    'snr_min': -2.5,
    'snr_max': 2.5,
    'n_train': 30000,
    'n_val': 6000,
    'n_test': 3000,
    'min_dur': 2.0,
    'val_ratio': 0.15,
    'skip': {'p315'},
}

# Core functions
def collect_speakers(vctk_dir, mic_id, min_dur, skip_speakers):
    audio_dir = Path(vctk_dir) / 'wav48_silence_trimmed'
    if not audio_dir.exists():
        raise FileNotFoundError(f"Folder not found: {audio_dir}")
    speakers = {}
    for spk_dir in audio_dir.iterdir():
        if not spk_dir.is_dir() or spk_dir.name in skip_speakers:
            continue
        files = [str(f) for f in spk_dir.glob(f'*_{mic_id}.flac')
                 if sf.info(str(f)).duration >= min_dur]
        if len(files) >= 5:
            speakers[spk_dir.name] = files
    return speakers

def mix_signals(path1, path2, snr_db):
    s1, _ = sf.read(path1, dtype='float32')
    s2, _ = sf.read(path2, dtype='float32')
    s1 = s1.mean(axis=1) if s1.ndim > 1 else s1
    s2 = s2.mean(axis=1) if s2.ndim > 1 else s2
    s1 /= np.max(np.abs(s1)) + 1e-8
    s2 /= np.max(np.abs(s2)) + 1e-8
    s2 /= 10 ** (snr_db / 20)
    min_len = min(len(s1), len(s2))
    s1, s2 = s1[:min_len], s2[:min_len]
    mix = s1 + s2
    peak = np.max(np.abs(mix)) + 1e-8
    return mix/peak, s1/peak, s2/peak

def generate_split(speakers, n_mix, output_dir, split_name, snr_min, snr_max, sample_rate):
    mix_dir, s1_dir, s2_dir = [Path(output_dir)/split_name/d for d in ['mix','s1','s2']]
    for d in [mix_dir, s1_dir, s2_dir]:
        d.mkdir(parents=True, exist_ok=True)
    meta_dir = Path(output_dir)/'metadata'
    meta_dir.mkdir(exist_ok=True)
    rows = []
    spk_list = list(speakers.keys())
    for i in range(n_mix):
        spk1, spk2 = random.sample(spk_list, 2)
        f1, f2 = random.choice(speakers[spk1]), random.choice(speakers[spk2])
        snr = random.uniform(snr_min, snr_max)
        mix, s1, s2 = mix_signals(f1, f2, snr)
        name = f"{spk1}_{Path(f1).stem}_{spk2}_{Path(f2).stem}.wav"
        sf.write(str(mix_dir/name), mix, sample_rate)
        sf.write(str(s1_dir/name), s1, sample_rate)
        sf.write(str(s2_dir/name), s2, sample_rate)
        rows.append({
            'mixture_path': str(mix_dir/name),
            'source_1_path': str(s1_dir/name),
            'source_2_path': str(s2_dir/name),
            'length': float(len(mix)),
            'speaker_1': spk1,
            'speaker_2': spk2,
            'snr_db': round(snr, 3),
        })
    csv_path = meta_dir / f'mixture_{split_name}_mix_clean.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"{split_name}: {len(rows)} mixtures → {csv_path}")

# ---- Main ----
if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    print("Generating VCTK-2mix (44.1 kHz)...")
    speakers = collect_speakers(CONFIG['vctk_dir'], CONFIG['mic_id'],
                                CONFIG['min_dur'], CONFIG['skip'])
    if len(speakers) < 10:
        print("❌ Too few speakers.")
        exit(1)
    spk_list = list(speakers.keys())
    random.shuffle(spk_list)
    n_val = max(3, int(len(spk_list) * CONFIG['val_ratio']))
    val_spks = {s: speakers[s] for s in spk_list[:n_val]}
    train_spks = {s: speakers[s] for s in spk_list[n_val:]}
    for split, n_mix, spks in [('train', CONFIG['n_train'], train_spks),
                               ('val', CONFIG['n_val'], val_spks),
                               ('test', CONFIG['n_test'], val_spks)]:
        generate_split(spks, n_mix, CONFIG['output_dir'], split,
                       CONFIG['snr_min'], CONFIG['snr_max'], CONFIG['sample_rate'])
    print("Done! 44.1 kHz dataset created.")
    print(f"Path: {CONFIG['output_dir']}")
    print("\nTo create the 16 kHz version, run:")
    print("  python resample_to_16k.py")