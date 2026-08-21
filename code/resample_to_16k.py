# resample_to_16k.py
"""
Resample the dataset from 44.1 kHz to 16 kHz.
Automatically finds the 44.1 kHz dataset in VCTK_DATA_BASE/VCTK2mix_44k
and creates a copy in VCTK_DATA_BASE/VCTK2mix_16k.
"""
import os
import csv
import torch
import torchaudio
import soundfile as sf
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---- Configuration via environment variables ----
BASE_DIR = Path(__file__).parent.parent
DATA_BASE = os.environ.get('VCTK_DATA_BASE', str(BASE_DIR / 'data'))

SRC_ROOT = os.environ.get('VCTK_44K_DIR', os.path.join(DATA_BASE, 'VCTK2mix_44k'))
DST_ROOT = os.environ.get('VCTK_16K_DIR', os.path.join(DATA_BASE, 'VCTK2mix_16k'))
TARGET_SR = 16000

print(f"Source dataset (44.1 kHz): {SRC_ROOT}")
print(f"Target dataset (16 kHz): {DST_ROOT}")

if not os.path.isdir(SRC_ROOT):
    print(f"Error: folder {SRC_ROOT} not found.")
    print("Please run generate_vctk2mix.py first to create the 44.1 kHz dataset.")
    exit(1)

SUBFOLDERS = ['train', 'test', 'val']
METADATA_SRC = os.path.join(SRC_ROOT, 'metadata')
METADATA_DST = os.path.join(DST_ROOT, 'metadata')

# ---- Resampling function ----
def resample_file(src_path, dst_path, target_sr=TARGET_SR):
    try:
        audio, orig_sr = sf.read(src_path, dtype='float32', always_2d=False)
        if audio.ndim == 1:
            waveform = torch.from_numpy(audio).unsqueeze(0)
        else:
            waveform = torch.from_numpy(audio.T)
        if orig_sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
            waveform = resampler(waveform)
        if waveform.shape[0] == 1:
            audio_resampled = waveform.squeeze(0).cpu().numpy()
        else:
            audio_resampled = waveform.cpu().numpy().T
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        sf.write(dst_path, audio_resampled, target_sr)
        new_length = waveform.shape[-1]
        return src_path, dst_path, new_length
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return None

# ---- Process all WAV files ----
def process_all_wavs():
    tasks = []
    for sub in SUBFOLDERS:
        src_dir = os.path.join(SRC_ROOT, sub)
        dst_dir = os.path.join(DST_ROOT, sub)
        if not os.path.isdir(src_dir):
            print(f"Folder {src_dir} not found, skipping.")
            continue
        for root, _, files in os.walk(src_dir):
            for file in files:
                if file.lower().endswith('.wav'):
                    src_path = os.path.join(root, file)
                    rel_path = os.path.relpath(src_path, src_dir)
                    dst_path = os.path.join(dst_dir, rel_path)
                    tasks.append((src_path, dst_path))
    print(f"Found {len(tasks)} WAV files.")
    if not tasks:
        return {}
    results = {}
    processed = 0
    total = len(tasks)
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_src = {executor.submit(resample_file, src, dst): src for src, dst in tasks}
        for future in as_completed(future_to_src):
            res = future.result()
            processed += 1
            if processed % 1000 == 0:
                print(f"Processed {processed}/{total} files.")
            if res is not None:
                src_path, dst_path, new_len = res
                results[src_path] = (dst_path, new_len)
    print(f"Successfully processed {len(results)} files.")
    return results

# ---- Update CSV files ----
def update_csv_files(length_mapping):
    os.makedirs(METADATA_DST, exist_ok=True)
    csv_files = [f for f in os.listdir(METADATA_SRC) if f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found in metadata folder.")
        return
    for csv_file in csv_files:
        src_csv = os.path.join(METADATA_SRC, csv_file)
        dst_csv = os.path.join(METADATA_DST, csv_file)
        print(f"Processing CSV: {csv_file}")
        with open(src_csv, 'r', newline='', encoding='utf-8') as fin:
            reader = csv.DictReader(fin)
            fieldnames = reader.fieldnames
            rows = list(reader)
        new_rows = []
        for row in rows:
            for col in ['mixture_path', 'source_1_path', 'source_2_path']:
                old_path = row[col]
                if old_path.startswith(SRC_ROOT):
                    row[col] = old_path.replace(SRC_ROOT, DST_ROOT)
                else:
                    row[col] = os.path.join(DST_ROOT, old_path.lstrip('./\\'))
            old_mix_path = row['mixture_path'].replace(DST_ROOT, SRC_ROOT)
            if old_mix_path in length_mapping:
                _, new_len = length_mapping[old_mix_path]
                row['length'] = str(new_len)
            else:
                # fallback using source_1_path
                old_s1_path = row['source_1_path'].replace(DST_ROOT, SRC_ROOT)
                if old_s1_path in length_mapping:
                    _, new_len = length_mapping[old_s1_path]
                    row['length'] = str(new_len)
                else:
                    print(f"Length not found for {old_mix_path}")
            new_rows.append(row)
        with open(dst_csv, 'w', newline='', encoding='utf-8') as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(new_rows)
        print(f"Saved: {dst_csv}")

# Main
if __name__ == '__main__':
    length_mapping = process_all_wavs()
    if not length_mapping:
        print("Failed to process any WAV files.")
        exit(1)
    update_csv_files(length_mapping)
    print("Done! 16 kHz dataset created.")
    print(f"Path: {DST_ROOT}")