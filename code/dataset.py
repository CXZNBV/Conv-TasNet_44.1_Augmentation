# dataset.py
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset
import csv
import os
import soundfile as sf
import numpy as np


class VCTK(Dataset):
    def __init__(self, csv_path, segment=3, sample_rate=48000,
                 data_root=None, preload=True, augment_prob=0.5, gain_range=(0.7, 1.3), 
                 time_stretch_range=(0.9, 1.1), speed_perturb_range=(0.9, 1.1)):
        self.sample_rate = sample_rate
        self.seg_len = int(segment * sample_rate)
        self.preload = preload
        self.data = []
        self.audio_cache = {}

        # Сохраняем параметры аугментации как атрибуты
        self.augment_prob = augment_prob
        self.gain_range = gain_range
        self.time_stretch_range = time_stretch_range
        self.speed_perturb_range = speed_perturb_range

        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                length = int(float(row['length']))
                if length >= self.seg_len:
                    if data_root is not None:
                        row['mixture_path'] = os.path.join(data_root, row['mixture_path'])
                        row['source_1_path'] = os.path.join(data_root, row['source_1_path'])
                        row['source_2_path'] = os.path.join(data_root, row['source_2_path'])
                    self.data.append(row)

        if self.preload:
            unique_paths = set()
            for row in self.data:
                unique_paths.add(row['mixture_path'])
                unique_paths.add(row['source_1_path'])
                unique_paths.add(row['source_2_path'])
            for path in unique_paths:
                audio, _ = sf.read(path, dtype='float32')
                self.audio_cache[path] = torch.from_numpy(audio)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        length = int(float(row['length']))
        start = torch.randint(0, length - self.seg_len + 1, (1,)).item()
        stop = start + self.seg_len

        if self.preload:
            mix = self.audio_cache[row['mixture_path']][start:stop]
            s1 = self.audio_cache[row['source_1_path']][start:stop]
            s2 = self.audio_cache[row['source_2_path']][start:stop]
        else:
            mix, _ = sf.read(row['mixture_path'], start=start, stop=stop, dtype='float32')
            s1, _ = sf.read(row['source_1_path'], start=start, stop=stop, dtype='float32')
            s2, _ = sf.read(row['source_2_path'], start=start, stop=stop, dtype='float32')
            mix = torch.from_numpy(mix)
            s1 = torch.from_numpy(s1)
            s2 = torch.from_numpy(s2)

	# Augment only during training (you can set self.training flag or use augment_prob)
        if self.augment_prob > 0 and torch.rand(1).item() < self.augment_prob:
    	    # 1. Gain
            gain_factor = torch.empty(1).uniform_(*self.gain_range).item()
            mix = mix * gain_factor
            s1 = s1 * gain_factor
            s2 = s2 * gain_factor
    
            # 3. Random shift (roll) — временной сдвиг
            shift = torch.randint(-int(self.seg_len * 0.05), int(self.seg_len * 0.05), (1,)).item()
            if shift != 0:
                mix = torch.roll(mix, shift, dims=-1)
                s1 = torch.roll(s1, shift, dims=-1)
                s2 = torch.roll(s2, shift, dims=-1)

        # --- Формируем выходные тензоры (уже аугментированные) ---
        mixture = mix
        sources = torch.stack([s1, s2])
        return mixture, sources