# config.py
import os
from datetime import datetime

# Determine the project root directory:
# config.py is located in the 'code' folder, so we move up one level
# to get the project root. (E:\CONV-TAS NET ENV)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data folder (it is assumed to be named 'data' and located at the project root)
# You can also allow the user to override DATA_ROOT via an environment variable
DATA_ROOT = os.environ.get('VCTK_DATA_ROOT', os.path.join(BASE_DIR, 'data', 'VCTK2mix_44k'))

current_date = datetime.now().strftime("%Y.%m.%d")
log_filename = f"train_log_{current_date}.txt"

CONFIG = {
    # Model's hyperparameters
    'sample_rate': 44100,
    'batch_size': 16,
    'n_src': 2,
    'n_filters': 256,
    'kernel_size': 88,
    'stride': 44,
    'bn_chan': 128,
    'hid_chan': 256,
    'n_blocks': 8,
    'n_repeats': 3,
    'epochs': 1,
    'lr': 1e-4,
    'max_norm': 5.0,
    'patience': 10, # Only for ReaduceLROnPlateu
    'segment': 2,
    'eval_batches': 10,   # число батчей для вычисления дополнительных метрик
    'metrics_interval': 10,
    # Augmentation
    'augment_prob': 0.2,               # вероятность применения
    'gain_range': (0.7, 1.3),
    'shift_range': 0.05,               # доля от seg_len

    # Paths to data and checkpoints (all relative)
    'checkpoint_dir': os.path.join(BASE_DIR, 'checkpoints'),
    'log_path': os.path.join(BASE_DIR, 'logs', log_filename),
    'train_csv': os.path.join(DATA_ROOT, 'metadata', 'mixture_train_mix_clean.csv'),
    'val_csv': os.path.join(DATA_ROOT, 'metadata', 'mixture_val_mix_clean.csv'),
    'data_root': None,   # we pass it to the dataset to construct full paths
}