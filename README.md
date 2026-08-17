# Conv-TasNet Augmentation for 44.1 kHz

This repository contains the **training** code for a modified **Conv-TasNet** architecture adapted for speech separation
on the _VCTK-2mix_ dataset at 44.1 kHz.

## Setup

 Setup python venv in your directory.
   
```python
python -m venv venv  
venv/bin/activate
```
On Windows: 
```python
python -m venv venv
venv\Scripts\activate
```
 Install dependencies: `pip install -r requirements.txt`.

3. Prepare your dataset:
   a. **Download** original VCTK-2mix.
   b. Set the required environment variables (PowerShell example):
```python
$env:VCTK\_SOURCE\_DIR = "D:\Dataset_path\VCTK-2mix"   # VCTK path.
$env:VCTK\_DATA\_BASE = "E:\Conv-Tas_Net_Aug_path\data"    # Base directory for generated datasets.
```

4. Generate the 44.1 kHz dataset. This creates VCTK2mix\_44k with 30,000 training mixtures (and 6k/3k val/test):
Run the first script:
 ```python
python generate_vctk2mix.py
```

6. Generate the 16 kHz version (optional, for two-stage training). This creates VCTK2mix\_16k/ from the 44.1 kHz dataset:
Run the second script:
```python
python resample_to_16k.py
```

## Training

1. Configurate model:
a. Open `config.py` in IDE.
b. You can adjust the following parameters as needed: _num_ of _epochs_, _learning rate_, _model size_ (bn\_chan; n\_filters; hid\_chan),
_kernel param_ (kernel\_size; stride), _segment_ or _sample rate_.
**VRAM/RAM WARNING**: The default configuration uses ~5 GB of VRAM and 40–60 GB of RAM (due to dataset preloading). Adjust batch_size or 'preload=False' in 'train.py' if memory is limited.

2. Run training file: python train.py
The best model will be saved in the checkpoint directory (checkpoints/).

## Separating

To separate a single audio file into two sources. The script will load the best model from checkpoints/best\_model.pth and save two files. 
Run separating files:
```python
python separate_audio.py --input "path/to/mixture.wav" --output "path/to/output_dir" --model "path/to/best_model.pth"
```

Example: 
```python
python separate_audio.py --input "E:\CONV-TAS NET ENV\test\test_mix.wav" --output "E:\CONV-TAS NET ENV\test" --model "E:\CONV-TAS NET ENV\checkpoints\best_model.pth"
```

## License

This project is licensed under the MIT License — see the LICENSE.txt for details.

## Citation

@article{luo2019conv,
    title={Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation},
    author={Luo, Yi and Mesgarani, Nima},
    journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
    year={2019}
}

Additionally, if you use our modifications or the VCTK-2mix dataset adaptation, please consider citing this repository:


@misc{conv-tasnet-44k,
    author = {Tyurin, Denis},
    title = {Conv-TasNet Augmentation for 44.1 kHz Speech Separation},
    year = {2026},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/CXZNBV/Conv-TasNet_44.1_Augmentation}}
}
