<h1 align="center">Conv-TasNet Augmentation for 44.1 kHz</h1>


This repository contains the **training code** for a modified **Conv-TasNet** architecture adapted for speech separation on the _VCTK-2mix_ dataset at 44.1 kHz. The model achieves an SI-SNRi of 13.3 dB (improvement over the mixture) and an absolute SI-SNR of 16.88 dB on ground-truth data. 
## Setup

1. Download and extract files. 
2. Navigate to the folder:
```python
cd: "D:\Conv-TasNet_44.1_Augmentation\code"
```
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

3. Prepare your dataset: \
   a. **Download** original VCTK-2mix. \
   b. Set the required environment variables (PowerShell example):
```python
$env:VCTK\SOURCE\DIR = "E:\Dataset_path\VCTK-2mix"   # VCTK path.
$env:VCTK\DATA\BASE = "D:\Conv-TasNet_44.1_Augmentation\data"    # Base directory for generated datasets.
```

4. Generate the 44.1 kHz dataset. This creates VCTK2mix\_44k with 30,000 training mixtures (and 6k/3k val/test): \
Run the first script:
 ```python
python generate_vctk2mix.py
```

6. Generate the 16 kHz version (optional, for two-stage training). This creates VCTK2mix\_16k/ from the 44.1 kHz dataset: \
Run the second script:
```python
python resample_to_16k.py
```

## Training

1. Configurate model: \
a. Open `config.py` in IDE. \
b. You can adjust the following parameters as needed: _num_ of _epochs_, _learning rate_, _model size_ (bn\_chan; n\_filters; hid\_chan),
_kernel param_ (kernel\_size; stride), _segment_ or _sample rate_. \
**VRAM/RAM WARNING**: The default configuration uses ~5 GB of VRAM and 40–60 GB of RAM (due to dataset preloading). Adjust batch_size or 'preload=False' in 'train.py' if memory is limited.

3. Run training file:
```py
python train.py
```
The best model will be saved in the checkpoint directory (checkpoints/).

## Separating

To separate a single audio file into two sources. The script will load the best model from checkpoints/best\_model.pth and save two files. \
Run separating files:
```python
python separate_audio.py --input "path/to/mixture.wav" --output "path/to/output_dir" --model "path/to/best_model.pth"
```

Example: 
```python
python separate_audio.py --input "E:\Conv-TasNet_44.1_Augmentation\test\test_mix.wav" --output "E:\Conv-TasNet_44.1_Augmentation\test" --model "E:\Conv-TasNet_44.1_Augmentation\checkpoints\best_model.pth"
```

## License

This project is licensed under the MIT License — see the LICENSE.txt for details.

## Acknowledgments
* ![Jusper Lee](https://github.com/JusperLee/Conv-TasNet)

* ![Luo Yi; Nima Mesgarani](https://github.com/naplab/Conv-TasNet)

