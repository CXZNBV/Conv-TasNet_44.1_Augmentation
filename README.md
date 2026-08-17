\# Conv-TasNet Augmentation for 44.1 kHz

This repository contains the training code for a modified Conv-TasNet architecture adapted for speech separation
on the VCTK-2mix dataset at 44.1 kHz.

\## Setup

1. Setup python venv in your directory.
'''bash
python -m venv venv  
source venv/bin/activate

# On Windows: venv\\Scripts\\activate

2\. Install dependencies: pip install -r requirements.txt.

3\. Prepare your dataset:
a. Download original VCTK-2mix.
b. Set the required environment variables (PowerShell example):
$env:VCTK\_SOURCE\_DIR = "D:\\Dataset\\VCTK-2mix"   # VCTK path.
$env:VCTK\_DATA\_BASE = "E:\\Conv-Tas Net Augmentation\\data"    # Base directory for generated datasets.

4\. Generate the 44.1 kHz dataset. This creates VCTK2mix\_44k with 30,000 training mixtures (and 6k/3k val/test):
Run the first script: python generate\_vctk2mix.py.



5\. Generate the 16 kHz version (optional, for two-stage training). This creates VCTK2mix\_16k/ from the 44.1 kHz dataset:
Run the second script: python resample\_to\_16k.py.

\## Training

1. Configurate model:
a. Open config.py in IDE.
b. You can adjust the following parameters as needed: num of epochs, learning rate, model size (bn\_chan; n\_filters; hid\_chan),
kernel param (kernel\_size; stride), segment or sample rate.
VRAM/RAM WARNING: The default configuration uses \~5 GB of VRAM and 40–60 GB of RAM (due to dataset preloading). Adjust batch\_size or 'preload=False' in 'train.py' if memory is limited.

2\. Run training file: python train.py
The best model will be saved in the checkpoint directory (checkpoints/).



\## Separating

1. To separate a single audio file into two sources. The script will load the best model from checkpoints/best\_model.pth and save two files:
python separate_audio.py \
    --input "path/to/mixture.wav" \
    --output "path/to/output_dir" \
    --model "path/to/best_model.pth"
Run separating file: 
python separate\_audio.py
    --input "E:\\CONV-TAS NET ENV\\test\\test\_mix.wav"
    --output "E:\\CONV-TAS NET ENV\\test"
    --model "E:\\CONV-TAS NET ENV\\checkpoints\\best\_model\_new.pth"



\## License

This project is licensed under the MIT License — see the LICENSE.txt for details.

\## Citation

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
