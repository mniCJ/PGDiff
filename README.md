# PGDiff：Perception-Guided Diffusion for Zero-Shot Low-Light Image Enhancement with Adaptive Frequency Modulation.
This is a Python implementation for the paper “Perception-Guided Diffusion for Zero-Shot Low-Light Image Enhancement with Adaptive Frequency Modulation”, which is under review in the journal "The Visual Computer". We have uploaded codes, datasets and all the test results for reference.

## Method
<img width="1481" height="772" alt="2-new" src="https://github.com/user-attachments/assets/49e7dc06-4594-4505-955d-96be75432852" />

## Codes and Environment
```
# git clone this repository
git clone https://github.com/mniCJ/PGDiff.git
cd PGDiff

# create new anaconda env
conda create -n pgdiff python=3.8 -y
conda activate pgdiff

# install python dependencies
conda install mpi4py
pip3 install -r requirements.txt
pip install -e .
```

## Datasets
Download the following datasets into the `data` folder:

LOL: [Google Drive](https://drive.google.com/file/d/1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H/view?usp=sharing)

ExDark: [Google Drive](https://drive.google.com/file/d/1BHmPgu8EsHoFDDkMGLVoXIlCth2dW6Yx/view?usp=sharing)

DarkFace: [Google Drive](https://drive.google.com/file/d/10W3TDvEAlZfEt88hMxoEuRULr42bIV7s/view?usp=sharing)

## Pretrained Model
Download the pretrained diffusion model from [guided-diffusion](https://github.com/openai/guided-diffusion?tab=readme-ov-file). Place the model in the `ckpt` folder.
```
cd ckpt
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
cd ..
```

## Evaluation
### Example usage:
```
python inference.py --in_dir ./data/demo --out_dir ./results/demo
```

