

This repository contains the code for our project  "Stylized Projected GAN: A Novel Architecture for Fast
and Realistic Image Generation"

by Mohammad Nurul Muttakin and Malik Shahid Sultan.



## Requirements ##
- 64-bit Python 3.8 and PyTorch 1.9.0 (or later). See https://pytorch.org for PyTorch install instructions.
- Use the following commands with Miniconda3 to create and activate your PG Python environment:
  - ```conda env create -f environment.yml```
  - ```conda activate pg```
- The StyleGAN2 generator relies on custom CUDA kernels, which are compiled on the fly. Hence you need:
  - CUDA toolkit 11.1 or later.
  - GCC 7 or later compilers. Recommended GCC version depends on CUDA version, see for example CUDA 11.4 system requirements.
  - If you run into problems when setting up for the custom CUDA kernels, we refer to the [Troubleshooting docs](https://github.com/NVlabs/stylegan3/blob/main/docs/troubleshooting.md#why-is-cuda-toolkit-installation-necessary) of the original StyleGAN repo. When using the FastGAN generator you will not need the custom kernels.

## Data Preparation ##
For a quick start, you can download the few-shot datasets provided by the authors of [FastGAN](https://github.com/odegeasslbc/FastGAN-pytorch). You can download them [here](https://drive.google.com/file/d/1aAJCZbXNHyraJ6Mi13dSbe7pTyfPXha0/view). To prepare the dataset at the respective resolution, run for example
```
python dataset_tool.py --source=./data/pokemon --dest=./data/pokemon256.zip \
  --resolution=256x256 --transform=center-crop
```
You can get the datasets we used in our project at their respective websites: 

[CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/), [FFHQ](https://github.com/NVlabs/ffhq-dataset), [Cityscapes](https://www.cityscapes-dataset.com/), [LSUN](https://github.com/fyu/lsun), [AFHQ](https://github.com/clovaai/stargan-v2), [Landscape](https://www.kaggle.com/arnaud58/landscape-pictures).

## Training ##

conda activate pg

Training on Pokemon for original Projected GAN
python train.py --outdir=./training-runs/ --cfg=fastgan --data=./data/pokemon256.zip \
--gpus=8 --batch=64 --mirror=1 --snap=50 --batch-gpu=8 --kimg=10000


Training for best deep model on FFHQ dataset
python train.py --outdir=./training-runs-70k-deep/ --cfg=styled_fastgan_deep --data=./data/ffhq70k-256.zip \
--gpus=8 --batch=64 --mirror=1 --snap=50 --batch-gpu=8 --kimg=80000


Training for original Projected GAN on FFHQ dataset
python train.py --outdir=./training-runs-70k/ --cfg=fastgan --data=./data/ffhq70k-256.zip \
--gpus=8 --batch=64 --mirror=1 --snap=50 --batch-gpu=8 --kimg=10000







Training for the best lightweight stylized projected gan on FFHQ

python train.py --outdir=./training-runs-70k/ --cfg=styled_fastgan_v1b --data=./data/ffhq70k-256.zip \
--gpus=8 --batch=64 --mirror=1 --snap=50 --batch-gpu=8 --kimg=10000

Training for the runner up stylized projected gan on FFHQ
python train.py --outdir=./training-runs-70k/ --cfg=styled_fastgan --data=./data/ffhq70k-256.zip \
--gpus=8 --batch=64 --mirror=1 --snap=50 --batch-gpu=8 --kimg=10000


## Generating Samples & Interpolations ##

To generate samples and interpolation videos, run
```
python gen_images.py --outdir=out --trunc=1.0 --seeds=10-15 \
  --network=PATH_TO_NETWORK_PKL
```
and
```
python gen_video.py --output=lerp.mp4 --trunc=1.0 --seeds=0-31 --grid=4x2 \
  --network=PATH_TO_NETWORK_PKL
```
  
## Quality Metrics ##
Per default, ```train.py``` tracks FID50k during training. To calculate metrics for a specific network snapshot, run

```
python calc_metrics.py --metrics=fid50k_full --network=PATH_TO_NETWORK_PKL
```

To see the available metrics, run
```
python calc_metrics.py --help
```



## Acknowledgments ##
Our codebase build and extends the awesome [ProjectedGAN](https://github.com/autonomousvision/projected_gan), [StyleGAN2-ADA repo](https://github.com/NVlabs/stylegan2-ada-pytorch) and [StyleGAN3 repo](https://github.com/NVlabs/stylegan3).

Furthermore, we use parts of the code of [FastGAN](https://github.com/odegeasslbc/FastGAN-pytorch) and [MiDas](https://github.com/isl-org/MiDaS).
