
# ARFlow: Autogressive Flow with Hybrid Linear Attention

This repo contains PyTorch model definitions, pre-trained weights and training/sampling code for our paper ARFlow: Autogressive Flow with Hybrid Linear Attention. [Arxiv](https://arxiv.org/abs/2501.16085)

## Introduction
We develop ARFlow, a novel framework integrating autoregressive modeling into flow models to better capture long-range dependencies in image generation. During training, ARFlow constructs causally-ordered sequences from multiple noisy images of the same semantic category, and during generation, it autoregressively conditions on previously denoised images. To handle the computational demands of sequence modeling, ARFlow employs a hybrid attention mechanism with full attention within images and linear attention across images

## Preparation

pip install -r requirements.txt


Download ImageNet dataset and cache VAE latents using cache_imagenet.py

## Training
```python
export TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1
torchrun --nnodes=1 --nproc_per_node=N train.py --model {model name} --data-path /path/to/imagenet
```
## Infering
Download the pertrained model weight [here](https://huggingface.co/MudeHui/ARFlow-Weights).
```python
export TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1
python sample.py --ckpt /path/to/ARFlow_weight --image-size=256/128 --model={model name} --cfg-scale={cfg scale}

torchrun --nnodes=1 --nproc_per_node=N --master_port 25999 sample_ddp.py  --model ARFlow-XL/2 --num-fid-samples {number} --cfg {cfg scale} --num-sampling-steps {setps} --ckpt /path/to/ARFlow_weight
```