# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from torchvision.datasets.utils import download_url
import torch
import os

def find_model(model_name):

    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage, weights_only=False)

    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]
    return checkpoint

def resume_model(model_name):
    
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage, weights_only=False)

    return checkpoint
    
