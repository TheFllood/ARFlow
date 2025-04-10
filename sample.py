# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained ARFlow.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from download import find_model
from models_infer import ARFlow_models
from train_utils import  parse_sde_args, parse_transport_args
from transport import create_transport, Sampler
import argparse
import sys
from time import time



torch.set_default_dtype(torch.float32)
def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed) 
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load model:
    latent_size = args.image_size // 8
    model = ARFlow_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        learn_sigma=True,
    ).to(device)
    ckpt_path = args.ckpt 
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )
    sampler = Sampler(transport)
    sample_fn = sampler.sample_sde(
            sampling_method=args.sampling_method,
            diffusion_form=args.diffusion_form,
            diffusion_norm=args.diffusion_norm,
            last_step=args.last_step,
            last_step_size=args.last_step_size,
            num_steps=args.num_sampling_steps,
        )
    
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 89, 979, 417, 280]

    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    start_time = time()
    print("start")
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
    for block in model.blocks:
        block.incremental_state = {}
    samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)[-1]
    image_samples, _ = samples.chunk(2, dim=0)  
    print(f"Sampling took {time() - start_time:.2f} seconds.")
    
    image_samples = vae.decode(image_samples / 0.18215).sample
    save_image(image_samples, f"samples.png", nrow=4, normalize=True, value_range=(-1, 1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    if len(sys.argv) < 2:
        print("Usage: program.py <mode> [options]")
        sys.exit(1)
    
    parser.add_argument("--model", type=str, choices=list(ARFlow_models.keys()), default="ARFlow-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image-size", type=int, choices=[128, 256], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a ARFlow checkpoint")

    parse_transport_args(parser)
    parse_sde_args(parser)
    
    args = parser.parse_known_args()[0]
    main(args)
