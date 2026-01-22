#!/usr/bin/env python3
"""
Training script for 5D blur field learning.
This script trains a neural network to learn the lens blur field from focal stack data.
"""

import sys
import json
import argparse
import os
from pathlib import Path

import tqdm
import imageio
import numpy as np

import torch
import torch.nn.functional as F

# Add parent directory to path to import util modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from util.utils import make_results_dir

try:
    import tinycudann as tcnn
except ImportError:
    print("This code requires the tiny-cuda-nn extension for PyTorch.")
    print("Please install it following the instructions in the README.")
    print("============================================================\n")
    sys.exit()

# Constants
BASIS = [torch.sin, torch.cos]

def encode(x, num_fourier_features, include_input=True):
    """ Encode a position into a sinusoidal vector"""
    if num_fourier_features == 0:
        return x

    encoding = [x] if include_input else []
    frequency_bands = torch.linspace(
        2.0 ** 0.0,
        2.0 ** (num_fourier_features - 1),
        num_fourier_features,
        dtype=x.dtype,
        device=x.device,
    )

    for freq in frequency_bands:
        for func in BASIS:
            encoding.append(func(x * freq))
    return torch.cat(encoding, dim=-1).to(x.device)

def generate_pts(ndims, device, num_points, num_fourier_features=0):
    """
    Generate a set of points for a given number of dimensions.
    """
    # Make a 2D grid of points for one kernel u,v
    du, dv = torch.meshgrid(
        torch.linspace(-1., 1., num_points, dtype=torch.float16, device=device),
        torch.linspace(-1., 1., num_points, dtype=torch.float16, device=device))
    
    pts = torch.stack([
        torch.flatten(du),
        torch.flatten(dv)], -1).to(device)

    # Add in the dimensions for the non-uv dimensions
    for nd in ndims:
        pts = torch.concat(
            [pts, torch.mul(torch.ones((num_points**2, 1), dtype=du.dtype, device=device), nd)], -1)

    # Apply encoding to input points
    pts_enc = encode(pts, num_fourier_features=num_fourier_features)
    return pts_enc

def infer(model, pts, N, n_channels):
    """ Infers the PSF given the input points."""
    prediction = model(pts)
    rgb = torch.sigmoid(prediction).to(torch.float16)
    kern = torch.reshape(rgb, shape=(1, 1, N//2 + 1, N//2 + 1, n_channels))
    rgb_kern = F.pad(input=kern, pad=(0, 0, N//4, N//4, N//4, N//4), mode='constant', value=0)
    return rgb_kern

def render(model, pts, sharp, N, n_channels, device):
    """ Renders the model given the input points and sharpened image."""
    rgb_kern = infer(model, pts, N, n_channels)

    # Concatenate the rgb and sigma into a single tensor.
    sharp_4d = sharp.to(torch.float16).to(device)
    s_h, s_w = sharp_4d.shape[-3:-1]
    convolved = torch.stack( 
        [torch.nn.functional.conv2d(torch.reshape(sharp_4d[...,i], (sharp_4d.shape[0],1,s_h,s_w)), rgb_kern[...,i], stride=1, padding="valid") for i in range(n_channels)
        ]
        , axis=-1)
    return torch.reshape(convolved, (sharp_4d.shape[0],s_h-N+1,s_w-N+1,n_channels)), torch.reshape(rgb_kern, shape=(1, N, N, n_channels))

def rmnan(tensor):
    """Remove NaN and Inf values from tensor."""
    tensor[torch.isnan(tensor)] = 0
    tensor[torch.isinf(tensor)] = 0
    return tensor

def bayerfy(tensor):
    """Apply Bayer pattern to tensor."""
    # Implementation depends on specific Bayer pattern requirements
    return tensor

def config_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train 5D blur field')
    
    # Data and model paths
    parser.add_argument('--config-path', type=str, required=True,
                       help='Path to configuration JSON file')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to training data (.npz file)')
    parser.add_argument('--results-path', type=str, required=True,
                       help='Path to save results and checkpoints')
    
    # Training parameters
    parser.add_argument('--dims', type=str, default='xyuvp',
                       help='Dimensions to use (e.g., xyuvp)')
    parser.add_argument('--training-stride', type=int, default=50,
                       help='Training stride')
    parser.add_argument('--rendering-stride', type=int, default=50,
                       help='Rendering stride')
    parser.add_argument('--kern-radius', type=int, default=40,
                       help='Kernel radius')
    parser.add_argument('--downsample', type=int, default=0,
                       help='Downsample factor')
    parser.add_argument('--ffs', type=int, default=0,
                       help='Number of Fourier features')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'test', 'eval'],
                       help='Training mode')
    parser.add_argument('--n-channels', type=int, default=4,
                       help='Number of color channels')
    parser.add_argument('--n-sampling', type=int, default=2,
                       help='Number of sampling points')
    parser.add_argument('--log-step', type=int, default=100000,
                       help='Logging frequency')
    
    # Leopard pattern parameters
    parser.add_argument('--leopard-freqs', nargs='+', type=int, 
                       default=[10, 20, 50, 70],
                       help='Leopard pattern frequencies')
    
    return parser.parse_args()

def main():
    """Main training function."""
    args = config_parser()
    
    # Load configuration
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    # Create results directory
    make_results_dir(args.results_path)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {args.data_path}")
    data = np.load(args.data_path)
    
    # Initialize model
    print("Initializing model...")
    model = tcnn.Network(
        n_input_dims=len(args.dims),
        n_output_dims=args.n_channels,
        encoding=config["encoding"],
        network=config["network"]
    ).to(device)
    
    # Initialize optimizer
    optimizer = tcnn.Optimizer(model, config["optimizer"])
    
    # Training loop
    print("Starting training...")
    for iteration in tqdm.tqdm(range(config["optimizer"]["iterations"])):
        # Training step implementation here
        # This is a simplified version - you'll need to implement the full training loop
        # based on your specific requirements
        
        if iteration % args.log_step == 0:
            print(f"Iteration {iteration}")
            # Save checkpoint
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'args': vars(args)
            }, os.path.join(args.results_path, f'checkpoint_{iteration}.pth'))
    
    print("Training completed!")

if __name__ == "__main__":
    main() 