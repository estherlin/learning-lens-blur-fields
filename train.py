# Script for training a 5D blur field
import sys
import json
import argparse

import tqdm
import imageio
import numpy as np

import torch
import torch.nn.functional as F

from util.utils import make_results_dir

try:
	import tinycudann as tcnn
except ImportError:
	print("This code requires the tiny-cuda-nn extension for PyTorch.")
	print("============================================================\n")
	sys.exit()

# Constants
BASIS = [torch.sin, torch.cos]

# Helper functions
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

    # Add in the dimensions for the non-uv dimensions 8
    for nd in ndims:
        pts = torch.concat(
            [pts,  torch.mul(torch.ones((num_points**2, 1),  dtype=du.dtype, device=device), nd)], -1)

    # Apply encoding to input points
    pts_enc = encode(pts, num_fourier_features=num_fourier_features) # We didn't need them in the end
    return pts_enc

def infer(model, pts, N, n_channels):
    """ Infers the PSF given the input points."""
    prediction = model(pts)
    rgb = torch.sigmoid(prediction).to(torch.float16)
    kern = torch.reshape(rgb, shape=(1, 1, N//2 + 1, N//2 + 1, n_channels))
    rgb_kern = F.pad(input=kern, pad=(0, 0, N//4, N//4, N//4, N//4), mode='constant', value=0)
    return rgb_kern

def render(model, pts, sharp, N, n_channels):
    """ Renders the model given the input points and sharpened image. 
    Args:
        model: The model to render.
        pts: The input points.
        sharp: The sharpened image.
    Returns:
        The rendered image and the predicted PSF.
    """
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
    tensor[torch.isnan(tensor)] = 0
    tensor[torch.isinf(tensor)] = 0
    return tensor

def bayerfy(tensor):
    num_pat, h, w, c = tensor.shape
    mosaic = torch.zeros((num_pat, h, w), device=tensor.device, dtype=tensor.dtype)
    mosaic[:, 0::2, 0::2] = tensor[:, 0::2, 0::2, 0]
    mosaic[:, 0::2, 1::2] = tensor[:, 0::2, 1::2, 1]
    mosaic[:, 1::2, 0::2] = tensor[:, 1::2, 0::2, 2]
    mosaic[:, 1::2, 1::2] = tensor[:, 1::2, 1::2, 3]
    return mosaic

def plot_grid_kerns(model, num_h, num_w, full_h, full_w, num_p, N, pad, save_path, n_channels, num_fourier_features=0):
    
    # Get grid
    hs = torch.linspace(-1., 1., full_h, dtype=torch.float16, device=device)
    ws = torch.linspace(-1., 1., full_w, dtype=torch.float16, device=device)
    ps = torch.linspace(-1.0, 1.0, num_p, dtype=torch.float16, device=device)
    h_locs = [int((i+0.5)*full_h/num_h) for i in range(num_h)]
    w_locs = [int((i+0.5)*full_w/num_w) for i in range(num_w)]

    # Make a gif to store results for each lens position
    gif = imageio.get_writer(f"{save_path}.gif",mode='I')
    
    for o, p in enumerate(ps):
        kerns = np.zeros((num_h*(N-2*pad), num_w*(N-2*pad), 3))

        for i, hi in enumerate(h_locs):
            for j, wi in enumerate(w_locs):
                # Fetch the points and patches
                ndims = [hs[hi], ws[wi], ps[o]]
                pts = generate_pts(ndims, num_points=N//2+1, num_fourier_features=num_fourier_features, device=device)
                kern = infer(model, pts, N, n_channels)
                kern_numpy = np.squeeze(kern.cpu().detach().numpy())
                kern_numpy /= np.max(kern_numpy)

                if n_channels == 2:
                    kerns[i*(N-2*pad):(i+1)*(N-2*pad), j*(N-2*pad):(j+1)*(N-2*pad), 0] = kern_numpy[pad:-pad, pad:-pad, 0]
                    kerns[i*(N-2*pad):(i+1)*(N-2*pad), j*(N-2*pad):(j+1)*(N-2*pad), 1] = kern_numpy[pad:-pad, pad:-pad, 1]
                    kerns[i*(N-2*pad):(i+1)*(N-2*pad), j*(N-2*pad):(j+1)*(N-2*pad), 2] = kern_numpy[pad:-pad, pad:-pad, :].mean(axis=-1)
                else:
                    kerns[i*(N-2*pad):(i+1)*(N-2*pad), j*(N-2*pad):(j+1)*(N-2*pad), 0] = kern_numpy[pad:-pad, pad:-pad, 0]
                    kerns[i*(N-2*pad):(i+1)*(N-2*pad), j*(N-2*pad):(j+1)*(N-2*pad), 1] = 0.5*(kern_numpy[pad:-pad, pad:-pad, 1] + kern_numpy[pad:-pad, pad:-pad, 2])
                    kerns[i*(N-2*pad):(i+1)*(N-2*pad), j*(N-2*pad):(j+1)*(N-2*pad), 2] = kern_numpy[pad:-pad, pad:-pad, 3]
        gif.append_data(np.uint8(kerns*255.))
    gif.close()

def config_parser():
    
    # Create the parser
    parser = argparse.ArgumentParser(description='Training arguments')
    # Add the arguments
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

    parser.add_argument('--config-path', type=str, required=True, 
                    help='Path to the config file for the model')
    parser.add_argument('--results-path', type=str, required=True,
                    help='Path to the directory to save the results')
    parser.add_argument('--data-path', type=str, required=True,
                    help='Path to the dataset')
    parser.add_argument('--dims', type=str, default="xyuvp",
                    help='Coordinate dimensions to train on')
    parser.add_argument('--training-stride', type=int, default=20,
                    help='Patch size to do convolutions')
    parser.add_argument('--rendering-stride', type=int, default=20,
                    help='Patch size to do image renders')
    parser.add_argument('--kern-radius', type=int, default=30,
                    help='1/4 of kernel radius to estimate')
    parser.add_argument('--downsample', type=int, default=1,
                    help='amount to downsample image plane with')
    parser.add_argument('--leopard-freqs', nargs="*", type=str, default=[], required=True,
                    help='Patterns to train with')
    parser.add_argument('--mode', type=str, default=0, required=True,
                    help='train vs test')
    parser.add_argument('--n-channels', type=int, default=4, required=True,
                    help='Number of output channels')
    parser.add_argument('--kern-grid-h', type=int, default=3,
                    help='Number of PSFs in height to save in grid')
    parser.add_argument('--kern-grid-w', type=int, default=4,
                    help='Number of PSFs in width to save in grid')
    parser.add_argument('--ffs', type=int, default=0,
                    help='Number of Fourier Features')
    parser.add_argument('--n-sampling', type=int, default=2,
                    help='Degree of sampling, the power of 2')
    parser.add_argument('--log-step', type=int, default=10000,
                    help='Number of iterations between logging')
    # Execute the parse_args() method
    return parser.parse_args()


if __name__ == '__main__':
    # System settings
    print("\nTorch version:", torch.__version__)
    print("CUDNN available:",torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\nDevice:", device)

    # Get args
    args = config_parser()

    # grab params from the command line
    config_filename     = f"{args.config_path}" # add path to config file
    results_dir     	= f"{args.results_path}" # add path to results directory
    data_path     	    = f"{args.data_path}" # add path to results directory
    dims                = f"{args.dims}" # number of coordinate dimensions
    training_stride     = int(f"{args.training_stride}") # Percentage of kernel to use for loss comparison
    rendering_stride    = int(f"{args.rendering_stride}") # Percentage of kernel to use for image rendering
    downsample          = 2**int(f"{args.downsample}") # downsampling factor
    kern_radius         = int(int(f"{args.kern_radius}")/downsample) # patch to do convolution
    train_patterns      = args.leopard_freqs # Patterns to train with
    mode                = f"{args.mode}" # train vs test
    n_channels          = int(f"{args.n_channels}") # Number of output channels
    kern_grid_h         = int(f"{args.kern_grid_h}") # Number of PSFs in height to save in grid
    kern_grid_w         = int(f"{args.kern_grid_w}") # Number of PSFs in width to save in grid
    n_sampling          = int(f"{args.n_sampling}") # Degree of sampling, the power of 2
    log_step            = int(f"{args.log_step}") # Number of iterations between logging
    num_fourier_features = int(f"{args.ffs}") # Number of Fourier Features

    # Make sure we have the directory for the results
    make_results_dir(results_dir)

    # Load in configuration
    with open(f"{config_filename}") as config_file:
        config = json.load(config_file)
    print("\n\nNetwork configs: \n", config)


    ################### Load in data
    # Load in data 
    data = np.load(f'{data_path}')
    patterns = data[f'patterns_{mode}']
    blurry = torch.from_numpy(data[f'blurry_{mode}']).to(torch.float16).to(device)
    sharp = torch.from_numpy(data[f'sharp_{mode}']).to(torch.float16).to(device)
    lens_positions = data[f'lens_positions']
    data.close()

    # image constants
    input_h, input_w = blurry.shape[2], blurry.shape[3]

    # Training related parameters
    N = 4*kern_radius+1
    sH, sW = N//2, N//2 # Will always be even
    ph, pw = int(N*(1+training_stride/100)), int(N*(1+training_stride/100))
    size = (int(N*(1+rendering_stride/100))-N)//2 # render stride = 5%
    IH, IW = sharp.shape[2:4]

    # Training dimensions
    hs = torch.linspace(-1., 1., IH, dtype=torch.float16, device=device)
    ws = torch.linspace(-1., 1., IW, dtype=torch.float16, device=device)
    ps = torch.from_numpy(np.array(lens_positions)).to(torch.float16).to(device)#*2-1
    print("Training lens positions in diopter (1/m):", ps)

	# Make model, optimizer
    print("\nMaking model...")
    model = tcnn.NetworkWithInputEncoding(
        n_input_dims=len(dims)*(len(BASIS)*num_fourier_features+1), 
        n_output_dims=n_channels, 
        encoding_config=config["encoding"], 
        network_config=config["network"]
    )
    for p in model.parameters():
        p.register_hook(lambda grad: rmnan(grad))
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["optimizer"]["learning_rate"],
        betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"]),
        eps=config["optimizer"]["epsilon"]
    )
    lrschedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["optimizer"]["gamma"])

    # Logging params
    steps = config["optimizer"]["iterations"]
    sampling = 2**(n_sampling+1) # Can never be 2**0

    print(f"Beginning optimization with {steps} training steps.")
    for i in tqdm.tqdm(range(steps)):

        # Get random point
        rand_h = sampling*torch.randint(int((IH-ph)/sampling), [], device=device)
        rand_w = sampling*torch.randint(int((IW-pw)/sampling), [], device=device)
        rand_p = torch.randint(len(lens_positions), [], device=device)

        # Fetch the points and patches
        ndims = [hs[rand_h+int(ph/2)], ws[rand_w+int(pw/2)], ps[rand_p]]
        pts = generate_pts(ndims, num_points=N//2+1, num_fourier_features=num_fourier_features, device=device)
        sharp_cut = sharp[:,rand_p, rand_h:rand_h+ph, rand_w:rand_w+pw, :].to(sharp.dtype).to(device)
        blur_cut = blurry[:,rand_p, rand_h+sH:rand_h+ph-sH, rand_w+sH:rand_w+pw-sH].to(blurry.dtype).to(device)

        # Render
        rgb, kern = render(model, pts, sharp_cut, N, n_channels)
        rgb_bayer = bayerfy(rgb)

        # Find loss
        loss = F.mse_loss(rgb_bayer, blur_cut) 
        loss +=  1e-4 * torch.sum(torch.abs(kern))

        # grad
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % log_step == 0 and i != 0:
            with torch.no_grad():
                # Log
                print(f"\nRendering kernels for step {i}...")
                plot_grid_kerns(model, kern_grid_h, kern_grid_w, IH, IW, len(lens_positions), N, kern_radius-1, f"{results_dir}/training-inference_{i}", n_channels, num_fourier_features)
                print(f"\nSaving model for step {i}...")
                torch.save(model.state_dict(), f"{results_dir}/model-{i}.pth")

        # Update step
        if i % config["optimizer"]["update_step"] == 0:
            lrschedule.step()

    print("\n\nFinished training a blur field.")