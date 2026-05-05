import numpy as np
import imageio
import sys
import os
import torch
import matplotlib.pyplot as plt

CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
sys.path.insert(0, CONFIGS_DIR)

WEIGHTS = {
    "red": 64,
    "green": 128,
    "blue": 192,
}
COLOURS = {
    0: "red",
    1: "green",
    2: "blue",
}

def write_image(path, im, write=True):
    im_new = np.zeros((im.shape[0], im.shape[1], 3), np.float32)

    if im.shape[-1] == 4:
        im_new[...,0] = im[...,0]
        im_new[...,2] = im[...,3]
        im_new[...,1] = 0.5*(im[...,1] + im[...,2])
    else:
        im_new = im
    im_new = im_new/im_new.max()
    for i in range(3):
        im_new[...,i] = im_new[...,i]*WEIGHTS[COLOURS[i]]

    im_new = np.uint8(im_new)
    if write:
        imageio.imwrite(path, im_new)
    else:
        return im_new

def convert_uint8(im):
	return write_image(None, im, write=False)

def normalize(im):
	return (im) / (im.max())

def plot_progress(rand_p, path, blur_cut, rgb, kern, itr, save_dir):
    fig, ax = plt.subplots(4, 4, figsize=(15, 15))
    for i in range(4):
        vmax = max(rgb.squeeze().detach().cpu().numpy()[1,...,i].max(), blur_cut.squeeze().detach().cpu().numpy()[1,...,i].max())
        vmin = min(rgb.squeeze().detach().cpu().numpy()[1,...,i].min(), blur_cut.squeeze().detach().cpu().numpy()[1,...,i].min())

        im = ax[i][0].imshow(rgb.squeeze().detach().cpu().numpy()[1,...,i], cmap='gray', vmax=vmax, vmin=vmin)
        fig.colorbar(im, ax=ax[i][0])
        ax[i][0].set_title("Synthetic Image")
        im = ax[i][1].imshow(blur_cut.squeeze().detach().cpu().numpy()[1,...,i], 'gray', vmax=vmax, vmin=vmin)
        fig.colorbar(im, ax=ax[i][1])
        ax[i][1].set_title("GND Image")
        im = ax[i][2].imshow(100*np.abs(blur_cut.squeeze().detach().cpu().numpy()[1,...,i] - rgb.squeeze().detach().cpu().numpy()[1,...,i])/vmax)
        fig.colorbar(im, ax=ax[i][2])
        ax[i][2].set_title("L1 |Error|")
        im = ax[i][3].imshow(kern.squeeze().detach().cpu().numpy()[...,i], cmap='gray')
        fig.colorbar(im, ax=ax[i][3])
        ax[i][3].set_title("Kernel")
    plt.suptitle("Iteration {}, lens pos {}".format(itr, rand_p))
    plt.savefig(save_dir+"progress/itr-{}-{}.png".format(itr, rand_p))
    plt.close()


def make_dirs(root_dir):

    # Check the root dir
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # Make a dir to save gifs
    if not os.path.exists(root_dir+'/|L1|'):
        os.mkdir(root_dir+'/|L1|')

    if not os.path.exists(root_dir+'/stats'):
        os.mkdir(root_dir+'/stats')

    if not os.path.exists(root_dir+'/progress'):
        os.mkdir(root_dir+'/progress')

def make_dir(dirpath):

    # Check the root dir
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    else:
        print(f"Directory {dirpath} already exists")

def make_results_dir(results_dir, skip=True):
    try:
        os.makedirs(results_dir)
    except FileExistsError:
        if skip:
            print("Directory already exists. Continuing with execution.\n")
        else:
            answer = input("\nDo you want to continue? (y/[n]): ")
            if answer.lower() == 'y':
                print("Continuing with execution.\n")
            else:
                print("Execution aborted.")
                sys.exit()


def make_4D_checkerboard(height=1500, width=2000, square_size=100):
    sharp = np.zeros((1, height, width, 4), dtype=np.float32)

    # make a checkerboard
    for i in range(height//square_size):
        for j in range(width//square_size):
            if (i+j)%2 == 0:
                sharp[0,i*square_size:(i+1)*square_size, j*square_size:(j+1)*square_size,:] = 1
    return sharp

def make_video_stack(restack, device):
	video_stack = restack.permute(2,0,1).to(device)
	video_stack /= video_stack.max()
	video_stack *= 255
	video_stack_numpy = video_stack.to(torch.uint8).detach().cpu().numpy()
	return video_stack_numpy


def restack(im):
    return torch.concat([im[:,i,...] for i in range(im.shape[1])], axis=-1)
