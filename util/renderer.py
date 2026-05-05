import array
import os

import Imath
import numpy as np
import OpenEXR

import tinycudann as tcnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage.io as io
from torchvision.transforms.functional import to_pil_image



class Renderer():
    def __init__(self, N:int, model_path:str, config):
        torch.backends.cudnn.deterministic = True
        self.N = N # Kernel size during solving, should be a multiple of 4 + 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims=5,
            n_output_dims=4,
            encoding_config=config["encoding"], 
            network_config=config["network"]).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def load_image(self, image_path, pattern="RGGB"):
        image_path_head_tail = os.path.split(image_path)
        self.image_folder = image_path_head_tail[0]
        self.image_name = image_path_head_tail[1]
        extension = self.image_name.split(".")[-1]
        self.image_name = "".join(self.image_name.split(".")[:-1])
        # Compute the size
        if extension == "exr":
            file = OpenEXR.InputFile(image_path)
            dw = file.header()['dataWindow']
            sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

            # Read the three color channels as 32-bit floats
            FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
            (R, G, B) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B")]
            R, G, B = [np.array(x).reshape(sz[1], sz[0]) for x in (R, G, B)]
            self.raw = np.stack((R, G, G, B), axis=-1)
            self.height = self.raw.shape[0]
            self.width = self.raw.shape[1]
            self.raw = np.expand_dims(self.raw, axis=0)
        else:
            rgb_image = np.float32(io.imread(image_path)/255.0)
            self.rgb_image = rgb_image
            self.height = rgb_image.shape[0]
            self.width = rgb_image.shape[1]
            self.raw = np.zeros((1, self.height, self.width, 4), dtype=np.float32)
            if rgb_image.shape[2] == 3 or (rgb_image.shape[2] == 4 and extension == "png"):
                # Convert to raw
                rgb_map = {'R':0, 'G':1, 'B':2}
                for c in range(4):
                    self.raw[0,...,c] = rgb_image[...,rgb_map[pattern[c]]]
            elif rgb_image.shape[2] == 4:
                self.raw = rgb_image
            else:
                raise ValueError("Image must be RGB or Bayer RAW.")
        # Init model output
        self.output = np.zeros((self.height, self.width, 4), dtype=np.float32)
    
    def load_depth(self, depth_path):
        self.depth = np.float32(io.imread(depth_path))
        if len(self.depth.shape) > 2:
            self.depth = self.depth[...,0]
        self.depth /= np.max(self.depth)
        # self.depth[self.depth > 0.85] += 0.25
        # self.depth /= np.max(self.depth)

        
    def sample_points(self, ndims, num_points):
        """
        Generate a set of points for a given number of dimensions.
        """
        # Make a 2D grid of points for one kernel u,v
        du, dv = torch.meshgrid(
            torch.linspace(-1., 1., num_points, dtype=torch.float32, device=self.device),
            torch.linspace(-1., 1., num_points, dtype=torch.float32, device=self.device))
        
        pts = torch.stack([
            torch.flatten(du),
            torch.flatten(dv)], -1).to(self.device)

        # Add in the dimensions for the non-uv dimensions 8
        for nd in ndims:
            pts = torch.concat(
                [pts, torch.mul(torch.ones((num_points**2, 1),  dtype=du.dtype, device=self.device), nd)], -1)

        return pts
    
    def inference(self, pts):
        """ Inference the model to get the PSF.
        Args:
            pts: The input sample points.
        Returns:
            The predicted PSF (blur kernel).
        """
        # Run network to get prediction
        prediction = self.model(pts)
        rgb = torch.sigmoid(prediction).to(torch.float32)
        kern = torch.reshape(rgb, shape=(1, 1, self.N//2 + 1, self.N//2 + 1, 4))
        # Normalize
        for c in range(4):
            kern[...,c] /= torch.sum(kern[...,c])
        # Pad the kernel for convolution
        kern = F.pad(input=kern, pad=(0, 0, self.N//4, self.N//4, self.N//4, self.N//4), mode='constant', value=0)
        return kern
    
    def convolve(self, kern, image):
        """ Blur the image patch with the given blur kernel. 
        Args:
            kern: The blur kernel.
            image: The image patch.
        Returns:
            The rendered image.
        """
        # Concatenate the rgb and sigma into a single tensor.
        image_4d = image.to(torch.float32).to(self.device)
        s_h, s_w = image_4d.shape[-3:-1]
        convolved = torch.stack(
            [torch.nn.functional.conv2d(torch.reshape(image_4d[...,0], (image_4d.shape[0],1,s_h,s_w)), kern[...,0], stride=1, padding="valid"), 
            torch.nn.functional.conv2d(torch.reshape(image_4d[...,1], (image_4d.shape[0],1,s_h,s_w)), kern[...,1], stride=1, padding="valid"), 
            torch.nn.functional.conv2d(torch.reshape(image_4d[...,2], (image_4d.shape[0],1,s_h,s_w)), kern[...,2], stride=1, padding="valid"), 
            torch.nn.functional.conv2d(torch.reshape(image_4d[...,3], (image_4d.shape[0],1,s_h,s_w)), kern[...,3], stride=1, padding="valid"), 
            ]
            , axis=-1)
        return torch.reshape(convolved, (image_4d.shape[0],s_h-self.N+1,s_w-self.N+1,4))
    
    def save(self, raw_output, filepath):
        # Save the synthetic image
        rgb_output = np.zeros((self.height, self.width, 3), dtype=np.float32)
        rgb_output[...,0] = raw_output[...,0]
        rgb_output[...,1] = 0.5*(raw_output[...,1]+raw_output[...,2])
        rgb_output[...,2] = raw_output[...,3]
        rgb_output = torch.from_numpy(np.transpose(rgb_output, (2, 0, 1)))
        to_pil_image(rgb_output).save(f"{filepath}")
        
    def rgb(self, raw_input):
        # Save the synthetic image
        rgb_output = np.zeros((self.height, self.width, 3), dtype=np.float32)
        rgb_output[...,0] = raw_input[...,0]
        rgb_output[...,1] = 0.5*(raw_input[...,1]+raw_input[...,2])
        rgb_output[...,2] = raw_input[...,3]
        return rgb_output        
        
    def render_single_layer(self, stride:int, lens_position:float=0., binary_mask=None):
        output = np.zeros((self.height, self.width, 4), dtype=np.float32)
        # Parameter space
        hs = torch.linspace(-2., 2., self.height, dtype=torch.float32, device=self.device)
        ws = torch.linspace(-2., 2., self.width, dtype=torch.float32, device=self.device)
        
        if binary_mask is not None:
            masked_image = np.zeros_like(self.raw)
            for c in range(4):
                masked_image[...,c] = self.raw[...,c] * binary_mask
        else:
            masked_image = self.raw
        
        # Patch sizes
        patch_height = int((100+stride)*self.N/100)
        patch_width = int((100+stride)*self.N/100)
        sH = (self.N)//2
        sW = (self.N)//2
        size = (patch_height-self.N)//2

        # Patch based rendering
        for rand_h in range(0, self.height-patch_height-1, 2*size):
            for rand_w in range(0, self.width-patch_width-1, 2*size):
                # fetch the points and patches
                ndims = [hs[rand_h+patch_height//2], ws[rand_w+patch_height//2], lens_position]
                pts = self.sample_points(ndims, num_points=self.N//2+1)
                raw_patch = torch.from_numpy(masked_image[:, rand_h:rand_h+patch_height, rand_w:rand_w+patch_width, :]).to(self.device)

                # render the image
                kern = self.inference(pts)
                convolved_raw_patch = self.convolve(kern, raw_patch)
                output[rand_h+sH:rand_h+patch_height-sH, rand_w+sW:rand_w+patch_width-sW,:] = convolved_raw_patch.detach().cpu().numpy()[0,...]

            
        return output

    def render_depth_layer(self, depth_layer, stride:int, lens_position:float=0.):
        output = np.zeros((self.height, self.width), dtype=np.float32)
        # Parameter space
        hs = torch.linspace(0., 1.25, self.height, dtype=torch.float32, device=self.device)
        ws = torch.linspace(0., 1.25, self.width, dtype=torch.float32, device=self.device)
        
        # Patch sizes
        patch_height = int((100+stride)*self.N/100)
        patch_width = int((100+stride)*self.N/100)
        sH = (self.N)//2
        sW = (self.N)//2
        size = (patch_height-self.N)//2

        # Patch based rendering
        for rand_h in range(0, self.height-patch_height-1, 2*size):
            for rand_w in range(0, self.width-patch_width-1, 2*size):
                # fetch the points and patches
                ndims = [hs[rand_h+patch_height//2]*2-1.25, ws[rand_w+patch_height//2]*2-1.25, lens_position]
                pts = self.sample_points(ndims, num_points=self.N//2+1)
                raw_patch = torch.from_numpy(depth_layer[rand_h:rand_h+patch_height, rand_w:rand_w+patch_width]).to(self.device)

                # render the image
                kern = self.inference(pts)
                # concatenate the rgb and sigma into a single tensor.
                depth_slice = raw_patch.to(torch.float32).to(self.device)
                s_h, s_w = depth_slice.shape[0:3]
                convolved = torch.nn.functional.conv2d(torch.reshape(depth_slice, (1,s_h,s_w)), torch.mean(kern, axis=-1), stride=1, padding="valid")
                convolved = torch.reshape(convolved, (s_h-self.N+1,s_w-self.N+1))
                output[rand_h+sH:rand_h+patch_height-sH, rand_w+sW:rand_w+patch_width-sW] = convolved.detach().cpu().numpy()
            
        return output

            
    def compose_layers(self, depth_path:str, n_layers:int, start_layer:int=0, near_in_focus:bool=True, save_folder_path:str=None):
        # Read depth map
        self.depth = np.float32(io.imread(depth_path))
        self.depth /= np.max(self.depth)
        
        composed = np.zeros((self.height, self.width, 3))
        digitized_layers = np.linspace(0, 1, n_layers) if near_in_focus else np.linspace(1, 0, n_layers)
        digitized_dpt = np.float32(np.digitize(self.depth, digitized_layers) - 1)

        # Mask each layer
        for i in range(n_layers):
            image = np.float32(io.imread(os.path.join(save_folder_path, self.image_name + f"_blurred_{i}.png")))
            binary_mask = (digitized_dpt == np.float32(i))
            for c in range(3):
                composed[...,c] += image[...,c] * binary_mask
        composed /= np.max(composed, axis=(0,1,2), keepdims=True)
        return composed
    
    def render_psf_array(self, n_samples:tuple, image_size:tuple, padding:int, save_folder_path=None):
        # get grid
        hs = torch.linspace(0., 1., image_size[0], dtype=torch.float32, device=self.device)
        ws = torch.linspace(0., 1., image_size[1], dtype=torch.float32, device=self.device)
        ps = torch.linspace(0., 1., 20, dtype=torch.float32, device=self.device)
        h_pos = [int((i+0.5)*image_size[0]/n_samples[0]) for i in range(n_samples[0])]
        w_pos = [int((i+0.5)*image_size[1]/n_samples[1]) for i in range(n_samples[1])]
        
        if save_folder_path == None:
            save_folder_path = os.path.join(self.image_folder, "psf")
        os.makedirs(save_folder_path, exist_ok=True)
        
        for pi, p in enumerate(ps):
            kerns = np.zeros((*n_samples, self.N, self.N, 3))
            for i, hi in enumerate(h_pos):
                for j, wi in enumerate(w_pos):
                    # Fetch the points and patches
                    ndims = [hs[hi]*2-1, ws[wi]*2-1, p*2-1]
                    pts = self.sample_points(ndims, num_points=self.N//2+1)
                    kern = self.inference(pts)
                    kern = np.squeeze(kern.cpu().detach().numpy())
                    kerns[i,j,...,0] = kern[...,0]
                    kerns[i,j,...,1] = 0.5*(kern[...,1]+kern[...,2])
                    kerns[i,j,...,2] = kern[...,3]
            kerns.resize((n_samples[0]*(self.N), n_samples[1]*(self.N), 3))
            
            io.imsave(os.path.join(save_folder_path, f"psf_{pi}.png"), (kerns*255).astype(np.uint8))
    
    def free(self):
        tcnn.free_temporary_memory()