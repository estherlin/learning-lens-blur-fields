# Learning Lens Blur Fields

>   **Note:** This branch contains development code that is not yet cleaned up and is very messy.
>
>   

This repository contains the official implementation of [Learning Lens Blur Fields](https://arxiv.org/abs/2310.11535), published in IEEE TPAMI 2025. This work introduces a high-dimensional neural representation of blur—the lens blur field—and a practical method for acquiring it from real-world camera systems.

If you use this code or find our work useful, please cite our paper:

```bibtex
@article{lin2025learning,
        title={Learning Lens Blur Fields},
        ISSN={1939-3539},
        url={http://dx.doi.org/10.1109/TPAMI.2025.3578587},
        DOI={10.1109/tpami.2025.3578587},
        journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
        publisher={Institute of Electrical and Electronics Engineers (IEEE)},
        author={Lin, Esther Y. H. and Wang, Zhecheng and Lin, Rebecca and Miau, Daniel and Kainz, Florian and Chen, Jiawen and Zhang, Xuaner and Lindell, David B. and Kutulakos, Kiriakos N.},
        year={2025},
        pages={1–12} }
```

## Directory Structure

```
learning-lens-blur-fields/
├── config/
│   └── iphone12pro-wide.json          # MLP configuration for iPhone 12 Pro
├── data/                              # Folder for unprocessed and training data
├── imgs/                              # Example images for testing
│   ├── lantern_depth.png
│   ├── lantern.exr
│   └── method.png
├── models/                            # Pre-trained blur field models
│   ├── iphone12pro0-wide.pth         # iPhone 12 Pro model 0 from paper
│   └── iphone12pro1-wide.pth         # iPhone 12 Pro model 1 from paper
├── notebooks/                         # Jupyter notebooks for demos
│   ├── preprocess_data.ipynb         # Demo of preprocessing pipeline
│   ├── rendering_with_blur_fields.ipynb  # How to render from trained models
│   └── visualizing_blur_fields.ipynb # How to extract PSFs after training
├── run/                              # Training scripts
│   └── train_iphone12pro_wide.sh     # Training script for iPhone 12 Pro wide
├── scripts/                          # Utility scripts
│   ├── __init__.py
│   └── preprocess.py                 # Preprocessing utilities
├── util/                             # Core utilities
│   ├── __init__.py
│   ├── generate_random_pattern.py    # Generates random binary noise images
│   ├── preprocess.py                 # Preprocessing functions
│   ├── renderer.py                   # Rendering utilities
│   └── utils.py                      # Miscellaneous helper functions
├── train.py                          # Main training script for 5D blur field
├── requirements.txt                  # Python package requirements
├── LICENSE                           # MIT License
└── README.md                         
```

## Installation

### Prerequisites

This code requires tiny-cuda-nn. See [NVlabs/tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) for installation instructions (we used version 1.6). Package requirements as of October 2023 are in `requirements.txt`.

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/learning-lens-blur-fields.git
   cd learning-lens-blur-fields
   ```

2. **Create a conda environment:**
   ```bash
   conda create --name blur-fields python=3.10
   conda activate blur-fields
   ```

3. **Install PyTorch:**
   ```bash
   # For CUDA 11.8 (adjust version as needed)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Install tiny-cuda-nn:**
   ```bash
   # Install tiny-cuda-nn (required for fast MLP training)
   pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
   ```

5. **Install other requirements:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Data Preprocessing

Place all downloaded data in the `data/` folder.

#### Option 1: Using Your Own Data Captures

To generate your own noise patterns, run `util/generate_random_pattern.py`. 

Next, see `notebooks/preprocess_data.ipynb` for a detailed demonstration of the preprocessing pipeline, including:
- Calibration pattern detection
- Homography estimation
- PSF extraction
- Data preparation for training

#### Option 2: Using Released Data

You can download data for both iPhone 12 Pro wide cameras used in the paper [here](https://drive.google.com/drive/folders/1zf2p2Bj_Jxhq4-smq1AsnEwlRfKJH0qC?usp=sharing). We include:

1. **4x downsampled processed iPhone 12 Pro wide lens data** (1.06GB): Ready for immediate use in Colab demos
2. **Full resolution processed iPhone 12 Pro wide lens data** (18.3GB): For high-quality results
3. **Unprocessed calibration pattern captures** (1.75GB): For demonstrating the preprocessing pipeline

Place all downloaded data in the `data/` folder.

### Step 2: Training a Blur Field

1. **Prepare your data** following the preprocessing notebook
2. **Configure training parameters** in `config/iphone12pro-wide.json`
3. **Run training:**
   ```bash
   bash run/train_iphone12pro_wide.sh
   ```

Training configurations for lenses in the paper are specified in JSON files in the `config/` directory.

### Step 3: Rendering with a Blur Field

See `notebooks/rendering_with_blur_fields.ipynb` for examples of:
- Loading pre-trained blur field models
- Rendering images with learned blur
- Extracting PSFs at different positions

Blur field models for the lenses shown in the paper can also be found in `models/`. 