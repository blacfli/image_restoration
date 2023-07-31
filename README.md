# Image Restoration

---

## :wrench: Dependencies and Installation

- Python == 3.10.10
- [PyTorch >= 2.0.1](https://pytorch.org/)
- Option: NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Option: Linux

## Dataset
To get dataset please download from the imagenet website https://www.image-net.org/download.php and get the ImageNet21k dataset from Winter 2021 release. Please make sure that you have 1 TB space in your computer. The second dataset is for testing, you can download it from https://github.com/dmc27/missing_pixels.

The file structure for the dataset will be as follows:
- dataset
    - original_image -> for testing dataset
    - training_image -> for training dataset

### Installation

To run the program for image restoration first you need to

1. Clone repo

    ```bash
    git clone https://github.com/blacfli/image_restoration.git
    cd 
    ```

2. Install packages

    ```bash
    pip install opencv-python tqdm scikit-learn matplotlib pandas
    ```
    or
   ```bash
   pip install -r requirement.txt

