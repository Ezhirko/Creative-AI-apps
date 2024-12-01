# Neural Style Transfer for MRI Images using CycleGAN: T1 to T2 Conversion

## Project Overview

This project utilizes **Cyclic Generative Adversarial Networks (CycleGAN)** to perform **Neural Style Transfer** for converting **T1 weighted MRI images** into **T2 weighted MRI images** and vice versa. The core principle of CycleGAN is to learn a mapping between unpaired datasets while preserving anatomical details through cyclic consistency. This ensures that converted images can be reverted back to their original domain, facilitating accurate image translation between T1 and T2 MRI scans.

CycleGANs consist of two neural networks:
- **Generator**: Transforms images from one domain (e.g., T1) to another (e.g., T2).
- **Discriminator**: Ensures the generated images are realistic and resemble the target domain.

The goal of this project is to use CycleGANs to enable more accurate and consistent translation of MRI images, enhancing medical imaging research and applications.

## Data Collection

The dataset used for this project is provided by the university and contains unpaired T1 and T2 MRI images. 

- **T1 Dataset**: 43 images, each of size 181 x 217.
- **T2 Dataset**: 46 images, each of size 181 x 217.
  
Both datasets contain 3-channel RGB images. Since the dataset is small and unpaired, all images are used for training. There is no separate split for training and testing in this case.

## Project Structure

- **Model**: The CycleGAN model is implemented in Python using TensorFlow/Keras and is used to convert between T1 and T2 weighted MRI images.
- **Training**: The model is trained using the unpaired T1 and T2 MRI images.
- **Results**: The resulting model performs the transformation from T1 to T2 and vice versa, maintaining the anatomical consistency across the conversions.

## Key Concepts

- **CycleGAN**: A deep learning architecture that performs image-to-image translation without paired data. It uses two neural networks — the generator and the discriminator — to learn to map between different domains.
- **Cyclic Consistency**: A key property of CycleGAN that ensures that converting an image to the target domain and back to the original domain results in an image similar to the original, preserving key features.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Other dependencies mentioned in the `requirements.txt` file

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/Neural-Style-Transfer-MRI-CycleGAN.git
   cd Neural-Style-Transfer-MRI-CycleGAN
