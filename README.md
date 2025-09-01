# Variational Autoencoder for 3D BMP Images

This project implements a Variational Autoencoder (VAE) designed to process and analyze 3D BMP images. The VAE architecture is capable of learning a compressed representation of the input images, allowing for efficient reconstruction and generation of new samples.

## Project Structure

```
vae-3d-bmp
├── src
│   ├── model.py           # Defines the VAE architecture
│   ├── train.py           # Contains the training loop for the VAE
│   ├── train_vae_tune.py  # Contains the ray tune framework for hyperparameter tuning
│   ├── evaluate.py        # Evaluates the performance of the trained VAE
│   ├── latent_extract.py  # Extract the latent space
│   ├── interpolation.py   # The sphearical interpolation (Slerp)
│   ├── inverse.py         # Inverse process with target properties
│   └── utils
│       └── data_loader.py # Utility functions for loading and preprocessing data
├── data
│   ├── data3d                 # Directory containing 3D BMP images 150x150x150
│   ├── data3D150.csv          # Effective Properties for 3D BMP images 150x150x150
│   ├── data3D100              # Directory containing 3D BMP images 100x100x100
│   ├── data3D100.csv          # Effective Properties for 3D BMP images 100x100x100
│   ├── data2d                 # Directory containing 2D BMP images. Each 200 BMP for each 3D from data3d
│   ├── PAMM2025_dataset       # Directory containing 2D BMP images, which are artificial simple straight pipe
│   ├── PAMM_dataset.csv       # Effective Properties for 2D BMP images, which are artificial simple straight pipe
├── requirements.txt       # Lists the required Python dependencies and data information
├── .gitignore             # Specifies files to be ignored by Git
└── README.md              # Documentation for the project
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd vae-3d-bmp
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Place your 3D BMP images in the `data3d` directory.

## Usage

To train the VAE, run the following command:
```
python src/train.py
```

After training, you can evaluate the model using:
```
python src/evaluate.py
```

## Model Overview

The VAE consists of an encoder and a decoder. The encoder compresses the input images into a latent space, while the decoder reconstructs the images from this latent representation. The model is trained to minimize the reconstruction loss and the Kullback-Leibler divergence between the learned latent distribution and a prior distribution.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.