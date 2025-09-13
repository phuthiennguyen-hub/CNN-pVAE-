## Property - Variational Autoencoder (pVAE) for inverse design

The pVAE framework is trained on two datasets: a synthetic dataset of artificial porous microstructures and CT-scan images of volume elements from real open-cell foams. The encoder-decoder architecture of the VAE captures key microstructural features, mapping them into a compact and interpretable latent space for efficient structure-property exploration. The study provides a detailed analysis and interpretation of the latent space, demonstrating its role in structure-property mapping, interpolation, and inverse design. This approach facilitates the generation of new metamaterials with desired properties

## Design Space Construction

1. Microstructures are represented in voxelized form (binary images).  

2. The effective properties are:  
   - **Porosity** $n^F$  
   - **Intrinsic permeability tensor** $\mathbf{K}^{S}$  

### Porosity
The porosity can be calculated directly from the volume fraction of the pore phase in the binary image:

$$
n^F = \frac{V_\text{pore}}{V_\text{total}}
$$

### Intrinsic Permeability
Estimating the intrinsic permeability tensor $\mathbf{K}^S$ is more challenging and computationally expensive.  

- A **mesh-based approach** can be used from the 3D binary image.  
- The **Lattice Boltzmann Method (LBM)** discretizes the pore space in velocity space and solves the **Navier–Stokes equations** in a stochastic sense with a collision operator.  
- From the LBM, we obtain the microscopic **velocity field** $\mathbf{u}(\mathbf{x})$.  

<p align="center">
  <img src="assets/lbm_illustraion.jpg" width="400">
</p>

The homogenized velocity is computed by volume averaging:

$$
\langle \mathbf{u} \rangle = \frac{1}{V_\text{pore}} \int_{V_\text{pore}} \mathbf{u}(\mathbf{x}) \, dV
$$

Applying **Darcy’s law**, the intrinsic permeability tensor is obtained as:

$$
\langle \mathbf{u} \rangle = -\frac{\mathbf{K}^S}{\mu} \nabla p
$$

where:
- $\mu$ is the dynamic viscosity,
- $\nabla p$ is the macroscopic pressure gradient,
- $\mathbf{K}^S$ is the intrinsic permeability tensor.  

Thus, $\mathbf{K}^S$ can be determined from the relationship between the averaged velocity and the applied pressure gradient.  

---

**Reference:**  
Nguyen Thien Phu, Uwe Navrath, Yousef Heider, Julaluk Carmai, Bernd Markert.  
*Investigating the impact of deformation on foam permeability through CT scans and the Lattice–Boltzmann method*.  
PAMM, 2023. [https://doi.org/10.1002/pamm.202300154]

## Probability-variational autoencoder (pVAE)

The Variational Autoencoder (VAE), introduced in 2013, is one of the most influential generative models.  
It combines **deep learning** with **probabilistic inference**, enabling the mapping between high-dimensional data and a structured latent space.  

For more details, see: Kingma & Welling (2013), *Auto-Encoding Variational Bayes*.  

---

### 1. KL Divergence

The **Kullback–Leibler (KL) divergence** measures how one probability distribution differs from another:

$$
D_\text{KL}(q(z) \,\|\, p(z)) = \int q(z) \log \frac{q(z)}{p(z)} \, dz
$$

It acts as a penalty term in VAE training, ensuring that the approximate posterior $q_\phi(z|x)$ stays close to the prior $p(z)$.

---

### 2. Variational Inference

Direct computation of the posterior $p(z|x)$ is often intractable.  
Instead, VAE introduces a **variational distribution** $q_\phi(z|x)$ parameterized by a neural network (the encoder).  

The true log-likelihood is:

$$
\log p_\theta(x) = \log \int p_\theta(x|z)p(z)\,dz
$$

Since this integral is intractable, we approximate it using **variational inference**, where the encoder learns $q_\phi(z|x)$.

---

### 3. Driving the ELBO Loss

The **Evidence Lower Bound (ELBO)** is maximized to train the VAE:

$$
\log p_\theta(x) \geq 
\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_\text{KL}\big(q_\phi(z|x)\,\|\,p(z)\big)
$$

The ELBO has two competing terms:  
- **Reconstruction term**: encourages accurate data reconstruction via the decoder $p_\theta(x|z)$  
- **Regularization term**: forces $q_\phi(z|x)$ to be close to the prior $p(z)$  

Thus, the VAE balances reconstruction quality with latent space regularization.

---

### 4. KL Divergence for Standard Normal

When the prior is chosen as a **standard normal distribution** $p(z) = \mathcal{N}(0, I)$ and the encoder outputs a Gaussian $q_\phi(z|x) = \mathcal{N}(\mu, \sigma^2 I)$, the KL divergence has a closed form:

$$
D_\text{KL}\big(\mathcal{N}(\mu, \sigma^2 I)\,\|\,\mathcal{N}(0, I)\big) 
= \tfrac{1}{2} \sum_{i=1}^d \left( \mu_i^2 + \sigma_i^2 - \log \sigma_i^2 - 1 \right)
$$

This term is added to the reconstruction loss to form the full VAE objective.

---

## Copyright

 © Prof. Dr.-Ing. habil. Fadi Aldakheel

 Leibniz Universität Hannover 

 Faculty of Civil Engineering and Geodetic Science 

 Institut für Baumechanik und Numerische Mechanik (IBNM)

 https://www.ibnm.uni-hannover.de
 
 Coded by Phu Thien Nguyen with the help of Copilot :smiley:

 Paper: Deep learning-aided inverse design of porous metamaterials
 The authors are:
 Phu Thien Nguyen, Yousef Heider, Dennis Kochmann, Fadi Aldakheel

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
│   ├── data3D150          # Directory containing 3D BMP images 100x100x100
│   ├── data3D150.csv      # Effective Properties for 3D BMP images 100x100x100
│   ├── syn-data.ipynb     # Notebook for creating synthetic data (100x100x100), random pore position is uniform distribution
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

The pVAE consists of a variational autoencoder (VAE) and a regressor. The VAE includes an encoder that compresses input images into a latent space and a decoder that reconstructs images from this latent representation. In addition to minimizing the reconstruction loss and the Kullback-Leibler divergence, the latent space is used by a regressor to predict effective material properties directly from the encoded representations. This joint framework enables both image reconstruction and property prediction, facilitating structure-property mapping and inverse design.

**Reference:**  
[1]. P.T. Nguyen, Y. Heider, D. Kochmann, and F. Aldakheel, Deep learning-aided inverse design of porous metamaterials, CMAME, (2025)

[2]. P.T. Nguyen, B.-e. Ayouch, Y. Heider, and F. Aldakheel, Impact of Dataset Size and Hyperparameters Tuning in a VAE for Structure-Property Mapping in Porous Metamaterials, PAMM, (2025)

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
