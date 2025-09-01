import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from model import VAE3D

'''
    Inverse optimization script for a trained Property - Variational Autoencoder (VAE) model.
    This script includes:
    - Loading the model and latent vectors
    - Optimizing a latent vector to match a target property set
    - Saving the optimal latent vector and target properties to a CSV file
    - Example of optimizing for a single target property set
    - Uses PyTorch for model and optimization
    - Uses Pandas for CSV handling
    - Uses NumPy for numerical operations
'''


# Config
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(project_root, "src", "trained_models","pvae_model_3d100.pth")
z_csv_path = os.path.join(project_root, "latent_vectors", "z_3D100.csv")
properties_path = os.path.join(project_root, "data", "data3D100.csv")
output_csv_path = os.path.join(project_root, "inverse_results", "optimal_z_progress.csv")
latent_dim = 250
epochs = 2000
lr = 0.005
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load latent vectors of the dataset
z_df = pd.read_csv(z_csv_path)
z_values = torch.tensor(z_df.values, dtype=torch.float32).to(device)

# Load properties
df_properties = pd.read_csv(properties_path)
property_tensor = torch.tensor(df_properties[["n_F","Permeability_x", "Permeability_y", "Permeability_z"]].values, dtype=torch.float32).to(device)

# Load VAE model
pvae = VAE3D(input_shape=(1, 100, 100, 100), latent_dim=latent_dim).to(device)
pvae.load_state_dict(torch.load(model_path, map_location=device))
pvae.eval()

# Define single target property set manually (example values)
target_nF = 0.7
target_perm_x = 2.0
target_perm_y = 2.0
target_perm_z = 2.0

# Initialize result CSV
if not os.path.exists(output_csv_path):
    pd.DataFrame(columns=["sample_id", "Target n_F", "Target Permeability_x", "Target Permeability_y", "Target Permeability_z", *[f"z_{i}" for i in range(latent_dim)]])\
        .to_csv(output_csv_path, index=False)

# Optimization function updated for 4 targets
def optimize_latent_vector(target_nF, target_perm_x, target_perm_y, target_perm_z):
    target = torch.tensor([target_nF, target_perm_x, target_perm_y, target_perm_z], dtype=torch.float32).to(device)

    # Use nearest z as starting point
    mse = torch.mean((property_tensor - target) ** 2, dim=1)
    start_idx = torch.argmin(mse)
    z_init = z_values[start_idx].clone().detach().unsqueeze(0).to(device)

    z_opt = torch.nn.Parameter(z_init.clone())
    optimizer = torch.optim.Adam([z_opt], lr=lr)

    best_loss = float("inf")
    best_z = None

    for epoch in range(epochs):
        optimizer.zero_grad()
        decoded = pvae.decode(z_opt)
        re_encoded_mu, _ = pvae.encode(decoded)
        prediction = pvae.regressor(re_encoded_mu)
        loss = nn.MSELoss()(prediction, target.unsqueeze(0))
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_z = z_opt.detach().clone()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    return best_z.cpu().numpy()

print(f"\nOptimizing for single target case...")

z_star = optimize_latent_vector(target_nF, target_perm_x, target_perm_y, target_perm_z)

row_data = [1, target_nF, target_perm_x, target_perm_y, target_perm_z, *z_star.flatten()]
pd.DataFrame([row_data], columns=["sample_id", "Target n_F", "Target Permeability_x", "Target Permeability_y", "Target Permeability_z", *[f"z_{i}" for i in range(latent_dim)]])\
    .to_csv(output_csv_path, mode='a', header=False, index=False)

print(f"Saved optimal latent vector for single target case.")
print(f"\nInverse optimization complete. Results saved to: {output_csv_path}")
