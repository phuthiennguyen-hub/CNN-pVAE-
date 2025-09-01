import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from utils.data_loader import BMP3DDataset
from model import VAE3D
import matplotlib.pyplot as plt

'''
    Evaluate a trained 2D Variational Autoencoder (VAE) model.
    This script includes:
    - Loading the model
    - Generating reconstructions from the test set
    - Calculating the loss on the test set
'''

def plot_3d_reconstructions(inputs, recons, n=10, save=False):
    """
    Plots center slices of 3D volumes (original vs. reconstructed).
    Assumes tensors of shape [1, D, H, W].
    """
    # Save directory in same directory as script
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_results")
    os.makedirs(save_dir, exist_ok=True)

    for i in range(min(n, len(recons))):
        input_vol = inputs[i].squeeze().numpy()   # shape: [D, H, W]
        recon_vol = recons[i].squeeze().numpy()

        center_slice = input_vol.shape[0] // 2
        input_slice = input_vol[center_slice]
        recon_slice = recon_vol[center_slice]

        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(input_slice, cmap='gray')
        plt.title("Original Slice")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(recon_slice, cmap='gray')
        plt.title("Reconstructed Slice")
        plt.axis('off')

        plt.tight_layout()
        if save:
            # Save in evaluation_results directory
            save_path = os.path.join(save_dir, f"recon_3d_val_{i}.png")
            plt.savefig(save_path, dpi=150)
        else:
            plt.show()
        plt.close()

def load_model(model_path, input_shape, latent_dim=16):
    model = VAE3D(input_shape=input_shape, latent_dim=latent_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def generate_reconstructions(model, data_loader, device):
    inputs = []
    reconstructions = []
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            recon, _, _, _ = model(data)
            reconstructions.append(recon.cpu())
            inputs.append(data.cpu())
    return inputs, reconstructions

def calculate_loss(model, data_loader, device):
    total_loss = 0
    total_recon_loss = 0
    total_reg_loss = 0
    total_kld = 0

    with torch.no_grad():
        for data, target_properties in data_loader:
            data, target_properties = data.to(device), target_properties.to(device)
            recon, mu, logvar, predicted_properties = model(data)
            loss, recon_loss, kld, reg_loss = model.loss_function(
                recon, data, mu, logvar,
                predicted_properties=predicted_properties,
                target_properties=target_properties
            )
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_reg_loss += reg_loss.item()
            total_kld += kld.item()

    num_batches = len(data_loader)
    return {
        'total': total_loss / num_batches,
        'reconstruction': total_recon_loss / num_batches,
        'regression': total_reg_loss / num_batches,
        'kld': total_kld / num_batches
    }

def evaluate(model_path, test_loader, latent_dim=250):
    first_batch, _ = next(iter(test_loader))
    input_shape = first_batch.shape[1:]  # [1, D, H, W]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, input_shape, latent_dim).to(device)

    inputs, reconstructions = generate_reconstructions(model, test_loader, device)
    losses = calculate_loss(model, test_loader, device)

    print("3D Evaluation Losses:")
    print(f"  Total Loss       : {losses['total']:.4f}")
    print(f"  Reconstruction   : {losses['reconstruction']:.4f}")
    print(f"  Regression       : {losses['regression']:.4f}")
    print(f"  KL Divergence    : {losses['kld']:.4f}")

    plot_3d_reconstructions(inputs, reconstructions, n=10, save=True)
    return reconstructions, losses

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "src", "trained_models", "pvae_model_3d100.pth")
    #model_path = "vae_model_3d100.pth"

    #project_root = os.path.dirname(os.path.dirname(os.getcwd()))
    data_path = os.path.join(project_root, "data/data3D100")
    test_indices_path = os.path.join(project_root, "data_splits", "splits3D", "val_indices.npy")
    properties_csv = os.path.join(project_root, "data/data3D100.csv")
    target_properties = ["n_F", "Permeability_x", "Permeability_y", "Permeability_z"]

    test_indices = np.load(test_indices_path)
    subset_indices = test_indices[:100]

    test_dataset = BMP3DDataset(
        root_directory=data_path,
        properties_csv=properties_csv,
        target_properties=target_properties,
        index_column="sample_id"
    )
    test_set = Subset(test_dataset, subset_indices)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    evaluate(model_path, test_loader)

