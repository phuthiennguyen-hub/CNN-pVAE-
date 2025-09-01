import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from utils.data_loader import BMP3DDataset
from model import VAE3D

'''
    Interpolation script for a trained 3D Variational Autoencoder (VAE) model.
    This script includes:
    - Loading the model
    - Interpolating between two latent vectors using Spherical Linear Interpolation (SLERP)
    - Saving the interpolated latent vectors to a CSV file
    - Saving individual slices of the reconstructed 3D volumes as BMP images
    - Plotting a grid of 2D slices from the reconstructed volumes
'''


def load_model(model_path, input_shape, latent_dim=250):
    model = VAE3D(input_shape=input_shape, latent_dim=latent_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def slerp(val, low, high):
    """Spherical linear interpolation between two vectors."""
    low_norm = low / torch.norm(low)
    high_norm = high / torch.norm(high)
    dot = torch.clamp(torch.sum(low_norm * high_norm), -1.0, 1.0)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    if sin_omega.item() < 1e-6:
        return (1.0 - val) * low + val * high  # fallback to LERP
    return (torch.sin((1.0 - val) * omega) / sin_omega) * low + (torch.sin(val * omega) / sin_omega) * high


def interpolate_latents_slerp(z1, z2, steps=10):
    return [slerp(t, z1, z2) for t in torch.linspace(0, 1, steps)]

def save_individual_slices(volumes, save_dir, slice_axis=2):
    """
    Save the center slice of each 3D volume as an individual BMP image.
    
    Args:
        volumes: list of 3D volumes (torch.Tensor or numpy array), shape (D,H,W) or (C,D,H,W)
        save_dir: directory path to save images
        slice_axis: axis along which to slice (0, 1, or 2)
    """
    os.makedirs(save_dir, exist_ok=True)

    for idx, vol in enumerate(volumes):
        vol_np = vol.squeeze().detach().cpu().numpy() if torch.is_tensor(vol) else np.array(vol)
        
        # If input shape is (C,D,H,W), remove channel dimmodel_path = os.path.join(project_root, "trained_models", "pvae_model_3d100.pth")
        if vol_np.ndim == 4 and vol_np.shape[0] == 1:
            vol_np = vol_np[0]

        mid_idx = vol_np.shape[slice_axis] // 2

        if slice_axis == 0:
            img = vol_np[mid_idx, :, :]
        elif slice_axis == 1:
            img = vol_np[:, mid_idx, :]
        else:
            img = vol_np[:, :, mid_idx]

        # Normalize the image to 0-255 and convert to uint8
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)  # avoid div by zero
        img_uint8 = (img_norm * 255).astype(np.uint8)

        # Create PIL image and save as BMP
        im = Image.fromarray(img_uint8)
        img_path = os.path.join(save_dir, f"slerp_{idx:03d}.bmp")
        im.save(img_path)

    print(f"Saved {len(volumes)} BMP images to {save_dir}")


def show_images_grid(volumes, title="", slice_axis=2, save_path="interpolation_grid"):
    """
    Save a grid of 2D slices from a list of 3D volumes as PNG and PDF.
    
    Args:
        volumes: List of 3D numpy arrays or tensors (C, D, H, W) or (D, H, W)
        title: Title for the figure
        slice_axis: Which axis to slice along (0=X, 1=Y, 2=Z)
        save_path: Path prefix to save images (without extension)
    """
    num_images = len(volumes)
    cols = min(6, num_images)
    rows = int(np.ceil(num_images / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows))
    axs = np.array(axs).reshape(-1)  # Flatten in case of 1D row

    for i, vol in enumerate(volumes):
        vol_np = vol.squeeze().detach().cpu().numpy() if torch.is_tensor(vol) else np.array(vol)
        mid_idx = vol_np.shape[slice_axis] // 2

        # Extract the 2D slice
        if slice_axis == 0:
            img = vol_np[mid_idx, :, :]
        elif slice_axis == 1:
            img = vol_np[:, mid_idx, :]
        else:
            img = vol_np[:, :, mid_idx]

        axs[i].imshow(img, cmap='gray')
        axs[i].axis('off')

    for j in range(i+1, len(axs)):
        axs[j].axis('off')

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save to file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(f"{save_path}.png", dpi=300)
    fig.savefig(f"{save_path}.pdf", dpi=300)

    plt.close(fig)
    print(f"Saved figure to {save_path}.png and .pdf")

def main():
    # ----- Configuration -----
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "src", "trained_models", "pvae_model_3d100.pth")
    #model_path = "pvae_model_3d100.pth"
    latent_dim = 250
    steps = 10
    idx1, idx2 = 10, 8010
    #project_root = os.path.dirname(os.path.dirname(os.getcwd()))
    data_path = os.path.join(project_root, "data/data3D100")
    properties_csv = os.path.join(project_root, "data/data3D100.csv")
    target_properties = ["n_F", "Permeability_x", "Permeability_y", "Permeability_z"]

    # ----- Load Dataset -----
    dataset = BMP3DDataset(
        root_directory=data_path,
        properties_csv=properties_csv,
        index_column="sample_id",
        target_properties=target_properties,
    )
    x1, _ = dataset[idx1]
    x2, _ = dataset[idx2]
    x1 = x1.unsqueeze(0)
    x2 = x2.unsqueeze(0)
    input_shape = x1.shape[1:]  # [1, D, H, W]

    # ----- Load Model -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, input_shape, latent_dim).to(device)

    
    save_dir = os.path.join(project_root, "slerp_results")
    os.makedirs(save_dir, exist_ok=True)

    # ----- Encode & SLERP -----
    with torch.no_grad():
        z1, _ = model.encode(x1.to(device))[:2]
        z2, _ = model.encode(x2.to(device))[:2]
        z_interp = interpolate_latents_slerp(z1.squeeze(), z2.squeeze(), steps=steps)

        # Save interpolated latent vectors to CSV
        csv_path = os.path.join(save_dir, "interpolated_latents.csv")
        z_interp_np = [z.cpu().numpy() for z in z_interp]
        z_interp_np = np.vstack(z_interp_np)
        np.savetxt(csv_path, z_interp_np, delimiter=",")
        print(f"Saved interpolated latent vectors to {csv_path}")

        # Decode interpolated latents
        reconstructions = [model.decode(z.unsqueeze(0).to(device)).cpu() for z in z_interp]

    # ----- Plot Grid -----
    #image_save_path = os.path.join(save_dir, "slerp_grid")
    #save_images_grid(reconstructions, title="3D Interpolation (SLERP)", save_path=image_save_path)
    save_dir = os.path.join(project_root, "slerp_results/slerp_imgs")
    save_individual_slices(reconstructions, save_dir, slice_axis=2)
    print("Interpolation complete.")

if __name__ == "__main__":
    main()