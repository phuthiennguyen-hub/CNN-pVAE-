import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from utils.data_loader import BMP3DDataset 
from model import VAE3D                   

def load_model(model_path, input_shape, latent_dim):
    model = VAE3D(input_shape=input_shape, latent_dim=latent_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def encode_dataset(model_path, data_path, latent_dim, output_csv="z.csv", batch_size=1,
                   properties_csv=None, index_column="sample_id", target_properties=None):
    dataset = BMP3DDataset(
        root_directory=data_path,
        properties_csv=properties_csv,
        index_column=index_column,
        target_properties=target_properties
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Infer input shape
    first_batch, _ = next(iter(loader))  # unpack (data, target)
    input_shape = first_batch.shape[1:]  # [1, D, H, W]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(model_path, input_shape, latent_dim).to(device)
    model.eval()

    latents = []

    with torch.no_grad():
        for data, _ in loader: # unpack (data, target)
            data = data.to(device)
            mu, _ = model.encode(data)[:2]
            latents.append(mu.cpu().numpy().squeeze())

    latents = np.vstack(latents)
    pd.DataFrame(latents).to_csv(output_csv, index=False)
    print(f"Saved latent vectors to {output_csv} (shape: {latents.shape})")

if __name__ == "__main__":
    #project_root = os.path.dirname(os.path.dirname(os.getcwd()))
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data/data3D100")
    properties_csv = os.path.join(project_root, "data/data3D100.csv")
    model_path = os.path.join(project_root, "src", "trained_models", "pvae_model_3d100.pth")
    latent_dim = 250
    target_properties = ["Permeability_x", "Permeability_y", "Permeability_z"]

    encode_dataset(
        model_path=model_path,
        data_path=data_path,
        latent_dim=latent_dim,
        output_csv=os.path.join(project_root, "latent_vectors/z_3D100.csv"),
        batch_size=1,
        properties_csv=properties_csv,
        index_column="sample_id",
        target_properties=target_properties
    )
