import os
import torch
import torch.optim as optim
from ray import tune
from ray import train
from model import VAE3D, VAE2D
from utils.data_loader import get_data_loaders_3d, get_data_loaders_2d

'''
    Turning function for training a Property - Variational Autoencoder (pVAE) model.
    This function is designed to be used with Ray Tune for hyperparameter tuning.
    It includes:
    - Model architecture definition
    - Training loop with validation
    - Early stopping based on validation loss
    - Model saving
'''

def train_vae_tune(config):
    # Load train and validation data
    neurons = [
        config["neurons_0"],
        config["neurons_1"],
        config["neurons_2"],
        config["neurons_3"]
    ]

    # Load train and validation data
    train_loader, val_loader, _ = get_data_loaders_2d(
        config["data_dir"],
        batch_size=config["batch_size"],
        train_pct=0.8,
        val_pct=0.2,
        test_pct=0.0,
        seed=42,
        shuffle=True,
        properties_csv="../PAMM_dataset.csv",
        target_properties=["n_F", "Permeability_x"],
        index_column="sample_id"
    )
    first_batch = next(iter(train_loader))
    inputs, _ = first_batch  # unpack input and target
    input_shape = inputs.shape[1:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE2D(
        input_shape=input_shape,
        latent_dim=config["latent_dim"],
        neurons=neurons,
        reg_neurons=[16, 16, 4],
        dropout_p=config.get("dropout_p", 0.2)
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kld = 0
        total_regression = 0
        for sample, target_properties in train_loader:
            sample, target_properties = sample.to(device), target_properties.to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar, predicted_properties = model(sample)
            loss, recon_loss, kld, regression_loss = model.loss_function(recon_x, sample, mu, logvar, predicted_properties, target_properties)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_recon += recon_loss.item() if hasattr(recon_loss, 'item') else recon_loss
            total_kld += kld.item() if hasattr(kld, 'item') else kld
            total_regression += regression_loss.item() if hasattr(regression_loss, 'item') else regression_loss

        avg_train_loss = total_loss / len(train_loader)
        avg_train_recon = total_recon / len(train_loader)
        avg_train_kld = total_kld / len(train_loader)
        avg_train_regression = total_regression / len(train_loader) if 'total_regression' in locals() else 0

        # Validation
        model.eval()
        val_loss = val_recon = val_kld = val_regression = 0
        with torch.no_grad():
            for sample, target_properties in val_loader:
                sample, target_properties = sample.to(device), target_properties.to(device)
                recon_x, mu, logvar, predicted_properties = model(sample)
                loss, recon_loss, kld, regression_loss = model.loss_function(recon_x, sample, mu, logvar, predicted_properties, target_properties)
                val_loss += loss.item()
                val_recon += recon_loss.item()
                val_kld += kld.item()
                val_regression += regression_loss.item() if hasattr(regression_loss, 'item') else regression_loss
        avg_val_loss = val_loss / len(val_loader)
        avg_val_recon = val_recon / len(val_loader)
        avg_val_kld = val_kld / len(val_loader)
        avg_val_regression = val_regression / len(val_loader) if 'val_regression' in locals() else 0

        # Print log for this epoch
        trial_dir = train.get_context().get_trial_dir()
        
        print(f"[Trial {trial_dir}] Epoch {epoch+1}/{config['epochs']} | Train Loss: {avg_train_loss:.4f} (Recon: {avg_train_recon:.4f}, KL:{avg_train_kld:.4f}, Reg:{avg_train_regression}) | Val Loss: {avg_val_loss:.4f} (Recon: {avg_val_recon:.4f}, KL: {avg_val_kld:.4f}), Reg: {avg_val_regression}")
        
        # Report validation loss to Ray Tune
        tune.report({
            "val_loss": avg_val_loss,
            "train_loss": avg_train_loss,
            "val_recon_loss": avg_val_recon,
            "val_kl_loss": avg_val_kld,
            "train_recon_loss": avg_train_recon,
            "train_kl_loss": avg_train_kld,
            "train_reg_loss": avg_train_regression,
            "val_reg_loss": avg_val_regression
        })

search_space = {
    "data_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data/PAMM2025_dataset")),
    "epochs": 30,
    "batch_size": tune.choice([32]),
    "lr": tune.loguniform(1e-4, 1e-3),
    "latent_dim": tune.qrandint(200, 600, 10),
    "neurons_0": tune.qrandint(16, 256, 8),   # 16, 32, 48, 64
    "neurons_1": tune.qrandint(16, 256, 8),  # 32, 64, 96, 128
    "neurons_2": tune.qrandint(64, 512, 8),  # 64, 128, 192, 256
    "neurons_3": tune.qrandint(64, 512, 8),# 128, 256, 384, 512
}

if __name__ == "__main__":
    tune.run(
        train_vae_tune,
        config=search_space,
        storage_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ray_tunes/test_ray_results2d")),
        resources_per_trial={"cpu": 2, "gpu": 1},
        num_samples=50,
    )