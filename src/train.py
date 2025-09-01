import os
import torch
from torchsummary import summary
import torch.optim as optim
from model import VAE3D, VAE3D_resNet, VAE2D, VAE2D_ResNET, VAE2D_pooling
from utils.data_loader import get_data_loaders_3d, get_data_loaders_2d

'''
    Training function for a Property - Variational Autoencoder (pVAE) model.
    It includes:
    - Model architecture definition
    - Training loop with validation
    - Early stopping based on validation loss
    - Model saving
'''

class EarlyStopping:
    def __init__(self, patience=150, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

def train_vae(
    data_dir,
    epochs=1000,
    batch_size=16,
    learning_rate=1e-4,
    latent_dim=250,
    #neurons=[128, 256, 512, 128], #this for 2d
    neurons = [48, 48, 192, 128], #this for 3d
    dropout_p=0.2,
    reg_neurons=[16, 16, 4],
    indices_dir=None,
    save_indices=False,
    load_indices=False,
    #properties_csv="../PAMM_dataset.csv", # This is for 2D samples
    properties_csv="../data3D100.csv", # This is for 3D samples
    #target_properties=["n_F", "Permeability_x"], # This is for 2D properties
    target_properties=["n_F", "Permeability_x","Permeability_y","Permeability_z"], # This is for 3D properties
    index_column="sample_id"
):
    # Load the dataset and split into train/val
    train_loader, val_loader, _ = get_data_loaders_3d( # This should be changed to  get_data_loaders_2d or  get_data_loaders_3d
        data_dir,
        batch_size=batch_size,
        train_pct=0.85,
        val_pct=0.1,
        test_pct=0.05,
        seed=42,
        shuffle=True,
        indices_dir=indices_dir,
        save_indices=save_indices,
        load_indices=load_indices,
        properties_csv=properties_csv,
        target_properties=target_properties,
        index_column=index_column
    )

    # Get input shape from the first batch of train_loader
    first_batch = next(iter(train_loader))
    inputs, _ = first_batch  # unpack input and target
    input_shape = inputs.shape[1:]

    # Initialize the VAE model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE3D(input_shape=input_shape, latent_dim=latent_dim, neurons=neurons, reg_neurons=reg_neurons, dropout_p=dropout_p).to(device)
    print("Model architecture:\n", model)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Scheduler and early stopping
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=300
    )
    early_stopper = EarlyStopping(patience=150)
    
    for epoch in range(epochs):
        model.train()
        total_loss = total_recon = total_kld = total_regression = 0
    
        for sample, target_properties in train_loader:
            sample, target_properties = sample.to(device), target_properties.to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar, predicted_properties = model(sample)
            loss, recon_loss, kld, regression_loss = model.loss_function(recon_x, sample, mu, logvar, predicted_properties, target_properties)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kld += kld.item()
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
    
        scheduler.step(avg_val_loss)
        early_stopper(avg_val_loss)
    
        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {avg_train_loss:.4f} (Recon: {avg_train_recon:.4f}, KL: {avg_train_kld:.4f}, Regression: {avg_train_regression:.4f}), "
              f"Val Loss: {avg_val_loss:.4f} (Recon: {avg_val_recon:.4f}, KL: {avg_val_kld:.4f}, Regression: {avg_val_regression:.4f}),")
    
        if early_stopper.should_stop:
            print(f"Early stopping triggered after epoch {epoch + 1}")
            break

    # Save the trained model

    os.makedirs('trained_models', exist_ok=True)
    #torch.save(model.state_dict(), 'pvae_model_2dPAMM.pth') # This is for 2D case.
    torch.save(model.state_dict(), 'trained_models/pvae_model_3d100.pth') # This is for 3D case.
    
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    #data_directory = os.path.join(project_root, 'data/PAMM2025_dataset')
    data_directory = os.path.join(project_root, 'data/data3D100')
    #splits_dir = os.path.join(project_root, 'pvae_splits2DPAMM')
    splits_dir = os.path.join(project_root, 'data_splits/pvae_splits3D100')
    print("Resolved data_directory:", data_directory)
    print("Exists?", os.path.exists(data_directory))
    train_vae(
        data_directory,
        indices_dir=splits_dir,
        save_indices=True
    )