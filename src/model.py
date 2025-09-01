import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    This module contains the implementation of a 3D Variational Autoencoder (VAE) with ResNet blocks.
    It includes:   
    - Pre-Activation ResNet blocks for both 2D and 3D
    - VAE architecture with encoder, decoder, and regressor
    - Loss function that combines reconstruction loss, KL divergence, and regression loss
    - Dynamic computation of flattened dimensions after convolutional layers
    - Support for dropout in both encoder and decoder
    - Support for regression tasks in addition to reconstruction
    - Compatibility with both 2D and 3D input shapes
    - Ability to handle different latent dimensions and neuron configurations
    - Support for early stopping and learning rate scheduling during training
    - Evaluation functions for generating reconstructions and calculating losses
    - Visualization of input and reconstructed slices
    - Support for saving and loading model weights
    - Compatibility with PyTorch's DataLoader for batching and shuffling data
    - Support for binary cross-entropy loss for reconstruction
    - Support for mean squared error loss for regression tasks
    - Support for dynamic input shapes, allowing flexibility in model architecture
    - Support for dropout layers to prevent overfitting
    - Support for both 2D and 3D convolutional layers
    - Support for both 2D and 3D transposed convolutional layers
    - Support for both 2D and 3D batch normalization
    - Support for both 2D and 3D max pooling layers
    - Support for both 2D and 3D upsampling layers
    - Support for both 2D and 3D activation functions (ReLU, Sigmoid, etc.)
'''

# 3D Pre-Activation ResNet Block (ResNet v2)
class PreActResNetBlock3D(nn.Module):
    def __init__(self, c_in, c_out=None, subsample=False, act_fn=nn.ReLU, dropout_p=0.2):
        super().__init__()
        if c_out is None:
            c_out = c_in
        stride = 2 if subsample else 1
        self.bn1 = nn.BatchNorm3d(c_in)
        self.act1 = act_fn()
        self.dropout1 = nn.Dropout3d(p=dropout_p)
        self.conv1 = nn.Conv3d(c_in, c_out, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm3d(c_out)
        self.act2 = act_fn()
        self.dropout2 = nn.Dropout3d(p=dropout_p)
        self.conv2 = nn.Conv3d(c_out, c_out, kernel_size=3, padding=1, bias=False)
        self.downsample = nn.Sequential(
            nn.BatchNorm3d(c_in),
            act_fn(),
            nn.Conv3d(c_in, c_out, kernel_size=1, stride=stride)
        ) if (subsample or c_in != c_out) else None

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.act1(out)
        out = self.dropout1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.dropout2(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        return out

# 2D Pre-Activation ResNet Block (ResNet v2)
class PreActResNetBlock2D(nn.Module):
    def __init__(self, c_in, c_out=None, subsample=False, act_fn=nn.ReLU):
        super().__init__()
        if c_out is None:
            c_out = c_in
        stride = 2 if subsample else 1

        self.bn1 = nn.BatchNorm2d(c_in)
        self.act1 = act_fn()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(c_out)
        self.act2 = act_fn()
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False)

        self.downsample = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_fn(),
            nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False)
        ) if (subsample or c_in != c_out) else None

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.act1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        return out

# 3D  VAE (ResNet v2)
class VAE3D_resNet(nn.Module):
    def __init__(self, input_shape, latent_dim, neurons=[48, 48, 192, 128], reg_neurons=[16,16,4], dropout_p=0.2):
        super(VAE3D_resNet, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.neurons = neurons
        self.dropout = nn.Dropout(p=dropout_p)
        self.reg_neurons = reg_neurons
        

        # Encoder with Pre-Activation ResNet blocks
        self.enc_block1 = PreActResNetBlock3D(1, neurons[0], subsample=True, dropout_p=dropout_p)
        self.enc_block2 = PreActResNetBlock3D(neurons[0], neurons[1], subsample=True, dropout_p=dropout_p)
        self.enc_block3 = PreActResNetBlock3D(neurons[1], neurons[2], subsample=True, dropout_p=dropout_p)
        self.enc_block4 = PreActResNetBlock3D(neurons[2], neurons[3], subsample=True, dropout_p=dropout_p)
        self.flatten = nn.Flatten()

        # Dynamically compute the flattened size after conv layers
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            x = self.enc_block1(dummy)
            x = self.enc_block2(x)
            x = self.enc_block3(x)
            x = self.enc_block4(x)
            flat_dim = x.numel() // x.shape[0]
            self._enc_out_shape = x.shape[1:]

        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        # ------ Regressor ------
        reg_layers = []
        in_features = latent_dim
        for out_features in reg_neurons:
            reg_layers.append(nn.Linear(in_features, out_features))
            reg_layers.append(nn.ReLU())
            reg_layers.append(nn.Dropout(p=0.1))
            in_features = out_features
        reg_layers.append(nn.Linear(in_features, 4))
        self.regressor_mlp = nn.Sequential(*reg_layers)

        # Decoder (unchanged)
        self.fc_dec = nn.Linear(latent_dim, flat_dim)
        self.dropout_dec = nn.Dropout(p=dropout_p)
        self.dec_deconv1 = nn.ConvTranspose3d(neurons[3], neurons[2], kernel_size=3, stride=2, padding=1)
        self.dec_deconv2 = nn.ConvTranspose3d(neurons[2], neurons[1], kernel_size=3, stride=2, padding=1) # output_padding=1 for 150 images
        self.dec_deconv3 = nn.ConvTranspose3d(neurons[1], neurons[0], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_deconv4 = nn.ConvTranspose3d(neurons[0], 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def encode(self, x):
        x = self.enc_block1(x)
        x = self.enc_block2(x)
        x = self.enc_block3(x)
        x = self.enc_block4(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = F.relu(self.fc_dec(z))
        x = self.dropout_dec(x)
        x = x.view(-1, *self._enc_out_shape)
        x = F.relu(self.dec_deconv1(x))
        x = F.relu(self.dec_deconv2(x))
        x = F.relu(self.dec_deconv3(x))
        x = torch.sigmoid(self.dec_deconv4(x))
        return x

    def regressor(self, mu):
        return self.regressor_mlp(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        predicted_properties = self.regressor(mu)
        return recon_x, mu, logvar, predicted_properties

    def loss_function(self, recon_x, x, mu, logvar, predicted_properties=None, target_properties=None, beta=1.0, alpha=1.0):
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + beta * kld

        if predicted_properties is not None and target_properties is not None:
            regression_loss = F.mse_loss(predicted_properties, target_properties, reduction='sum')
            total_loss += alpha * regression_loss
        else: 
            regression_loss = torch.tensor(0.0, device = x.device)

        return total_loss, recon_loss, kld, regression_loss

class VAE3D(nn.Module):
    def __init__(self, input_shape, latent_dim, neurons=[48, 48, 192, 128], reg_neurons=[16,16,4], dropout_p=0.2):
        super(VAE3D, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.neurons = neurons
        self.dropout_p = dropout_p
        self.reg_neurons = reg_neurons

        # --------- Encoder ---------
        self.enc_conv1 = nn.Conv3d(1, neurons[0], kernel_size=3, stride=2, padding=1)  # Downsample
        self.enc_dropout1 = nn.Dropout3d(p=dropout_p)
        self.enc_conv2 = nn.Conv3d(neurons[0], neurons[1], kernel_size=3, stride=2, padding=1)
        self.enc_dropout2 = nn.Dropout3d(p=dropout_p)
        self.enc_conv3 = nn.Conv3d(neurons[1], neurons[2], kernel_size=3, stride=2, padding=1)
        self.enc_dropout3 = nn.Dropout3d(p=dropout_p)
        self.enc_conv4 = nn.Conv3d(neurons[2], neurons[3], kernel_size=3, stride=2, padding=1)
        self.enc_dropout4 = nn.Dropout3d(p=dropout_p)
        self.flatten = nn.Flatten()

        # Compute flattened dimension dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            x = F.relu(self.enc_dropout1(self.enc_conv1(dummy)))
            x = F.relu(self.enc_dropout2(self.enc_conv2(x)))
            x = F.relu(self.enc_dropout3(self.enc_conv3(x)))
            x = F.relu(self.enc_dropout4(self.enc_conv4(x)))
            flat_dim = x.numel() // x.shape[0]
            self._enc_out_shape = x.shape[1:]

        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        # --------- Regressor --------
        reg_layers = []
        in_features = latent_dim
        for out_features in reg_neurons:
            reg_layers.append(nn.Linear(in_features, out_features))
            reg_layers.append(nn.ReLU())
            reg_layers.append(nn.Dropout(p=0.1))
            in_features = out_features
        reg_layers.append(nn.Linear(in_features, 4))  # final output layer
        self.regressor_mlp = nn.Sequential(*reg_layers)   

        # --------- Decoder ---------
        self.fc_dec = nn.Linear(latent_dim, flat_dim)
        self.dec_deconv1 = nn.ConvTranspose3d(neurons[3], neurons[2], kernel_size=3, stride=2, padding=1)
        self.dec_dropout1 = nn.Dropout3d(p=dropout_p)
        self.dec_deconv2 = nn.ConvTranspose3d(neurons[2], neurons[1], kernel_size=3, stride=2, padding=1) # output_padding=1 for 150 images
        self.dec_dropout2 = nn.Dropout3d(p=dropout_p)
        self.dec_deconv3 = nn.ConvTranspose3d(neurons[1], neurons[0], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_dropout3 = nn.Dropout3d(p=dropout_p)
        self.dec_deconv4 = nn.ConvTranspose3d(neurons[0], 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def encode(self, x):
        x = F.relu(self.enc_dropout1(self.enc_conv1(x)))
        x = F.relu(self.enc_dropout2(self.enc_conv2(x)))
        x = F.relu(self.enc_dropout3(self.enc_conv3(x)))
        x = F.relu(self.enc_dropout4(self.enc_conv4(x)))
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = F.relu(self.fc_dec(z))
        x = x.view(-1, *self._enc_out_shape)
        x = F.relu(self.dec_dropout1(self.dec_deconv1(x)))
        x = F.relu(self.dec_dropout2(self.dec_deconv2(x)))
        x = F.relu(self.dec_dropout3(self.dec_deconv3(x)))
        x = torch.sigmoid(self.dec_deconv4(x))  # Needed for binary_cross_entropy
        return x

    def regressor(self, mu):
        return self.regressor_mlp(mu)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        predicted_properties = self.regressor(mu)
        return recon_x, mu, logvar, predicted_properties

    def loss_function(self, recon_x, x, mu, logvar, predicted_properties=None, target_properties=None, beta=1.0, alpha=1.0):
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')  # Input should be in [0, 1]
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + beta * kld

        if predicted_properties is not None and target_properties is not None:
            regression_loss = F.mse_loss(predicted_properties, target_properties, reduction='sum')
            total_loss += alpha * regression_loss
        else:
            regression_loss = torch.tensor(0.0, device=x.device)

        return total_loss, recon_loss, kld, regression_loss

# 2D VAE without ResNET, using maxPooling and Bilinear upSampling layers.
class VAE2D_pooling(nn.Module):
    def __init__(self, input_shape, latent_dim, neurons=[128, 256, 512, 128],reg_neurons = [16,16,4], dropout_p=0.2):
        super(VAE2D_pooling, self).__init__()
        self.input_shape = input_shape  # e.g., (1, H, W)
        self.latent_dim = latent_dim
        self.neurons = neurons
        self.dropout = nn.Dropout(p=dropout_p)
        self.reg_neurons = reg_neurons

        # ----- Encoder -----
        self.enc_conv1 = nn.Conv2d(1, neurons[0], kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.enc_conv2 = nn.Conv2d(neurons[0], neurons[1], kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.enc_conv3 = nn.Conv2d(neurons[1], neurons[2], kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.enc_conv4 = nn.Conv2d(neurons[2], neurons[3], kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        # Compute the flattened dimension
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            x = self.pool1(F.relu(self.enc_conv1(dummy)))
            x = self.pool2(F.relu(self.enc_conv2(x)))
            x = self.pool3(F.relu(self.enc_conv3(x)))
            x = self.pool4(F.relu(self.enc_conv4(x)))
            flat_dim = x.numel() // x.shape[0]
            self._enc_out_shape = x.shape[1:]

        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        # ----- Decoder -----
        self.fc_dec = nn.Linear(latent_dim, flat_dim)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec_conv1 = nn.Conv2d(neurons[3], neurons[2], kernel_size=3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec_conv2 = nn.Conv2d(neurons[2], neurons[1], kernel_size=3, padding=1) #(padding=2) for 150
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec_conv3 = nn.Conv2d(neurons[1], neurons[0], kernel_size=3, padding=2) #(padding=1) for 100
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec_conv4 = nn.Conv2d(neurons[0], 1, kernel_size=3, padding=1)

        # ----- Regressor -----
        reg_layers = []
        in_features = latent_dim
        for out_features in reg_neurons:
            reg_layers.append(nn.Linear(in_features, out_features))
            reg_layers.append(nn.ReLU())
            reg_layers.append(nn.Dropout(p=0.1))
            in_features = out_features
        reg_layers.append(nn.Linear(in_features, 2))  #final output layer
        self.regressor_mlp = nn.Sequential(*reg_layers)

    def encode(self, x):
        x = self.pool1(F.relu(self.dropout(self.enc_conv1(x))))
        x = self.pool2(F.relu(self.dropout(self.enc_conv2(x))))
        x = self.pool3(F.relu(self.dropout(self.enc_conv3(x))))
        x = self.pool4(F.relu(self.dropout(self.enc_conv4(x))))
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = F.relu(self.fc_dec(z))
        x = x.view(-1, *self._enc_out_shape)
        x = self.upsample1(x)
        x = F.relu(self.dropout(self.dec_conv1(x)))
        x = self.upsample2(x)
        x = F.relu(self.dropout(self.dec_conv2(x)))
        x = self.upsample3(x)
        x = F.relu(self.dropout(self.dec_conv3(x)))
        x = self.upsample4(x)
        x = torch.sigmoid(self.dec_conv4(x))
        return x
    
    def regressor(self, mu):
        return self.regressor_mlp(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        predicted_properties = self.regressor(mu)
        return recon_x, mu, logvar, predicted_properties

    def loss_function(self, recon_x, x, mu, logvar, predicted_properties=None, target_properties=None, beta=1.0, alpha=1.0):
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + beta * kld

        if predicted_properties is not None and target_properties is not None:
            regression_loss = F.mse_loss(predicted_properties, target_properties, reduction='sum')
            total_loss += alpha * regression_loss
        else:
            regression_loss = torch.tensor(0.0, device=x.device)

        return total_loss, recon_loss, kld, regression_loss


# 2D VAE without ResNET
class VAE2D(nn.Module):
    def __init__(self, input_shape, latent_dim, neurons=[32, 64, 128, 256], reg_neurons=[16,16,4], dropout_p=0.2):
        super(VAE2D, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.neurons = neurons
        self.dropout = nn.Dropout(p=dropout_p)
        self.reg_neurons = reg_neurons

        # Encoder
        self.enc_conv1 = nn.Conv2d(1, neurons[0], kernel_size=3, stride=2, padding=1)  # H/2
        self.enc_conv2 = nn.Conv2d(neurons[0], neurons[1], kernel_size=3, stride=2, padding=1)  # H/4
        self.enc_conv3 = nn.Conv2d(neurons[1], neurons[2], kernel_size=3, stride=2, padding=1)  # H/8
        self.enc_conv4 = nn.Conv2d(neurons[2], neurons[3], kernel_size=3, stride=2, padding=1)  # H/16

        # Flattening and latent vectors
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            x = self.enc_conv1(dummy)
            x = self.enc_conv2(x)
            x = self.enc_conv3(x)
            x = self.enc_conv4(x)
            flat_dim = x.numel() // x.shape[0]
            self._enc_out_shape = x.shape[1:]

        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

         # ----- Regressor -----
        reg_layers = []
        in_features = latent_dim
        for out_features in reg_neurons:
            reg_layers.append(nn.Linear(in_features, out_features))
            reg_layers.append(nn.ReLU())
            reg_layers.append(nn.Dropout(p=0.1))
            in_features = out_features
        reg_layers.append(nn.Linear(in_features, 2))  # final output layer
        self.regressor_mlp = nn.Sequential(*reg_layers)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, flat_dim)
        self.dec_deconv1 = nn.ConvTranspose2d(neurons[3], neurons[2], kernel_size=3, stride=2, padding=1, output_padding=0)
        self.dec_deconv2 = nn.ConvTranspose2d(neurons[2], neurons[1], kernel_size=3, stride=2, padding=1, output_padding=0)
        self.dec_deconv3 = nn.ConvTranspose2d(neurons[1], neurons[0], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_deconv4 = nn.ConvTranspose2d(neurons[0], 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = F.relu(self.enc_conv4(x))
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = F.relu(self.fc_dec(z))
        x = x.view(-1, *self._enc_out_shape)
        x = F.relu(self.dec_deconv1(x))
        x = F.relu(self.dec_deconv2(x))
        x = F.relu(self.dec_deconv3(x))
        x = torch.sigmoid(self.dec_deconv4(x))
        return x

    def regressor(self, mu):
        return self.regressor_mlp(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        predicted_properties = self.regressor(mu)
        return recon_x, mu, logvar, predicted_properties

    def loss_function(self, recon_x, x, mu, logvar, predicted_properties=None, target_properties=None, beta=1.0, alpha=1.0):
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + beta * kld

        if predicted_properties is not None and target_properties is not None:
            regression_loss = F.mse_loss(predicted_properties, target_properties, reduction='sum')
            total_loss += alpha * regression_loss
        else:
            regression_loss = torch.tensor(0.0, device=x.device)

        return total_loss, recon_loss, kld, regression_loss

# 2D VAE with ResNET

class VAE2D_ResNET(nn.Module):
    def __init__(self, input_shape, latent_dim, neurons=[32, 64, 128, 256], reg_neurons=[16, 16, 4], dropout_p=0.2):
        super(VAE2D_ResNET, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.neurons = neurons
        self.dropout = nn.Dropout(p=dropout_p)
        self.reg_neurons = reg_neurons

        # Encoder using PreActResNetBlock2D
        self.enc_block1 = PreActResNetBlock2D(1, neurons[0], subsample=True)
        self.enc_block2 = PreActResNetBlock2D(neurons[0], neurons[1], subsample=True)
        self.enc_block3 = PreActResNetBlock2D(neurons[1], neurons[2], subsample=True)
        self.enc_block4 = PreActResNetBlock2D(neurons[2], neurons[3], subsample=True)

        # Dynamically compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            x = self.enc_block1(dummy)
            x = self.enc_block2(x)
            x = self.enc_block3(x)
            x = self.enc_block4(x)
            flat_dim = x.numel() // x.shape[0]
            self._enc_out_shape = x.shape[1:]

        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        # ----- Regressor -----
        reg_layers = []
        in_features = latent_dim
        for out_features in reg_neurons:
            reg_layers.append(nn.Linear(in_features, out_features))
            reg_layers.append(nn.ReLU())
            reg_layers.append(nn.Dropout(p=0.1))
            in_features = out_features
        reg_layers.append(nn.Linear(in_features, 2))  # final output layer
        self.regressor_mlp = nn.Sequential(*reg_layers)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, flat_dim)
        self.dec_deconv1 = nn.ConvTranspose2d(neurons[3], neurons[2], kernel_size=3, stride=2, padding=1, output_padding=0)
        self.dec_deconv2 = nn.ConvTranspose2d(neurons[2], neurons[1], kernel_size=3, stride=2, padding=1, output_padding=0)
        self.dec_deconv3 = nn.ConvTranspose2d(neurons[1], neurons[0], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_deconv4 = nn.ConvTranspose2d(neurons[0], 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def encode(self, x):
        x = self.enc_block1(x)
        x = self.enc_block2(x)
        x = self.enc_block3(x)
        x = self.enc_block4(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = F.relu(self.fc_dec(z))
        x = x.view(-1, *self._enc_out_shape)
        x = F.relu(self.dec_deconv1(x))
        x = F.relu(self.dec_deconv2(x))
        x = F.relu(self.dec_deconv3(x))
        x = torch.sigmoid(self.dec_deconv4(x))
        return x

    def regressor(self, mu):
        return self.regressor_mlp(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        predicted_properties = self.regressor(mu)
        return recon_x, mu, logvar, predicted_properties

    def loss_function(self, recon_x, x, mu, logvar, predicted_properties=None, target_properties=None, beta=1.0, alpha=1.0):
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + beta * kld

        if predicted_properties is not None and target_properties is not None:
            regression_loss = F.mse_loss(predicted_properties, target_properties, reduction='sum')
            total_loss += alpha * regression_loss
        else:
            regression_loss = torch.tensor(0.0, device=x.device)

        return total_loss, recon_loss, kld, regression_loss