import os
import numpy as np
from PIL import Image
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset

'''
    Data loader for 2D and 3D datasets of BMP images.
    This module includes:
    - Dataset classes for 2D and 3D images
    - Functions to load data and split into train/val/test sets
    - Utility functions for handling properties CSV files
'''

# DATA_LOADER FOR 3D CASE
class BMP3DDataset(Dataset):
    def __init__(self, root_directory, transform=None, properties_csv="properties.csv", target_properties=None, index_column="sample_id"):
        self.root_directory = root_directory
        self.transform = transform
        self.sample_folders = sorted([
            os.path.join(root_directory, d)
            for d in os.listdir(root_directory)
            if os.path.isdir(os.path.join(root_directory, d)) and d.isdigit()
        ], key=lambda x: int(os.path.basename(x)))

        self.folder_names = [str(os.path.basename(folder)) for folder in self.sample_folders]

        # Load properties if available
        # If properties_csv is relative and starts with "../", go up
        if os.path.isabs(properties_csv):
            self.properties_csv_path = properties_csv
        else:
            self.properties_csv_path = os.path.normpath(os.path.join(root_directory, properties_csv))

        if not os.path.exists(self.properties_csv_path):
            raise FileNotFoundError(f"Properties CSV file not found: {self.properties_csv_path}")
        self.properties_df = pd.read_csv(self.properties_csv_path)

        if index_column not in self.properties_df.columns:
            raise ValueError(f"Index column '{index_column}' not found in properties CSV.")
        
        if target_properties is None:
            raise ValueError("target_properties must be specified as a list of column names.")
        
        if isinstance(target_properties, str):
            target_properties = [target_properties]

        for prop in target_properties:
            if prop not in self.properties_df.columns:
                raise ValueError(f"Target property '{prop}' not found in properties CSV.")
        
        self.target_properties = target_properties
        self.index_column = index_column

        # Create a dictionary: folder_name -> property value
        self.property_dict = {}
        for _, row in self.properties_df.iterrows():
            sample_id_value = str(int(row[self.index_column]))

            # Convert to string key for folder lookup
            try:
                key = str(int(float(sample_id_value)))
            except (ValueError, TypeError):
                continue

            # Build array with sample_id as the first element
            try:
                full_array = np.array(
                    row[target_properties].astype(float).tolist(),
                    dtype=np.float32
                )
                self.property_dict[key] = full_array
            except Exception as e:
                print(f"Error parsing row for sample_id {sample_id_value}: {e}")

    def __len__(self):
        return len(self.sample_folders)

    def __getitem__(self, idx):
        folder = self.sample_folders[idx]
        folder_name = self.folder_names[idx]

        slice_files = sorted(
            [f for f in os.listdir(folder) if f.endswith('.bmp')],
            key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else -1
        )
        slices = []
        for fname in slice_files:
            img_path = os.path.join(folder, fname)
            img = Image.open(img_path).convert('L')
            img = np.array(img, dtype=np.float32) / 255.0
            slices.append(img)

        volume = np.stack(slices, axis=0)
        if self.transform:
            volume = self.transform(volume)
        volume = torch.from_numpy(volume).unsqueeze(0) # Add channel dimension: [1, D, H, W]

        prop = self.property_dict.get(folder_name)
        if prop is None:
            raise ValueError(f"Target property/ies missing for folder '{folder_name}' in properties CSV.")
        
        # Convert to numpy array explicitly for safe nan check
        prop_array = np.array(prop, dtype=np.float32)
        
        if np.isnan(prop_array).any():
            raise ValueError(f"Target property/ies contain NaNs for folder '{folder_name}' in properties CSV.")
        
        target_value = torch.tensor(prop_array, dtype=torch.float32)

        return volume, target_value

# DATA_LOADER FOR 2D CASE
class BMP2DDataset(Dataset):
    def __init__(self, root_directory, transform=None, properties_csv="properties.csv", target_properties=None, index_column="sample_id"):
        self.root_directory = root_directory
        self.transform = transform
        self.image_files = sorted(
            [f for f in os.listdir(root_directory) if f.endswith('.bmp')],
            key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else -1
        )

        self.image_names = [os.path.splitext(f)[0] for f in self.image_files]

        # Load properties if available
        # If properties_csv is relative and starts with "../", go up
        if os.path.isabs(properties_csv):
            self.properties_csv_path = properties_csv
        else:
            self.properties_csv_path = os.path.normpath(os.path.join(root_directory, properties_csv))

        if not os.path.exists(self.properties_csv_path):
            raise FileNotFoundError(f"Properties CSV file not found: {self.properties_csv_path}")
        self.properties_df = pd.read_csv(self.properties_csv_path)

        if index_column not in self.properties_df.columns:
            raise ValueError(f"Index column '{index_column}' not found in properties CSV.")
        
        if target_properties is None:
            raise ValueError("target_properties must be specified as a list of column names.")
        
        if isinstance(target_properties, str):
            target_properties = [target_properties]

        for prop in target_properties:
            if prop not in self.properties_df.columns:
                raise ValueError(f"Target property '{prop}' not found in properties CSV.")
        
        self.target_properties = target_properties
        self.index_column = index_column

        # Create a dictionary: image_name -> property value
        self.property_dict = {}
        
        for _, row in self.properties_df.iterrows():
            sample_id_value = row[self.index_column]
        
            # Convert to string key for folder lookup
            try:
                key = str(int(float(sample_id_value)))
            except (ValueError, TypeError):
                continue
        
            # Build array with sample_id as the first element
            try:
                full_array = np.array(
                    row[target_properties].astype(float).tolist(),
                    dtype=np.float32
                )
                self.property_dict[key] = full_array
            except Exception as e:
                print(f"Error parsing row for sample_id {sample_id_value}: {e}")


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_directory, self.image_files[idx])
        image_name = self.image_names[idx]
        img = Image.open(image_path).convert('L')  # Grayscale
        img = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        if self.transform:
            img = self.transform(img)
        img = torch.from_numpy(img).unsqueeze(0)  # Add channel dimension: [1, H, W]

        prop = self.property_dict.get(image_name)
        if prop is None:
            raise ValueError(f"Target property/ies missing for folder '{image_name}' in properties CSV.")
        
        # Convert to numpy array explicitly for safe nan check
        prop_array = np.array(prop, dtype=np.float32)
        
        if np.isnan(prop_array).any():
            raise ValueError(f"Target property/ies contain NaNs for folder '{image_name}' in properties CSV.")
        
        target_value = torch.tensor(prop_array, dtype=torch.float32)

        return img, target_value

# DATSET SPLIT
def split_dataset(
    dataset, 
    train_pct=0.7, 
    val_pct=0.15, 
    test_pct=0.15, 
    seed=42, 
    indices_dir=None, 
    save_indices=False, 
    load_indices=False
):
    assert abs(train_pct + val_pct + test_pct - 1.0) < 1e-6, "Percents must sum to 1"
    num_samples = len(dataset)
    indices = np.arange(num_samples)

    # Paths for saving/loading indices
    if indices_dir is not None:
        os.makedirs(indices_dir, exist_ok=True)
        train_idx_path = os.path.join(indices_dir, "train_indices.npy")
        val_idx_path = os.path.join(indices_dir, "val_indices.npy")
        test_idx_path = os.path.join(indices_dir, "test_indices.npy")
    else:
        train_idx_path = val_idx_path = test_idx_path = None

    if load_indices and train_idx_path and os.path.exists(train_idx_path):
        train_idx = np.load(train_idx_path)
        val_idx = np.load(val_idx_path)
        test_idx = np.load(test_idx_path)
    else:
        np.random.seed(seed)
        np.random.shuffle(indices)
        train_end = int(train_pct * num_samples)
        val_end = train_end + int(val_pct * num_samples)
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        if save_indices and train_idx_path:
            np.save(train_idx_path, train_idx)
            np.save(val_idx_path, val_idx)
            np.save(test_idx_path, test_idx)

    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)

# LOAD DATA FOR 3D
def get_data_loaders_3d(
    directory,
    batch_size=4, 
    train_pct=0.7, 
    val_pct=0.15, 
    test_pct=0.15, 
    seed=42, 
    shuffle=True,
    indices_dir=None,
    save_indices=False,
    load_indices=False,
    properties_csv="properties.csv",
    target_properties=None,
    index_column="sample_id"
):
    dataset = BMP3DDataset(directory,
                           transform=None,
                           properties_csv=properties_csv,
                           target_properties=target_properties,
                           index_column=index_column
                           )
    train_set, val_set, test_set = split_dataset(
        dataset, train_pct, val_pct, test_pct, seed, 
        indices_dir=indices_dir, save_indices=save_indices, load_indices=load_indices
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, generator=generator)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# LOAD DATA FOR 2D
def get_data_loaders_2d(
    directory, 
    batch_size=32, 
    train_pct=0.7, 
    val_pct=0.15, 
    test_pct=0.15, 
    seed=42, 
    shuffle=True,
    indices_dir=None,
    save_indices=False,
    load_indices=False,
    properties_csv="properties.csv",
    target_properties=None,
    index_column="sample_id"
):
    dataset = BMP2DDataset(directory,
                           transform=None,
                           properties_csv=properties_csv,
                           target_properties=target_properties,
                           index_column=index_column
                           )
    train_set, val_set, test_set = split_dataset(
        dataset, train_pct, val_pct, test_pct, seed, 
        indices_dir=indices_dir, save_indices=save_indices, load_indices=load_indices
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, generator=generator)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader