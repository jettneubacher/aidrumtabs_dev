import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def standardize(X_train, X_test):
    scaler = StandardScaler()

    # Flatten the data to apply scaling across all features
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

    # Fit the scaler on the training data and transform both train and test
    X_train_standardized = scaler.fit_transform(X_train_reshaped)
    X_test_standardized = scaler.transform(X_test_reshaped)

    # Reshape the standardized data back to the original shape (samples, time_steps, features)
    X_train_standardized = X_train_standardized.reshape(X_train.shape)
    X_test_standardized = X_test_standardized.reshape(X_test.shape)

    return X_train_standardized, X_test_standardized


class DrumHitDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.float32)


def load_data(npz_file, test_size=0.2, batch_size=32):
    npz_data = np.load(npz_file, allow_pickle=True)
    X = npz_data['X']
    Y = npz_data['Y']
    
    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

    # Apply standardization to the data
    X_train, X_test = standardize(X_train, X_test)
    
    # Create datasets
    train_dataset = DrumHitDataset(X_train, Y_train)
    test_dataset = DrumHitDataset(X_test, Y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
