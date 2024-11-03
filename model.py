# model.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_prob=0.3):
        super(NeuralNetwork, self).__init__()
        layers = []
        prev_size = input_size

        # Creating hidden layers with batch normalization and dropout
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout_prob))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Softmax(dim=1))

        # Sequential container of layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def BuildDataset():
    # Load the Iris dataset
    data = load_iris()
    X = data.data
    y = data.target

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor
