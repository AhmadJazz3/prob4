# demo.ipynb

import torch
import torch.nn as nn
import torch.optim as optim
from model import NeuralNetwork, BuildDataset
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
X_train, X_test, y_train, y_test = BuildDataset()

# Define the network parameters
input_size = X_train.shape[1]
hidden_sizes = [64, 32]  # Multi-layer model
output_size = 3  # Number of classes in the Iris dataset
dropout_prob = 0.3

# Initialize the model, loss function, optimizer, and learning rate scheduler
model = NeuralNetwork(input_size, hidden_sizes, output_size, dropout_prob)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

# Training the neural network
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    scheduler.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluating the model
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)

    # Calculate accuracy
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, predicted))

    # Confusion matrix
    cm = confusion_matrix(y_test, predicted)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()
