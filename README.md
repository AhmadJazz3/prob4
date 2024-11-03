# Advanced Feed-Forward Neural Network with PyTorch

This project implements a multi-layer feed-forward neural network using PyTorch. The model includes advanced elements such as batch normalization, dropout, and a learning rate scheduler to improve performance and generalization.

## Project Structure

- `model.py`: Contains the `NeuralNetwork` class and `BuildDataset` function.
- `demo.py`: A demonstration on how to use the neural network to train, evaluate, and analyze performance on the Iris dataset.
- `README.md`: Project documentation and usage guide.

## Installation

To run this project, clone the repository and install the necessary packages:

```bash
git clone [<repository_url>](https://github.com/AhmadJazz3/prob4)
cd prob4
pip install -r requirements.txt

````

## Usage

Training the Neural Network

Import the necessary components from model.py and load the dataset:

from model import NeuralNetwork, BuildDataset

X_train, X_test, y_train, y_test = BuildDataset()

Initialize the neural network with desired parameters and train:


input_size = X_train.shape[1]

hidden_sizes = [64, 32]

output_size = 3

dropout_prob = 0.3

model = NeuralNetwork(input_size, hidden_sizes, output_size, dropout_prob)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

  for epoch in range(epochs):
    
    model.train()
    
    optimizer.zero_grad()
    
    outputs = model(X_train)
    
    loss = criterion(outputs, y_train)
    
    loss.backward()
    
    optimizer.step()
    
    scheduler.step()

## Testing and Evaluation

After training, evaluate the model on the test data and visualize performance metrics:


  with torch.no_grad():
      
      model.eval()
      
      outputs = model(X_test)
      
      _, predicted = torch.max(outputs, 1)
      
      accuracy = (predicted == y_test).sum().item() / y_test.size(0)
      
      print(f"Test Accuracy: {accuracy * 100:.2f}%")

Additional evaluation includes a confusion matrix and a classification report, which can be visualized using matplotlib and seaborn in demo.py.

## Model Details

The neural network consists of:

An input layer

Two hidden layers with 64 and 32 neurons respectively, each with ReLU activation, batch normalization, and dropout (30% dropout rate)

An output layer using Softmax for multi-class classification

The model is trained with Cross-Entropy Loss and optimized using the Adam optimizer, with a StepLR learning rate scheduler.

## Dataset
The model is trained on the Iris dataset, which is loaded and preprocessed using sklearn. The dataset is split into training and test sets with standard scaling applied.

## Requirements
The required packages are listed in requirements.txt:

torch

scikit-learn

seaborn

matplotlib


