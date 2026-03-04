import torch
import torch.nn as nn
import torch.optim as optim

class TSP_Predictor(nn.Module):
    """
    Neural Network model to predict if an edge between two cities is 'good'.
    This is a Multi-Layer Perceptron (MLP).
    """
    def __init__(self):
        super(TSP_Predictor, self).__init__()
        
        # We use a Sequential container to stack layers of neurons
        self.fc = nn.Sequential(
            # Layer 1: Input Layer
            # Input size = 4: (x_city1, y_city1, x_city2, y_city2)
            # Output size = 32 neurons
            nn.Linear(4, 32),
            nn.ReLU(), # Activation function: adds non-linearity to learn complex patterns
            
            # Layer 2: Hidden Layer
            # Input = 32, Output = 16 neurons
            nn.Linear(32, 16),
            nn.ReLU(),
            
            # Layer 3: Output Layer
            # Input = 16, Output = 1 (A single score/probability)
            nn.Linear(16, 1),
            
            # Sigmoid squashes the output value between 0 and 1
            # 0 = "Bad edge", 1 = "Perfect edge for the path"
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Defines the forward pass: how data travels from input to output.
        """
        return self.fc(x)

def create_model():
    """Helper function to instantiate the model."""
    return TSP_Predictor()