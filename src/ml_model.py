import torch
import torch.nn as nn
import torch.optim as optim

class TSP_Predictor(nn.Module):
    """
    Neural Network model to predict if an edge between two cities is 'good'.
    This is a Multi-Layer Perceptron (MLP).

    Architecture:
        Input  (5 features) → Hidden (32, ReLU) → Hidden (16, ReLU) → Output (1, Sigmoid)
 
    Input features for edge (i, j):
        1. tau_ij      : Pheromone level on edge (i,j)
        2. eta_ij      : Visibility = 1 / d_ij
        3. d_ij        : Euclidean distance between city i and city j
        4. partial_len : Length of the partial tour built so far
        5. visited_ratio: Ratio of cities already visited (progress of the tour)
 
    Output:
        q_hat_ij ∈ [0, 1] : Quality score of edge (i,j)
        0 = bad edge, 1 = promising edge
    """
    def __init__(self):
        super(TSP_Predictor, self).__init__()
        
        # We use a Sequential container to stack layers of neurons
        self.fc = nn.Sequential(
           # Input layer: 5 features → 32 neurons
            nn.Linear(5, 32),
            nn.ReLU(), # Activation function: adds non-linearity to learn complex patterns
            
            # Layer 2: Hidden Layer
            # Input = 32, Output = 16 neurons
            nn.Linear(32, 16),
            nn.ReLU(),
            
            # Layer 3: Output Layer
            # Input = 16, Output = 1 (A single score/probability)
            nn.Linear(16, 1),
            
            # Sigmoid squashes the output to [0, 1]
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