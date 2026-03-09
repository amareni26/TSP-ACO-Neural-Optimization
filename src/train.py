import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.utils import generate_cities
from src.aco_base import ACO_Base
from src.ml_model import create_model

def collect_training_data(n_problems=10):
    """
    DATA GENERATION PHASE:
    Uses the classic ACO algorithm to create a "Gold Standard" dataset.
    We teach the AI by showing it what the ants found to be the best paths.
    """

    print(f"Génération de {n_problems} problèmes TSP pour l'entraînement...")
    inputs = []
    targets = []

    for _ in range(n_problems):
        # Generate a small map of 10 cities
        villes = generate_cities(10)
        # Run standard ACO to find the "best" path for this map
        aco = ACO_Base(villes, n_ants=10, n_iterations=30)
        best_path, _ = aco.run()

        # We examine every possible connection (edge) between cities
        for i in range(len(villes)):
            for j in range(len(villes)):
                if i != j
                    # Input: The coordinates of the two cities [x1, y1, x2, y2]
                    feat = np.concatenate([villes[i], villes[j]])
                    inputs.append(feat)

                    # Target (Label): 
                    # 1 if this specific edge was part of the winning path, 
                    # 0 otherwise. This is "Binary Classification".
                    is_best = 0
                    for k in range(len(best_path)-1):
                        if (best_path[k] == i and best_path[k+1] == j) or (best_path[k] == j and best_path[k+1] == i):
                            is_best = 1
                    targets.append([is_best])
    
    # Convert lists to PyTorch Tensors (the data format required for Deep Learning)
    return torch.FloatTensor(np.array(inputs)), torch.FloatTensor(np.array(targets))

def train_model():
    """
    TRAINING LOOP:
    The process where the Neural Network adjusts its internal weights to reduce error.
    """
    # Initialize the architecture we defined in ml_model.py
    model = create_model()

    # Binary Cross Entropy Loss: Standard for 0 or 1 classification
    criterion = nn.BCELoss()

    # Adam Optimizer: Automatically adjusts the learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Step 1: Create the dataset (X = features, y = labels)
    X, y = collect_training_data(20)

    print("Début de l'entraînement du réseau de neurones...")

    # Step 2: Loop through the data multiple times (Epochs)
    for epoch in range(100):
        # Reset gradients
        optimizer.zero_grad()

        # Forward pass: Make predictions
        outputs = model(X)
        
        # Calculate the error (Loss)
        loss = criterion(outputs, y)

        # Backward pass: Calculate how to change weights to fix the error
        loss.backward()

        # Update weights
        optimizer.step()

        # Print progress every 20 steps
        if (epoch+1) % 20 ==0:
            print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")
    
    # Step 3: Save the trained "brain" to a file
    # This allows us to use the AI later without retraining it
    torch.save(model.state_dict(), "tsp_model.pth")
    print("Entraînement terminé. Modèle sauvegardé sous 'tsp_model.pth'")

if __name__ == "__main__":
    train_model()