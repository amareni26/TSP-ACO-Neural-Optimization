import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import generate_cities
from aco_base import ACO_Base
from ml_model import create_model

def collect_training_data(n_problems=50):

    print(f"Génération de {n_problems} problèmes TSP pour l'entraînement...")
    inputs  = []
    targets = []

    # Train on multiple city sizes for better generalization
    city_sizes = [20, 30, 48]  

    for problem_idx in range(n_problems):

        # Pick a random city size each problem
        n_cities = np.random.choice(city_sizes) 

        villes = generate_cities(n_cities)
        aco = ACO_Base(villes, n_ants=10, n_iterations=30)
        all_paths = aco._construct_all_paths()

        all_distances = [dist for _, dist in all_paths]
        L_bar = np.mean(all_distances)

        for path, L_k in all_paths:
            partial_len = 0.0

            for step in range(len(path) - 1):
                i = path[step]
                j = path[step + 1]

                visited_ratio = step / len(path)
                tau_ij = aco.pheromones[i][j]
                d_ij   = aco.distances[i][j] + 1e-10
                eta_ij = 1.0 / d_ij

                feat = [tau_ij, eta_ij, d_ij, partial_len, visited_ratio]
                inputs.append(feat)

                label = 1 if L_k <= L_bar else 0
                targets.append([label])

                partial_len += aco.distances[i][j]

        if (problem_idx + 1) % 5 == 0:
            print(f"  Problème {problem_idx + 1}/{n_problems} "
                  f"({n_cities} villes) traité.")

    print(f"  Total d'exemples générés : {len(inputs)}")
    return (torch.FloatTensor(np.array(inputs)),
            torch.FloatTensor(np.array(targets)))
    

def train_model(n_epochs=200):
    """
    TRAINING LOOP
 
    Trains the Neural Network using Binary Cross-Entropy loss
    and the Adam optimizer, as described in the report.
 
    Loss function : Weighted Binary Cross-Entropy (BCE)
    Optimizer     : Adam (lr=0.001)
    """
    model     = create_model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Generate training data
    X, y = collect_training_data(n_problems=50)

    # FIX: Calculate class weight to handle imbalance
    # Count how many 1s and 0s we have
    n_positive = y.sum().item()          # number of label=1
    n_negative = len(y) - n_positive     # number of label=0

    # pos_weight = n_negative / n_positive
    # This tells the model: "label=1 is much rarer, pay more attention to it"
    pos_weight = torch.tensor([n_negative / n_positive])
    print(f"  Class balance → positifs: {int(n_positive)}, négatifs: {int(n_negative)}")
    print(f"  pos_weight appliqué     : {pos_weight.item():.2f}")

    # Use BCEWithLogitsLoss instead of BCELoss (more stable + supports pos_weight)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"\nDébut de l'entraînement du réseau de neurones ({n_epochs} epochs)...")

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass — note: remove Sigmoid from model output for BCEWithLogitsLoss
        outputs = model(X)

        # Compute weighted loss
        loss = criterion(outputs, y)

        # Backward pass + weight update
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch [{epoch+1:3d}/{n_epochs}]  Loss: {loss.item():.4f}")

    # Save trained model
    torch.save(model.state_dict(), "tsp_model.pth")
    print("\nEntraînement terminé. Modèle sauvegardé : 'tsp_model.pth'")

    return model

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    train_model()