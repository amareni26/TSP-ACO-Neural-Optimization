import torch
import numpy as np
from src.aco_base import ACO_Base
from src.ml_model import create_model

class ACO_Hybrid(ACO_Base):
    """"
    Hybrid ACO that integrates a Neural Network into the city-selection
    probability formula during tour construction.
 
    Modified probability formula (P_ij^NN):
 
        P_ij^NN = (tau_ij^alpha * eta_ij^beta * q_hat_ij^gamma)
                  / sum_k (tau_ik^alpha * eta_ik^beta * q_hat_ik^gamma)
    where:
        q_hat_ij = Neural Network prediction score for edge (i,j)
        gamma    = weight controlling the NN's influence
                   (gamma=0 → classic ACO, gamma>0 → NN guides search)
    """
    def __init__(self, city_coords, model_path="tsp_model.pth", gamma=1.0, **kwargs):
        super().__init__(city_coords, **kwargs)

        # gamma: influence of the Neural Network on the probability formula
        self.gamma = gamma

        # 1. Load the trained Neural Network (AI brain)
        self.model = create_model()
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        # Set to evaluation mode
        self.model.eval()

        # Precompute NN scores for all edges (i,j) at initialization
        # This avoids calling the model inside every probability computation
        self.nn_scores = self._precompute_nn_scores()

    def _precompute_nn_scores(self):
        """
        Runs the Neural Network once for all edges (i,j) and stores
        the predicted quality scores in a matrix (N x N).
 
        Input features for each edge (i,j):
            [tau_ij, eta_ij, d_ij, partial_len=0, visited_ratio=0]
        Note: partial_len and visited_ratio are 0 at initialization.
        They will be updated dynamically during tour construction.
        """
        scores = np.zeros((self.n_cities, self.n_cities))
        with torch.no_grad():
            for i in range(self.n_cities):
                for j in range(self.n_cities):
                    if i != j:
                        feat = self._build_features(i, j, partial_len=0.0, visited_ratio=0.0)
                        scores[i][j] = self.model(feat).item() 

        return scores
    
    def _build_features(self, i, j, partial_len, visited_ratio):
        """
        Builds the 5-feature input vector for edge (i,j):
            [tau_ij, eta_ij, d_ij, partial_len, visited_ratio]
        """
        tau_ij = self.pheromones[i][j]
        d_ij = self.distances[i][j] + 1e-10
        eta_ij = 1.0 / d_ij

        feat = torch.FloatTensor([tau_ij, eta_ij, d_ij, partial_len, visited_ratio])
        # shape: (1, 5)
        return feat.unsqueeze(0) 

    def _construct_single_path(self):
        """
        Overrides the classic path construction.
        The NN score q_hat_ij is integrated into the probability formula
        at each step of the tour construction.
        """
        path = [np.random.randint(0, self.n_cities)]
        partial_len = 0.0
 
        while len(path) < self.n_cities:
            i = path[-1]
            visited_ratio = len(path) / self.n_cities
 
            # Update NN scores dynamically using current tour state
            with torch.no_grad():
                for j in range(self.n_cities):
                    if j not in path:
                        feat = self._build_features(i, j, partial_len, visited_ratio)
                        self.nn_scores[i][j] = self.model(feat).item()
 
            # Compute modified probability P_ij^NN
            probs = self._calculate_probabilities_nn(i, path)
            next_city = np.random.choice(range(self.n_cities), p=probs)
 
            # Update partial tour length
            partial_len += self.distances[i][next_city]
            path.append(next_city)
 
        return path
 
    def _calculate_probabilities_nn(self, i, visited):
        """
        Modified ACO probability formula with Neural Network guidance:
 
            P_ij^NN = (tau^alpha * eta^beta * q_hat^gamma)
                      / sum_k(tau^alpha * eta^beta * q_hat^gamma)
 
        When gamma=0 → equivalent to classic ACO.
        """
        tau = np.copy(self.pheromones[i])
        q   = np.copy(self.nn_scores[i])
 
        # Zero out visited cities
        for city in visited:
            tau[city] = 0.0
            q[city]   = 0.0
 
        eta = 1.0 / (self.distances[i] + 1e-10)
 
        # Ensure NN scores are positive (avoid 0^gamma issues)
        q = np.clip(q, 1e-10, 1.0)
 
        # Modified probability formula
        weights = (tau ** self.alpha) * (eta ** self.beta) * (q ** self.gamma)
        total = weights.sum()
 
        if total == 0:
            # Fallback: uniform over unvisited
            unvisited = [c for c in range(self.n_cities) if c not in visited]
            probs = np.zeros(self.n_cities)
            for c in unvisited:
                probs[c] = 1.0 / len(unvisited)
            return probs
 
        return weights / total 