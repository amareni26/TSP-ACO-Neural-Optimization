import numpy as np
import random

class ACO_Base:
    def __init__(self, city_coords, n_ants=10, n_iterations=50, alpha=1, beta=2, evaporation=0.5, Q=100):
        """
        Initializes the Ant Colony Optimization parameters.
        - alpha: Controls the influence of the pheromone (smell).
        - beta: Controls the influence of the distance (visibility).
        - evaporation: How much pheromone disappears after each turn.
        - Q: Pheromone deposit constant. A shorter tour deposits more pheromone.
             Formula: delta_tau = Q / L_k
        """
        self.city_coords = city_coords
        self.n_cities = len(city_coords)
        self.n_ants = n_ants
        self.n_iterations = n_iterations

        # Pheromone importance
        self.alpha = alpha 
        # Distance importance 
        self.beta = beta    
        self.evaporation = evaporation
        # Pheromone deposit constant

        
        # Create a square matrix (N x N) initialized with 1s for pheromones
        self.pheromones = np.ones((self.n_cities, self.n_cities))
        # Precompute all distances between cities to save time during execution
        self.distances = self._calculate_dist_matrix()
    

    def _calculate_dist_matrix(self):
        """Calculates the Euclidean distance between every pair of cities."""

        matrix = np.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                if i != j:
                    # np.linalg.norm calculates the straight-line distance
                    matrix[i][j] = np.linalg.norm(self.city_coords[i] - self.city_coords[j])
        return matrix
    
    def run(self):
        """Main loop: runs all iterations and returns the best path found."""
        best_path = None
        best_dist = float('inf')  # Start with 'infinity' as the best distance

        for _ in range(self.n_iterations):
            # Step 1: Each ant builds a complete tour
            all_paths = self._construct_all_paths()

            # Step 2: Update pheromone levels based on the paths found
            self._update_pheromones(all_paths)

            # Step 3: Check if any ant found a new shortest path
            for path, dist in all_paths:
                if dist < best_dist:
                    best_dist = dist
                    best_path = path
        return best_path, best_dist
    
    def _construct_all_paths(self):
        """Simulates all ants moving through the cities."""
        all_paths = []
        for _ in range(self.n_ants):
            path = self._construct_single_path()
            dist = self._path_distance(path)
            all_paths.append((path, dist))
        return all_paths
    
    def _construct_single_path(self):
        """Simulates one ant building a full tour (visiting every city once)."""
        # Start at a random city
        path = [random.randint(0, self.n_cities - 1)]
        while len(path) < self.n_cities:
            i = path[-1] # Current city

            # Decide which city to visit next based on probability
            probs = self._calculate_probabilities(i, path)
            next_city = np.random.choice(range(self.n_cities), p=probs)
            path.append(next_city)
        return path
    
    def _calculate_probabilities(self, i, visited):
        """Computes the probability of moving from city i to each unvisited city j.
           Formula (classic ACO):
               P_ij = (tau_ij^alpha * eta_ij^beta) / sum_k(tau_ik^alpha * eta_ik^beta)
           where eta_ij = 1 / d_ij  (visibility)"""
        
        # Get pheromone levels for all possible next cities
        tau = np.copy(self.pheromones[i])

        # Constraint: An ant cannot visit a city it has already been to
        for city in visited:
            tau[city] = 0
        
        # Visibility (eta) is the inverse of distance (1/distance)
        eta = 1.0 / (self.distances[i] + 1e-10)

        # The ACO Formula: (Pheromone^alpha) * (Visibility^beta)
        weights = (tau ** self.alpha) * (eta ** self.beta)
        total = weights.sum()

        if total == 0:
            # Fallback: uniform probability over unvisited cities
            unvisited = [c for c in range(self.n_cities) if c not in visited]
            probs = np.zeros(self.n_cities)
            for c in unvisited:
                probs[c] = 1.0 / len(unvisited)
            return probs
        
        return weights / total
    
    def _path_distance(self, path):
        """Calculates the total distance of a tour (including return to start)."""
        d = 0
        for k in range(len(path) - 1):
            d += self.distances[path[k], path[k+1]]
        # Add distance back to the starting city to close the loop
        d += self.distances[path[-1]][path[0]]
        return d
    
    def _update_pheromones(self, all_paths):
        """Updates pheromone trails after each iteration.
        Formula:
            tau_ij ← (1 - rho) * tau_ij + sum_k(delta_tau_ij_k)
        where:
            delta_tau_ij_k = Q / L_k  if ant k used edge (i,j)
                           = 0        otherwise"""
        
        # Step 1: Evaporation (some pheromone disappears over time)
        self.pheromones *= (1 - self.evaporation)

        # Step 2: Deposit — shorter tours leave more pheromone (Q / L_k)
        for path, dist in all_paths:
            deposit = self.Q / dist
            for k in range(len(path) - 1):
                # Shorter paths (smaller 'dist') result in more pheromone deposit
                self.pheromones[path[k]][path[k+1]] += deposit
            # Also deposit on the return arc (closing the loop)
            self.pheromones[path[-1]][path[0]] += deposit


