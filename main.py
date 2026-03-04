import numpy as np
from src.utils import generate_cities, plot_tsp_result
from src.aco_base import ACO_Base

def run_experiment():
    """
    Main function to run the TSP optimization.
    Step 1: Generate cities
    Step 2: Run classic ACO
    Step 3: Display results
    """
    print("Étape 1 : Génération des données...")
    
    # We start with 20 cities for testing
    num_cities = 20
    villes = generate_cities(num_cities)
    
    print(f"Étape 2 : Lancement de l'ACO Classique sur {num_cities} villes...")
    
    # Initialize ACO with standard parameters
    # alpha=1 (pheromone importance), beta=2 (distance importance)
    aco = ACO_Base(villes, n_ants=15, n_iterations=100, alpha=1, beta=2)
    
    # Run the algorithm
    best_path, best_dist = aco.run()
    
    print("\n--- RÉSULTATS FINAUX (ACO Classique) ---")
    print(f"Meilleure distance trouvée : {best_dist:.4f}")
    print(f"Chemin optimal : {best_path}")
    print("-----------------------------------------")

    plot_tsp_result(villes, best_path, best_dist)

if __name__ == "__main__":
    # Ensure the random results are reproducible
    np.random.seed(42) 
    run_experiment()