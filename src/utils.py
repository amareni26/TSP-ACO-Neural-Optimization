import numpy as np
import matplotlib.pyplot as plt

def generate_cities(n_cities):
    # Generates random x, y coordinates for n cities
    return np.random.rand(n_cities, 2)

def plot_cities(cities):
    """Plots city positions only (no path)."""
    plt.figure(figsize=(8, 6))
    plt.scatter(cities[:, 0], cities[:, 1], c='blue', edgecolors='k', s=100)
    for i, (x, y) in enumerate(cities):
        plt.text(x + 0.01, y + 0.01, str(i), fontsize=12, fontweight='bold', color='darkred')
    plt.title("Position des villes (TSP)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.tight_layout()
    plt.show()

def plot_tsp_result(cities, path, distance, title="TSP Result"):
    """Plots the cities and the tour found by ACO.
    - Green star  : Starting city
    - Red cross   : Last city before returning to start
    - Violet lines: Tour path"""
    plt.figure(figsize=(10, 8))

    # Extract x and y coordinates of the cities
    x = cities[:, 0]
    y = cities[:, 1]

    # Plot cities as blue dots
    plt.scatter(x, y, c='blue', edgecolors='k', s=100, zorder=3)

    # Highlight Start City (Green) and End City
    start_city = path[0]
    end_city = path[-1]

    plt.scatter(cities[start_city, 0], cities[start_city, 1],
                c='green', s=250, edgecolors='violet', marker='P', zorder=5, label='Départ')
    plt.scatter(cities[end_city, 0], cities[end_city, 1],
                c='red', s=250, edgecolors='black', marker='X', zorder=5, label='Dernier arrêt')

    # Label each city with its index (0, 1, 2,...)
    for i, (xi, yi) in enumerate(cities):
        plt.text(xi + 0.01, yi +0.01, str(i), fontsize=12, fontweight='bold', color='darkred')

    # Draw the path
    for i in range(len(path) - 1):
        plt.plot([cities[path[i], 0], cities[path[i+1], 0]],
                 [cities[path[i], 1], cities[path[i+1], 1]],
                 c='violet', alpha=0.6, linewidth=2, zorder=2)
        
    # Return arc (last city → start city)
    plt.plot([cities[path[-1], 0], cities[path[0], 0]],
             [cities[path[-1], 1], cities[path[0], 1]], 
             c='violet', alpha=0.6, linewidth=2, zorder=2)
    
    plt.title(f"{title}\nDistance totale : {distance:.4f}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    villes = generate_cities(20)
    print("Villes générées avec succès.")
    plot_cities(villes)
