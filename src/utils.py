import numpy as np
import matplotlib.pyplot as plt

def generate_cities(n_cities):
    # Generates random x, y coordinates for n cities
    return np.random.rand(n_cities, 2)

def plot_cities(cities):
    plt.scatter(cities[:, 0], cities[:, 1])
    plt.title("Position des villes (TSP)")
    plt.show()

    # Test it
    if __name__ == "__main__":
        villes = generate_cities(20)
        print("Villes générées avec succès.")
        plot_cities(villes)

def plot_tsp_result(cities, path, distance):
    """Plots the cities and the optimal path found."""
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
                c='orange', s=250, edgecolors='black', marker='X', zorder=5, label='Arrivée')

    # Label each city with its index (0, 1, 2,...)
    for i, (xi, yi) in enumerate(cities):
        plt.text(xi + 0.01, yi +0.01, str(i), fontsize=12, fontweight='bold', color='darkred')

    # Draw the path
    for i in range(len(path) - 1):
        plt.plot([cities[path[i], 0], cities[path[i+1], 0]],
                 [cities[path[i], 1], cities[path[i+1], 1]],
                 c='red', alpha=0.6, linewidth=2, zorder=2)
        
    # Draw the line back to the start
    plt.plot([cities[path[-1], 0], cities[path[0], 0]],
             [cities[path[-1], 1], cities[path[0], 1]], 
             c='red', alpha=0.6, linewidth=2, zorder=2)
    
    plt.title(f"TSP Result - Distance: {distance:.4f}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()

