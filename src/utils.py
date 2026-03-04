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


