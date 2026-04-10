# Optimisation du Travelling Salesman Problem (TSP) via ACO et Réseau de Neurones
**M1 MIAGE MIXTE — Université Paris Nanterre**  
**Auteur : NIGATU Amare**

---

## Description
Ce projet explore l'apport du Machine Learning pour améliorer une métaheuristique.  
On combine :
- **ACO** (Ant Colony Optimization) — algorithme de base
- **Réseau de neurones simple (Multi-Layer Perceptron - MLP)** — guide les fourmis via la formule modifiée :

$$P_{ij}^{NN} = \frac{\tau_{ij}^\alpha \cdot \eta_{ij}^\beta \cdot \hat{q}_{ij}^\gamma}{\sum_k \tau_{ik}^\alpha \cdot \eta_{ik}^\beta \cdot \hat{q}_{ik}^\gamma}$$

---

## Structure du dépôt

```
├── src/
│   ├── aco_base.py      # ACO classique
│   ├── aco_hybrid.py    # ACO + Neural Network (formule P_ij^NN)
│   ├── ml_model.py      # Architecture du réseau de neurones (5 entrées)
│   ├── train.py         # Entraînement du réseau (labels basés sur L_bar)
│   └── utils.py         # Génération de villes + visualisation
├── data/
│   ├── ulysses22.tsp # Instance TSPLIB (22 villes)
│   ├── att48.tsp # Instance TSPLIB (48 villes)
│   ├── kroA100.tsp # Instance TSPLIB (100 villes)
├── main.py              # Comparaison ACO classique vs hybride
├── tsp_model.pth        # Modèle entraîné (généré par train.py)
├── README.md
```

---

## Installation

```bash
pip install numpy matplotlib torch tsplib95
```

Cette commande installe les dépendances nécessaires au projet :

```
NumPy : Pour le calcul matriciel et la manipulation des distances.

Matplotlib : Pour la visualisation des tournées et des graphiques de résultats.

Torch (PyTorch) : Pour la conception et l'entraînement du réseau de neurones (MLP).

tsplib95 : Pour le chargement et le parsing des instances standards de la bibliothèque TSPLIB (fichiers .tsp).
```
---

## Exécution

### Étape 1 — Entraîner le réseau de neurones
```bash
python src/train.py
```
Génère `tsp_model.pth`.

### Étape 2 — Lancer la comparaison
```bash
python main.py
```
Affiche les distances et le meilleur chemin trouvé.

---

## Paramètres principaux

| Paramètre | Valeur | Rôle |
|---|---|---|
| `alpha` | 1 | Importance des phéromones |
| `beta` | 2 | Importance de la visibilité |
| `evaporation` | 0.5 | Taux d'évaporation |
| `Q` | 100 | Constante de dépôt de phéromone |
| `gamma` | 1.0 | Influence du réseau de neurones |

---

## Résultats attendus
Comparer la distance trouvée par l'ACO classique vs l'ACO hybride (Machine Learning (ML)-enhanced) sur la même instance TSP.
