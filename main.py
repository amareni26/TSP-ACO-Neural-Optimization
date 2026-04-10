import random

import numpy as np
import torch
from src.utils import generate_cities, plot_tsp_result, load_tsplib_instance
from src.aco_base import ACO_Base
from src.aco_hybrid import ACO_Hybrid

def run_comparison_on_instance(instance_name, filepath,
                                n_ants, n_iterations,
                                alpha, beta, evaporation, Q, gamma=1.5,
                                N_RUNS=5):
    """
    Runs N_RUNS comparisons of Classic ACO vs Hybrid ACO
    on a fixed TSPLIB instance.
    """
    print(f"\n{'='*55}")
    print(f"  INSTANCE : {instance_name}")
    print(f"{'='*55}")

    # Load fixed TSPLIB instance
    villes = load_tsplib_instance(filepath)
    num_cities = len(villes)
    print(f"  Nombre de villes : {num_cities}")

    classic_distances = []
    hybrid_distances  = []

    last_path_c = last_path_h = None
    last_dist_c = last_dist_h = None

    for run in range(N_RUNS):
        print(f"\n  --- Run {run+1}/{N_RUNS} ---")

        # --- Classic ACO ---
        aco_classic = ACO_Base(
            villes,
            n_ants=n_ants,
            n_iterations=n_iterations,
            alpha=alpha,
            beta=beta,
            evaporation=evaporation,
            Q=Q
        )
        path_c, dist_c = aco_classic.run()
        classic_distances.append(dist_c)
        print(f"  Classique : {dist_c:.4f}")

        # --- Hybrid ACO ---
        try:
            aco_hybrid = ACO_Hybrid(
                villes,
                model_path="tsp_model.pth",
                gamma=gamma,
                n_ants=n_ants,
                n_iterations=n_iterations,
                alpha=alpha,
                beta=beta,
                evaporation=evaporation,
                Q=Q
            )
            path_h, dist_h = aco_hybrid.run()
            hybrid_distances.append(dist_h)
            print(f"  Hybride   : {dist_h:.4f}")

        except FileNotFoundError:
            print("  [ERREUR] tsp_model.pth non trouvé.")
            print("           Veuillez exécuter : python src/train.py")
            return None

        last_path_c = path_c
        last_path_h = path_h
        last_dist_c = dist_c
        last_dist_h = dist_h

    # --- Summary ---
    avg_classic = np.mean(classic_distances)
    avg_hybrid  = np.mean(hybrid_distances)
    improvement = ((avg_classic - avg_hybrid) / avg_classic) * 100

    print(f"\n  {'='*45}")
    print(f"  RÉSULTATS — {instance_name} ({num_cities} villes)")
    print(f"  {'='*45}")
    for i in range(N_RUNS):
        winner = "H" if hybrid_distances[i] < classic_distances[i] else "C"
        print(f"  Run {i+1} → C: {classic_distances[i]:.4f} "
              f"| H: {hybrid_distances[i]:.4f}  [✓{winner}]")
    print(f"  {'-'*45}")
    print(f"  Moyenne Classique : {avg_classic:.4f}")
    print(f"  Moyenne Hybride   : {avg_hybrid:.4f}")
    print(f"  {'-'*45}")
    if avg_hybrid < avg_classic:
        print(f"  ✓ Hybride MEILLEUR de {improvement:.2f}%")
    elif avg_hybrid > avg_classic:
        print(f"  ✗ Classique MEILLEUR de {abs(improvement):.2f}%")
    else:
        print(f"  = Résultats identiques.")
    print(f"  {'='*45}")

    # Plot last run
    print(f"\n  Affichage du meilleur chemin (dernier run)...")
    if last_dist_h <= last_dist_c:
        plot_tsp_result(villes, last_path_h, last_dist_h,
                        title=f"ACO Hybride — {instance_name}")
    else:
        plot_tsp_result(villes, last_path_c, last_dist_c,
                        title=f"ACO Classique — {instance_name}")

    return {
        "instance"    : instance_name,
        "n_cities"    : num_cities,
        "avg_classic" : avg_classic,
        "avg_hybrid"  : avg_hybrid,
        "improvement" : improvement
    }


def run_all():
    """
    Tests Classic ACO vs Hybrid ACO on 3 TSPLIB instances
    of increasing size, as recommended by the teacher.
    """

    # Fix seed for reproducibility
    np.random.seed(42)      # ← fixes numpy randomness
    random.seed(42)         # ← fixes Python randomness
    torch.manual_seed(42)   # ← fixes PyTorch randomness
    
    print("=" * 55)
    print("  ACO CLASSIQUE VS ACO HYBRIDE — TSPLIB BENCHMARK")
    print("=" * 55)

    # --- Common parameters ---
    alpha      = 1.0
    beta       = 2.0
    evaporation = 0.5
    Q          = 100
    gamma      = 1.5
    N_RUNS     = 20

    # --- 3 instances: small, medium, large ---
    instances = [
        # (name,        filepath,              n_ants, n_iter)
        ("ulysses22",  "data/ulysses22.tsp",   10,     50 ),
        ("att48",      "data/att48.tsp",        20,     100),
        ("kroA100",    "data/kroA100.tsp",      10,     50),
    ]

    all_results = []

    for name, filepath, n_ants, n_iter in instances:
        result = run_comparison_on_instance(
            instance_name=name,
            filepath=filepath,
            n_ants=n_ants,
            n_iterations=n_iter,
            alpha=alpha,
            beta=beta,
            evaporation=evaporation,
            Q=Q,
            gamma=gamma,
            N_RUNS=N_RUNS
        )
        if result:
            all_results.append(result)

    # --- Global summary table ---
    if all_results:
        print(f"\n{'='*55}")
        print("  TABLEAU RÉCAPITULATIF GLOBAL")
        print(f"{'='*55}")
        print(f"  {'Instance':<12} {'Villes':>6} "
              f"{'Classique':>10} {'Hybride':>10} {'Δ%':>8}")
        print(f"  {'-'*50}")
        for r in all_results:
            sign = "+" if r['improvement'] < 0 else "-"
            print(f"  {r['instance']:<12} {r['n_cities']:>6} "
                  f"{r['avg_classic']:>10.4f} {r['avg_hybrid']:>10.4f} "
                  f"{sign}{abs(r['improvement']):>6.2f}%")
        print(f"  {'='*55}")


if __name__ == "__main__":
    run_all()