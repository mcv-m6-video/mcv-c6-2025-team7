from pathlib import Path

import numpy as np

from Week1.evaluate_output_video import evaluate_video

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import griddata

def plot_3d_metrics(results, metric='f1'):
    # Prepare data for plotting
    alphas, rhos, z_values = [], [], []

    for (alpha, rho), (recall, precision, ap) in results.items():
        alphas.append(alpha)
        rhos.append(rho)

        if metric == 'recall':
            z_values.append(recall)
        elif metric == 'precision':
            z_values.append(precision)
        elif metric == 'ap':
            z_values.append(ap)
        else:  # F1 score (default)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            z_values.append(f1)

    # Convert lists to numpy arrays
    alphas = np.array(alphas)
    rhos = np.array(rhos)
    z_values = np.array(z_values)

    # Create grid for surface plot
    alpha_grid, rho_grid = np.meshgrid(np.unique(alphas), np.unique(rhos))
    z_grid = griddata((alphas, rhos), z_values, (alpha_grid, rho_grid), method='cubic')

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Surface plot
    surf = ax.plot_surface(alpha_grid, rho_grid, z_grid, cmap='viridis', edgecolor='k')
    fig.colorbar(surf)

    # Labels and title
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Ro')
    ax.set_zlabel('F1')

    plt.show()

if __name__ == "__main__":
    video_path = Path("Output_Videos") / "AdaptiveModelling"
    assert video_path.exists(), f"{video_path} doesn't exists!"

    parameter_search = {
        "alpha": list(np.arange(0.1, 0.7, 0.1)),
        "rho": list(np.arange(0, 10, 1)),
    }

    results = {}
    for alpha in parameter_search["alpha"]:
        for rho in parameter_search["rho"]:
            rec, prec, ap = evaluate_video(
                video_file=f"Output_Videos/AdaptiveModelling/task_2_1_mean_alpha{alpha}_rho{rho}.avi",
            )
            results[(alpha, rho)] = {"rec": rec.mean(), "prec": prec.mean(), "ap": ap}

    plot_3d_metrics(results, metric='f1')
