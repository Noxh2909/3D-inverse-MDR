from typing import cast
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from PIL import Image
from scipy.stats import spearmanr
import seaborn as sns

def safe_spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    r, _ = spearmanr(x, y)
    return cast(float, r)


# -------------------------------------------------
# CONFIG
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PICTURES_DIR = BASE_DIR / "pictures"
ANALYSIS_DIR = BASE_DIR / "analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)

ANALYSIS_2D_DIR = ANALYSIS_DIR / "2d"
ANALYSIS_3D_DIR = ANALYSIS_DIR / "3d"
ANALYSIS_2D_DIR.mkdir(exist_ok=True)
ANALYSIS_3D_DIR.mkdir(exist_ok=True)

ANALYSIS_COMPARE_DIR = ANALYSIS_DIR / "compare"
ANALYSIS_COMPARE_DIR.mkdir(exist_ok=True)


# -------------------------------------------------
# LOAD CSV
# -------------------------------------------------

def load_embedding(csv_path):
    names = []
    coords = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("mask_png"):
                continue
            if row[0].startswith("Participant"):
                continue
            if len(row) != 4:
                continue

            names.append(row[0])
            coords.append([float(row[1]), float(row[2]), float(row[3])])

    return names, np.asarray(coords)


# -------------------------------------------------
# IMAGE HELPERS
# -------------------------------------------------

def load_image(name, zoom=0.25):
    path = PICTURES_DIR / name
    if not path.exists():
        return None
    img = Image.open(path)
    arr = np.asarray(img)
    return OffsetImage(arr, zoom=zoom)

# -------------------------------------------------
# PLOT MATRIX WITH IMAGES + VALUES
# -------------------------------------------------

def plot_dissimilarity_matrix(names, D, csv_name, out_dir):
    n = len(names)
    fig, ax = plt.subplots(figsize=(1.2 * n, 1.2 * n))

    im = ax.imshow(D, cmap="viridis")
    ax.set_xticks([])
    ax.set_yticks([])

    # --- values inside cells ---
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i,
                f"{D[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if D[i, j] > D.max() * 0.5 else "black",
                fontsize=10
            )

    # --- image ticks ---
    for i, name in enumerate(names):
        img = load_image(name)
        if img is None:
            continue

        # X axis
        ab_x = AnnotationBbox(
            img,
            (i, -0.9),
            xycoords="data",
            frameon=False,
            box_alignment=(0.5, 1)
        )
        ax.add_artist(ab_x)

        # Y axis
        ab_y = AnnotationBbox(
            img,
            (-0.6, i),
            xycoords="data",
            frameon=False,
            box_alignment=(1, 0.5)
        )
        ax.add_artist(ab_y)

    ax.set_xlim(-1, n - 0.5)
    ax.set_ylim(n - 0.5, -1)

    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Dissimilarity Matrix (Euclidean Distance)")

    out = out_dir / f"{csv_name}_dissimilarity.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out}")


# -------------------------------------------------
# COMPARISON ANALYSES
# -------------------------------------------------

def vectorize_upper_triangle(D):
    return D[np.triu_indices_from(D, k=1)]

def mantel_test(D1, D2, permutations=500, seed=42):
    rng = np.random.default_rng(seed)

    vec1 = vectorize_upper_triangle(D1)
    vec2 = vectorize_upper_triangle(D2)

    obs_corr = safe_spearmanr(vec1, vec2)

    null_dist = np.zeros(permutations, dtype=float)

    n = D1.shape[0]
    for i in range(permutations):
        perm = rng.permutation(n)
        D2_perm = D2[perm][:, perm]
        vec2_perm = vectorize_upper_triangle(D2_perm)
        null_dist[i] = safe_spearmanr(vec1, vec2_perm)

    p_value = (np.sum(null_dist >= obs_corr) + 1) / (permutations + 1)
    return obs_corr, null_dist, p_value

def plot_distance_distribution(D2d_list, D3d_list, out_path):
    all_2d = np.concatenate([vectorize_upper_triangle(D) for D in D2d_list])
    all_3d = np.concatenate([vectorize_upper_triangle(D) for D in D3d_list])

    plt.figure(figsize=(8,6))
    sns.kdeplot(all_2d, label="2D", fill=True)
    sns.kdeplot(all_3d, label="3D", fill=True)
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Density")
    plt.title("Distance Distribution: 2D vs 3D")
    plt.legend()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def plot_spearman_scatter(D2d_mean, D3d_mean, out_path):
    vec_2d = vectorize_upper_triangle(D2d_mean)
    vec_3d = vectorize_upper_triangle(D3d_mean)
    corr = safe_spearmanr(vec_2d, vec_3d)

    plt.figure(figsize=(7,7))
    sns.scatterplot(x=vec_2d, y=vec_3d, alpha=0.6)
    min_val = min(vec_2d.min(), vec_3d.min())
    max_val = max(vec_2d.max(), vec_3d.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("Mean 2D Distances")
    plt.ylabel("Mean 3D Distances")
    plt.title(f"Spearman Correlation between 2D and 3D\nr = {corr:.3f}")
    plt.grid(True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def plot_mantel_test(obs_corr, null_dist, out_path):
    plt.figure(figsize=(8,6))
    sns.histplot(null_dist, bins=30, kde=False, color="skyblue")
    plt.axvline(obs_corr, color="red", linestyle="--", label=f"Observed r = {obs_corr:.3f}")
    plt.xlabel("Spearman Correlation")
    plt.ylabel("Frequency")
    plt.title("Mantel Test Null Distribution")
    plt.legend()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def knn_overlap(D2d_mean, D3d_mean, k_values=[3,5]):
    n = D2d_mean.shape[0]
    overlaps = {k: [] for k in k_values}

    for k in k_values:
        for i in range(n):
            neighbors_2d = set(np.argsort(D2d_mean[i])[1:k+1])
            neighbors_3d = set(np.argsort(D3d_mean[i])[1:k+1])
            overlap = len(neighbors_2d.intersection(neighbors_3d)) / k
            overlaps[k].append(overlap)

    return overlaps

def plot_knn_overlap(overlaps, out_path):
    means = [np.mean(overlaps[k]) for k in sorted(overlaps.keys())]
    ks = sorted(overlaps.keys())

    plt.figure(figsize=(6,5))
    sns.barplot(hue=[str(k) for k in ks], y=means, palette="pastel")
    plt.ylim(0,1)
    plt.xlabel("k")
    plt.ylabel("Average k-NN Overlap")
    plt.title("Neighborhood Preservation: 2D vs 3D")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def analyze_csv(csv_path, out_dir):
    names, coords = load_embedding(csv_path)
    if len(coords) < 2:
        print("Not enough points:", csv_path)
        return None, None

    D = squareform(pdist(coords, metric="euclidean"))
    csv_name = csv_path.stem
    plot_dissimilarity_matrix(names, D, csv_name, out_dir)
    return names, D


def main(): 
    all_names_2d = None
    all_names_3d = None
    dissimilarities_2d = []
    dissimilarities_3d = []

    for cond in ("2d", "3d"):
        cond_dir = RESULTS_DIR / cond
        if not cond_dir.exists():
            continue
        out_dir = ANALYSIS_2D_DIR if cond == "2d" else ANALYSIS_3D_DIR
        for csv_file in cond_dir.glob("*.csv"):
            names, D = analyze_csv(csv_file, out_dir)
            if D is None:
                continue
            if cond == "2d":
                if all_names_2d is None:
                    all_names_2d = names
                dissimilarities_2d.append(D)
            else:
                if all_names_3d is None:
                    all_names_3d = names
                dissimilarities_3d.append(D)

    # Only proceed if we have data for both conditions
    if dissimilarities_2d and dissimilarities_3d:
        # Check that names match between 2d and 3d
        if all_names_2d != all_names_3d:
            print("Warning: Stimulus names do not match between 2D and 3D. Comparison may be invalid.")

        # Compute mean dissimilarity matrices
        D2d_mean = np.mean(dissimilarities_2d, axis=0)
        D3d_mean = np.mean(dissimilarities_3d, axis=0)

        # 1) Distance distribution plot
        dist_dist_path = ANALYSIS_COMPARE_DIR / "distance_distribution_2d_vs_3d.png"
        plot_distance_distribution(dissimilarities_2d, dissimilarities_3d, dist_dist_path)

        # 2) Spearman correlation comparison
        spearman_path = ANALYSIS_COMPARE_DIR / "spearman_2d_vs_3d.png"
        plot_spearman_scatter(D2d_mean, D3d_mean, spearman_path)

        # 3) Mantel test
        obs_corr, null_dist, p_value = mantel_test(D2d_mean, D3d_mean, permutations=500)
        mantel_path = ANALYSIS_COMPARE_DIR / "mantel_2d_vs_3d.png"
        plot_mantel_test(obs_corr, null_dist, mantel_path)
        print(f"Mantel test p-value: {p_value:.4f}")

        # 4) Neighborhood preservation
        overlaps = knn_overlap(D2d_mean, D3d_mean, k_values=[3,5])
        knn_path = ANALYSIS_COMPARE_DIR / "knn_overlap_2d_vs_3d.png"
        plot_knn_overlap(overlaps, knn_path)


if __name__ == "__main__":
    main()