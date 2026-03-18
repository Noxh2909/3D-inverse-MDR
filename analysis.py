import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import procrustes
from scipy.stats import spearmanr, kendalltau
from PIL import Image


# -------------------------------------------------
# CONFIG
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
FINAL_RESULTS_DIR = BASE_DIR / "final_results"
PICTURES_DIR = BASE_DIR / "pictures"
ANALYSIS_DIR = BASE_DIR / "analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)

ANALYSIS_2D_DIR = ANALYSIS_DIR / "2d"
ANALYSIS_3D_DIR = ANALYSIS_DIR / "3d"
ANALYSIS_2D_DIR.mkdir(exist_ok=True)
ANALYSIS_3D_DIR.mkdir(exist_ok=True)

PROCRUSTES_DIR = ANALYSIS_DIR / "procrustes"
PROCRUSTES_2D_DIR = PROCRUSTES_DIR / "2d"
PROCRUSTES_3D_DIR = PROCRUSTES_DIR / "3d"
PROCRUSTES_DIR.mkdir(exist_ok=True)
PROCRUSTES_2D_DIR.mkdir(exist_ok=True)
PROCRUSTES_3D_DIR.mkdir(exist_ok=True)

# Set to True for anonymous participant labels (Participant 1, 2, 3...)
# Set to False to use actual folder names
ANONYMOUS = True


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
        img_x = load_image(name)
        img_y = load_image(name)
        if img_x is None:
            continue

        # X axis
        ab_x = AnnotationBbox(
            img_x,
            (i, -0.9),
            xycoords="data",
            frameon=False,
            box_alignment=(0.5, 1)
        )
        ax.add_artist(ab_x)

        # Y axis
        if img_y is not None:
            ab_y = AnnotationBbox(
                img_y,
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

    out = out_dir / f"{csv_name}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out}")


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def analyze_csv(csv_path, out_dir):
    names, coords = load_embedding(csv_path)
    if len(coords) < 2:
        print("Not enough points:", csv_path)
        return

    D = squareform(pdist(coords, metric="euclidean"))
    if D.max() > 0:
        D = D / D.max()
    csv_name = csv_path.stem
    plot_dissimilarity_matrix(names, D, csv_name, out_dir)


# -------------------------------------------------
# PROCRUSTES ANALYSIS
# -------------------------------------------------

def generalized_procrustes_analysis(coords_list):
    """
    Perform Generalized Procrustes Analysis (GPA) on a list of coordinate arrays.
    Returns the mean shape and all aligned configurations.
    """
    if len(coords_list) == 0:
        return None, []
    
    # Start with the first configuration as reference
    aligned = [coords_list[0].copy()]
    reference = coords_list[0].copy()
    
    # Align all other configurations to the reference
    for coords in coords_list[1:]:
        _, aligned_coords, _ = procrustes(reference, coords)
        aligned.append(aligned_coords)
    
    # Iterative alignment to mean
    for _ in range(10):  # 10 iterations usually sufficient
        # Compute mean shape
        mean_shape = np.mean(aligned, axis=0)
        
        # Re-align all to mean
        new_aligned = []
        for coords in aligned:
            _, aligned_coords, _ = procrustes(mean_shape, coords)
            new_aligned.append(aligned_coords)
        aligned = new_aligned
    
    # Final mean shape
    mean_shape = np.mean(aligned, axis=0)
    return mean_shape, aligned


def compute_procrustes_distance(coords1, coords2):
    """Compute Procrustes distance (disparity) between two configurations."""
    _, _, disparity = procrustes(coords1, coords2)
    return disparity


def plot_intersubject_consistency(disparities, participant_names, condition, out_path):
    """Plot bar chart of Procrustes disparities for each participant."""
    fig, ax = plt.subplots(figsize=(max(8, len(disparities) * 0.8), 6))
    
    x = np.arange(len(disparities))
    bars = ax.bar(x, disparities, color='steelblue', edgecolor='black')
    
    # Add mean line
    mean_disp = np.mean(disparities)
    ax.axhline(mean_disp, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_disp:.4f}')
    
    ax.set_xlabel('Participant')
    ax.set_ylabel('Procrustes Disparity')
    ax.set_title(f'Intersubject Consistency - {condition.upper()}\n(lower = more consistent)')
    ax.set_xticks(x)
    ax.set_xticklabels(participant_names, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

# -------------------------------------------------
# SHEPARD DIAGRAM (GLOBAL)
# -------------------------------------------------

def plot_shepard_global(coords_list, stimulus_names, condition, out_path):
    """Aggregate Shepard diagram across participants."""

    # compute consensus distances
    D_mats = [squareform(pdist(c, metric="euclidean")) for c in coords_list]
    D_consensus = np.mean(D_mats, axis=0)

    v_ref = squareform(D_consensus, checks=False)

    all_embed = []

    for D in D_mats:
        v = squareform(D, checks=False)
        all_embed.extend(v)

    all_embed = np.asarray(all_embed)
    v_ref_rep = np.tile(v_ref, len(coords_list))

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(v_ref_rep, all_embed, alpha=0.45, s=18)

    # identity line (perfect reconstruction)
    min_v = min(v_ref_rep.min(), all_embed.min())
    max_v = max(v_ref_rep.max(), all_embed.max())
    ax.plot(
        [min_v, max_v],
        [min_v, max_v],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Perfect reconstruction"
    )

    # trend line following the mass of the data (binned mean)
    bins = np.linspace(min_v, max_v, 25)

    bin_centers = []
    bin_means = []

    for i in range(len(bins) - 1):
        mask = (v_ref_rep >= bins[i]) & (v_ref_rep < bins[i + 1])
        if np.sum(mask) > 0:
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            bin_means.append(np.mean(all_embed[mask]))

    bin_centers = np.array(bin_centers)
    bin_means = np.array(bin_means)

    # global R² between reference and embedding distances
    if len(v_ref_rep) > 1:
        r = np.corrcoef(v_ref_rep, all_embed)[0, 1]
        r2 = r ** 2
    else:
        r2 = 0.0

    ax.plot(
        bin_centers,
        bin_means,
        linewidth=3,
        label=f"Mean distortion trend (R²={r2:.3f})"
    )

    ax.set_xlabel("Reference distance")
    ax.set_ylabel("Embedding distance")
    ax.set_title(f"Shepard Diagram ({condition.upper()})")
    ax.legend()

    ax.grid(True, linestyle=":", linewidth=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {out_path}")


# -------------------------------------------------
# KNN PRESERVATION CURVE
# -------------------------------------------------

def knn_overlap(D1, D2, k):
    """Compute kNN overlap between two distance matrices."""
    n = D1.shape[0]

    overlap = []

    for i in range(n):
        nn1 = np.argsort(D1[i])[1:k+1]
        nn2 = np.argsort(D2[i])[1:k+1]

        overlap.append(len(set(nn1).intersection(set(nn2))) / k)

    return np.mean(overlap)


def plot_knn_curve(coords_list, condition, out_path):

    D_mats = [squareform(pdist(c, metric="euclidean")) for c in coords_list]
    D_consensus = np.mean(D_mats, axis=0)

    n = D_consensus.shape[0]

    ks = range(1, n)

    scores = []

    for k in ks:
        vals = []
        for D in D_mats:
            vals.append(knn_overlap(D, D_consensus, k))
        scores.append(np.mean(vals))

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(list(ks), scores, marker="o")

    ax.set_xlabel("k Nearest Neighbours")
    ax.set_ylabel("Neighbour preservation")
    ax.set_title(f"kNN Preservation Curve ({condition.upper()})")

    ax.set_ylim(0, 1)
    ax.grid(True, linestyle=":", linewidth=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {out_path}")


# -------------------------------------------------
# AXIS VARIANCE (DIMENSION USAGE)
# -------------------------------------------------

def plot_axis_variance(coords_list, condition, out_path):

    vars_all = []

    for coords in coords_list:
        v = np.var(coords, axis=0)
        v = v / np.sum(v)
        vars_all.append(v)

    vars_all = np.asarray(vars_all)

    mean_vars = np.mean(vars_all, axis=0)

    dims = ["X", "Y", "Z"][:len(mean_vars)]

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.bar(dims, mean_vars)

    ax.set_ylabel("Variance ratio")
    ax.set_title(f"Axis Variance ({condition.upper()})")

    ax.set_ylim(0, 1)

    ax.grid(True, axis="y", linestyle=":", linewidth=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {out_path}")


# -------------------------------------------------
# GLOBAL DISTANCE CORRELATION METRICS
# -------------------------------------------------

def plot_global_similarity_metrics(coords_list, condition, out_path):
    """
    Compute global similarity metrics between participant RDMs and the
    consensus RDM. Metrics: Pearson, Spearman, Kendall, Cosine.
    """

    # compute distance matrices
    D_mats = [squareform(pdist(c, metric="euclidean")) for c in coords_list]

    # consensus matrix
    D_consensus = np.mean(D_mats, axis=0)

    # flatten consensus
    v_ref = squareform(D_consensus, checks=False)

    # flatten all participant distances
    all_embed = []
    for D in D_mats:
        v = squareform(D, checks=False)
        all_embed.extend(v)

    all_embed = np.asarray(all_embed)

    # repeat reference for same length
    v_ref_rep = np.tile(v_ref, len(coords_list))

    # --- metrics ---
    pearson = np.corrcoef(v_ref_rep, all_embed)[0, 1]

    spearman, _ = spearmanr(v_ref_rep, all_embed)

    kendall, _ = kendalltau(v_ref_rep, all_embed)

    cosine = np.dot(v_ref_rep, all_embed) / (
        np.linalg.norm(v_ref_rep) * np.linalg.norm(all_embed)
    )

    metrics = {
        "Pearson": pearson,
        "Spearman": spearman,
        "Kendall": kendall,
        "Cosine": cosine,
    }

    # --- plot ---
    fig, ax = plt.subplots(figsize=(6, 5))

    names = list(metrics.keys())
    values = list(metrics.values())

    ax.bar(names, values)

    mean_val = np.mean(values)
    ax.axhline(float(mean_val), color="red", linestyle="--", linewidth=2, label=f"Mean = {mean_val:.3f}")

    ax.set_ylim(-1, 1)
    ax.set_ylabel("Similarity")
    ax.set_title(f"Global Distance Similarity Metrics ({condition.upper()})")

    ax.legend()
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {out_path}")


# -------------------------------------------------
# GLOBAL KRUSKAL STRESS + MANTEL CORRELATION
# -------------------------------------------------

def plot_global_stress_mantel(coords_list, condition, out_path):
    """
    Compute global Kruskal Stress-1 and Mantel correlation between
    participant RDM distances and the consensus RDM.
    Aggregated across all participants.
    """

    # distance matrices
    D_mats = [squareform(pdist(c, metric="euclidean")) for c in coords_list]

    # consensus matrix
    D_consensus = np.mean(D_mats, axis=0)

    # flatten consensus
    v_ref = squareform(D_consensus, checks=False)

    # flatten all participant distances
    all_embed = []
    for D in D_mats:
        v = squareform(D, checks=False)
        all_embed.extend(v)

    all_embed = np.asarray(all_embed)

    # repeat reference vector
    v_ref_rep = np.tile(v_ref, len(coords_list))

    # --- Mantel correlation (Pearson between distance vectors) ---
    mantel_r = np.corrcoef(v_ref_rep, all_embed)[0, 1]

    # --- Kruskal Stress-1 ---
    stress = np.sqrt(
        np.sum((all_embed - v_ref_rep) ** 2) / np.sum(v_ref_rep ** 2)
    )

    # plot
    fig, ax = plt.subplots(figsize=(6, 5))

    names = ["Mantel r", "Kruskal Stress"]
    values = [mantel_r, stress]

    ax.bar(names, values)

    ax.set_title(f"Global Structural Metrics ({condition.upper()})")
    ax.set_ylabel("Value")

    ax.grid(True, axis="y", linestyle=":", linewidth=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {out_path}")


# -------------------------------------------------
# CORRELATION SCATTERPLOTS (GLOBAL)
# -------------------------------------------------

def plot_correlation_scatter_metrics(coords_list, condition, out_path):
    """
    Scatterplots for Pearson, Spearman, and Kendall correlations between
    participant RDM distances and the consensus RDM distances.
    Aggregated across all participants (global).
    """

    # compute distance matrices
    D_mats = [squareform(pdist(c, metric="euclidean")) for c in coords_list]

    # consensus matrix
    D_consensus = np.mean(D_mats, axis=0)

    # flatten consensus
    v_ref = squareform(D_consensus, checks=False)

    # flatten all participant distances
    all_embed = []
    for D in D_mats:
        v = squareform(D, checks=False)
        all_embed.extend(v)

    all_embed = np.asarray(all_embed)

    # repeat reference
    v_ref_rep = np.tile(v_ref, len(coords_list))

    # compute ranks for rank-based correlations
    ref_rank = np.argsort(np.argsort(v_ref_rep))
    emb_rank = np.argsort(np.argsort(all_embed))

    # prepare filenames
    base = Path(out_path)
    pearson_path = base.with_name(base.stem + "_pearson.png")
    spearman_path = base.with_name(base.stem + "_spearman.png")
    kendall_path = base.with_name(base.stem + "_kendall.png")

    # -------------------------------------------------
    # PEARSON SCATTER
    # -------------------------------------------------
    fig, ax = plt.subplots(figsize=(6,6))

    ax.scatter(v_ref_rep, all_embed, alpha=0.35, s=18)

    # perfect reconstruction line (y = x)
    min_v = min(v_ref_rep.min(), all_embed.min())
    max_v = max(v_ref_rep.max(), all_embed.max())
    ax.plot(
        [min_v, max_v],
        [min_v, max_v],
        linestyle="--",
        linewidth=2,
        label="Perfect reconstruction"
    )

    # regression line
    m, b = np.polyfit(v_ref_rep, all_embed, 1)
    x_line = np.linspace(v_ref_rep.min(), v_ref_rep.max(), 200)
    y_line = m * x_line + b
    ax.plot(x_line, y_line, linewidth=3, label="Regression trend")

    r = np.corrcoef(v_ref_rep, all_embed)[0,1]

    ax.set_title(f"Pearson Correlation ({condition.upper()})  r={r:.3f}")
    ax.set_xlabel("Consensus distance")
    ax.set_ylabel("Participant distance")
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend()

    plt.tight_layout()
    plt.savefig(pearson_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {pearson_path}")

    # -------------------------------------------------
    # SPEARMAN SCATTER
    # -------------------------------------------------
    fig, ax = plt.subplots(figsize=(6,6))

    ax.scatter(ref_rank, emb_rank, alpha=0.35, s=18)

    # perfect reconstruction line (rank agreement)
    min_v = min(ref_rank.min(), emb_rank.min())
    max_v = max(ref_rank.max(), emb_rank.max())
    ax.plot(
        [min_v, max_v],
        [min_v, max_v],
        linestyle="--",
        linewidth=2,
        label="Perfect reconstruction"
    )

    # regression line
    m, b = np.polyfit(ref_rank, emb_rank, 1)
    x_line = np.linspace(ref_rank.min(), ref_rank.max(), 200)
    y_line = m * x_line + b
    ax.plot(x_line, y_line, linewidth=3, label="Regression trend")

    rho, _ = spearmanr(v_ref_rep, all_embed)

    ax.set_title(f"Spearman Correlation ({condition.upper()})  ρ={rho:.3f}")
    ax.set_xlabel("Consensus rank")
    ax.set_ylabel("Participant rank")
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend()

    plt.tight_layout()
    plt.savefig(spearman_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {spearman_path}")

    # -------------------------------------------------
    # KENDALL SCATTER
    # -------------------------------------------------
    fig, ax = plt.subplots(figsize=(6,6))

    ax.scatter(ref_rank, emb_rank, alpha=0.35, s=18)

    # perfect reconstruction line (rank agreement)
    min_v = min(ref_rank.min(), emb_rank.min())
    max_v = max(ref_rank.max(), emb_rank.max())
    ax.plot(
        [min_v, max_v],
        [min_v, max_v],
        linestyle="--",
        linewidth=2,
        label="Perfect reconstruction"
    )

    # regression line
    m, b = np.polyfit(ref_rank, emb_rank, 1)
    x_line = np.linspace(ref_rank.min(), ref_rank.max(), 200)
    y_line = m * x_line + b
    ax.plot(x_line, y_line, linewidth=3, label="Regression trend")

    tau, _ = kendalltau(v_ref_rep, all_embed)

    ax.set_title(f"Kendall Correlation ({condition.upper()})  τ={tau:.3f}")
    ax.set_xlabel("Consensus rank")
    ax.set_ylabel("Participant rank")
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend()

    plt.tight_layout()
    plt.savefig(kendall_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {kendall_path}")


def plot_procrustes_dissimilarity_matrix(coords_list, participant_names, condition, out_path):
    """Plot pairwise Procrustes dissimilarity matrix between all participants."""
    n = len(coords_list)
    D = np.zeros((n, n))
    
    # Compute pairwise Procrustes disparities
    for i in range(n):
        for j in range(i + 1, n):
            _, _, disparity = procrustes(coords_list[i], coords_list[j])
            D[i, j] = disparity
            D[j, i] = disparity

    # compute mean disparity (ignore diagonal zeros)
    mask = ~np.eye(n, dtype=bool)
    mean_disp = np.mean(D[mask])
    
    # Plot matrix
    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(8, n * 0.8)))
    im = ax.imshow(D, cmap="viridis")
    
    # Add values inside cells
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i,
                f"{D[i, j]:.3f}",
                ha="center",
                va="center",
                color="white" if D[i, j] > D.max() * 0.5 else "black",
                fontsize=8
            )
    
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(participant_names, rotation=45, ha='right')
    ax.set_yticklabels(participant_names)
    
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f"Procrustes Dissimilarity Matrix - {condition.upper()}  (Mean disparity = {mean_disp:.4f})")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main(): 
    # Collect all embeddings per condition
    embeddings_2d = []  # list of (participant_name, names, coords)
    embeddings_3d = []
    
    # Iterate through all participant folders in final_results
    for participant_dir in FINAL_RESULTS_DIR.iterdir():
        if not participant_dir.is_dir() or participant_dir.name.startswith('.'):
            continue
        
        participant_name = participant_dir.name
        
        for cond in ("2d", "3d"):
            cond_dir = participant_dir / cond
            if not cond_dir.exists():
                continue
            out_dir = ANALYSIS_2D_DIR if cond == "2d" else ANALYSIS_3D_DIR
            for csv_file in cond_dir.glob("*.csv"):
                # Create dissimilarity matrix
                analyze_csv(csv_file, out_dir)
                
                # Load embedding for Procrustes
                names, coords = load_embedding(csv_file)
                if len(coords) >= 2:
                    if cond == "2d":
                        embeddings_2d.append((participant_name, names, coords))
                    else:
                        embeddings_3d.append((participant_name, names, coords))
    
    # Procrustes analysis for 2D condition
    if len(embeddings_2d) >= 2:
        print("\n--- Procrustes Analysis for 2D ---")
        coords_list_2d = [e[2] for e in embeddings_2d]
        if ANONYMOUS:
            participant_names_2d = [f"Participant {i+1}" for i in range(len(embeddings_2d))]
        else:
            participant_names_2d = [e[0] for e in embeddings_2d]
        
        mean_shape_2d, aligned_2d = generalized_procrustes_analysis(coords_list_2d)
        
        # Compute disparity of each participant to mean
        disparities_2d = []
        for aligned_coords in aligned_2d:
            disp = compute_procrustes_distance(mean_shape_2d, aligned_coords)
            disparities_2d.append(disp)
        
        # Plot intersubject consistency
        consistency_path_2d = PROCRUSTES_2D_DIR / "intersubject_consistency_2d.png"
        plot_intersubject_consistency(disparities_2d, participant_names_2d, "2d", consistency_path_2d)
        
        # Plot Procrustes dissimilarity matrix
        dissim_path_2d = PROCRUSTES_2D_DIR / "procrustes_dissimilarity_matrix_2d.png"
        plot_procrustes_dissimilarity_matrix(coords_list_2d, participant_names_2d, "2d", dissim_path_2d)
        
        print(f"Mean disparity 2D: {np.mean(disparities_2d):.4f}")

        # Shepard diagram
        shepard_path = PROCRUSTES_2D_DIR / "shepard_2d.png"
        plot_shepard_global(coords_list_2d, embeddings_2d[0][1], "2d", shepard_path)

        # kNN curve
        knn_path = PROCRUSTES_2D_DIR / "knn_preservation_2d.png"
        plot_knn_curve(coords_list_2d, "2d", knn_path)

        # axis variance
        axis_path = PROCRUSTES_2D_DIR / "axis_variance_2d.png"
        plot_axis_variance(coords_list_2d, "2d", axis_path)

        # global similarity metrics
        metrics_path = PROCRUSTES_2D_DIR / "distance_similarity_metrics_2d.png"
        plot_global_similarity_metrics(coords_list_2d, "2d", metrics_path)
        # correlation scatterplots
        scatter_path = PROCRUSTES_2D_DIR / "distance_correlation_scatter_2d.png"
        plot_correlation_scatter_metrics(coords_list_2d, "2d", scatter_path)

        # Kruskal stress + Mantel correlation
        stress_mantel_path = PROCRUSTES_2D_DIR / "global_stress_mantel_2d.png"
        plot_global_stress_mantel(coords_list_2d, "2d", stress_mantel_path)
    
    # Procrustes analysis for 3D condition
    if len(embeddings_3d) >= 2:
        print("\n--- Procrustes Analysis for 3D ---")
        coords_list_3d = [e[2] for e in embeddings_3d]
        if ANONYMOUS:
            participant_names_3d = [f"Participant {i+1}" for i in range(len(embeddings_3d))]
        else:
            participant_names_3d = [e[0] for e in embeddings_3d]
        
        mean_shape_3d, aligned_3d = generalized_procrustes_analysis(coords_list_3d)
        
        # Compute disparity of each participant to mean
        disparities_3d = []
        for aligned_coords in aligned_3d:
            disp = compute_procrustes_distance(mean_shape_3d, aligned_coords)
            disparities_3d.append(disp)
        
        # Plot intersubject consistency
        consistency_path_3d = PROCRUSTES_3D_DIR / "intersubject_consistency_3d.png"
        plot_intersubject_consistency(disparities_3d, participant_names_3d, "3d", consistency_path_3d)
        
        # Plot Procrustes dissimilarity matrix
        dissim_path_3d = PROCRUSTES_3D_DIR / "procrustes_dissimilarity_matrix_3d.png"
        plot_procrustes_dissimilarity_matrix(coords_list_3d, participant_names_3d, "3d", dissim_path_3d)
        
        print(f"Mean disparity 3D: {np.mean(disparities_3d):.4f}")

        # Shepard diagram
        shepard_path = PROCRUSTES_3D_DIR / "shepard_3d.png"
        plot_shepard_global(coords_list_3d, embeddings_3d[0][1], "3d", shepard_path)

        # kNN curve
        knn_path = PROCRUSTES_3D_DIR / "knn_preservation_3d.png"
        plot_knn_curve(coords_list_3d, "3d", knn_path)

        # axis variance
        axis_path = PROCRUSTES_3D_DIR / "axis_variance_3d.png"
        plot_axis_variance(coords_list_3d, "3d", axis_path)

        # global similarity metrics
        metrics_path = PROCRUSTES_3D_DIR / "distance_similarity_metrics_3d.png"
        plot_global_similarity_metrics(coords_list_3d, "3d", metrics_path)
        # correlation scatterplots
        scatter_path = PROCRUSTES_3D_DIR / "distance_correlation_scatter_3d.png"
        plot_correlation_scatter_metrics(coords_list_3d, "3d", scatter_path)

        # Kruskal stress + Mantel correlation
        stress_mantel_path = PROCRUSTES_3D_DIR / "global_stress_mantel_3d.png"
        plot_global_stress_mantel(coords_list_3d, "3d", stress_mantel_path)


if __name__ == "__main__":
    main()
