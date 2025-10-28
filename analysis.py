from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

import math

# Matplotlib nur beim Plotten importieren, damit das Skript auch ohne GUI nutzbar bleibt
import matplotlib
# Falls kein Display (z. B. Headless), Agg benutzen
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (benötigt für 3D)
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"

Points = List[Tuple[str, float, float, float]]

def _coords_array(points: Points) -> np.ndarray:
    return np.array([[x, y, z] for (_n, x, y, z) in points], dtype=float)

def _names(points: Points) -> List[str]:
    return [n for (n, *_rest) in points]

# ------------------------------ CSV Laden ------------------------------

def _detect_delimiter(sample_path: Path) -> str:
    """Erkennt Komma oder Semikolon als Trennzeichen (Fallback: Komma)."""
    try:
        with sample_path.open("r", encoding="utf-8") as f:
            head = f.read(2048)
        if head.count(";") > head.count(","):
            return ";"
    except Exception:
        pass
    return ","


def load_points_from_csv(path: Path) -> Tuple[List[Tuple[str, float, float, float]], Tuple[str, str, str]]:
    """Parst eine CSV und liefert (points, axis_labels).
    points: [(name, x, y, z), ...]
    axis_labels: (label_x, label_y, label_z)

    Erkennt Name-Spalte tolerant und nimmt die ersten drei numerischen Spalten als X/Y/Z,
    sodass auch Header wie "Color(dark–bright), Age(old–young), Size(big-small)" funktionieren.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    delim = _detect_delimiter(path)
    rows: List[Tuple[str, float, float, float]] = []

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delim)
        if not reader.fieldnames:
            raise ValueError("CSV ohne Header/Spaltennamen.")

        # Normalisierte Feldnamen → Originalheader
        field_map: Dict[str, str] = { (k or "").strip(): (k or "") for k in reader.fieldnames }
        lower_map: Dict[str, str] = { k.lower().strip(): (k or "") for k in reader.fieldnames }

        # Name-Spalte tolerant finden
        name_key = lower_map.get("name") or lower_map.get("id") or lower_map.get("image")
        if not name_key:
            # Fallback: erste Spalte als Name
            name_key = reader.fieldnames[0]

        # Kandidaten für numerische Spalten (alle außer name_key)
        numeric_headers: List[str] = [h for h in reader.fieldnames if h != name_key]

        # Wir sammeln die ersten drei Spalten, die in der ersten Datenzeile numerisch sind
        sample_row = next(iter(reader), None)
        if sample_row is None:
            raise ValueError("CSV enthält keine Datenzeilen.")

        # Prüffunktion für numerische Strings (erlaubt Komma als Dezimaltrenner)
        def _to_float(val: object) -> Optional[float]:
            try:
                return float(str(val).replace(",", ".").strip())
            except Exception:
                return None

        # Reset reader, damit wir alle Zeilen lesen können
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delim)

        # Bestimme die drei numerischen Spalten
        numeric_cols: List[str] = []
        for h in numeric_headers:
            if len(numeric_cols) >= 3:
                break
            v = _to_float(sample_row.get(h, "") if isinstance(sample_row, dict) else None)
            if v is not None:
                numeric_cols.append(h)

        if len(numeric_cols) < 3:
            raise ValueError(f"Finde nicht genug numerische Spalten (gefunden: {numeric_cols}). Erwartet: 3.")

        # Achsenlabels = Original-Header der ausgewählten numerischen Spalten
        axis_labels = tuple(numeric_cols[:3])  # type: ignore[assignment]

        for r in reader:
            try:
                name = str(r[name_key]).strip()
                x = _to_float(r[numeric_cols[0]])
                y = _to_float(r[numeric_cols[1]])
                z = _to_float(r[numeric_cols[2]])
                if x is None or y is None or z is None:
                    continue
                rows.append((name, float(x), float(y), float(z)))
            except Exception:
                continue

    return rows, (axis_labels[0], axis_labels[1], axis_labels[2])


def latest_csv(results_dir: Path = RESULTS_DIR) -> Optional[Path]:
    """Gibt die letzte (neueste) CSV aus `results/` zurück."""
    if not results_dir.exists():
        return None
    cands = sorted(results_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


# ------------------------------ Metrics & Vergleich ------------------------------

def metrics_single(points: Points) -> Dict[str, float]:
    """Berechnet Basis-Metriken für eine Einbettung."""
    C = _coords_array(points)
    if len(C) < 2:
        return {"n": float(len(C))}
    # Varianz je Achse
    var = np.var(C, axis=0)
    total_var = float(np.sum(var))
    # PCA
    try:
        from sklearn.decomposition import PCA  # type: ignore
        pca = PCA(n_components=min(3, C.shape[1]))
        pca.fit(C)
        evr = pca.explained_variance_ratio_
        evr_pad = list(evr) + [0.0] * (3 - len(evr))
    except Exception:
        evr_pad = [np.nan, np.nan, np.nan]
    # Distanzmatrix
    try:
        from scipy.spatial.distance import pdist  # type: ignore
        D = pdist(C)
        mean_d = float(np.mean(D))
        std_d = float(np.std(D))
    except Exception:
        mean_d, std_d = float("nan"), float("nan")
    return {
        "n": float(len(C)),
        "var_x": float(var[0]),
        "var_y": float(var[1]),
        "var_z": float(var[2]),
        "var_total": total_var,
        "pca_ev1": float(evr_pad[0]),
        "pca_ev2": float(evr_pad[1]),
        "pca_ev3": float(evr_pad[2]),
        "dist_mean": mean_d,
        "dist_std": std_d,
    }

def _align_names(pA: Points, pB: Points) -> Tuple[Points, Points]:
    """Schneidet auf gemeinsame Namen zu und sortiert identisch."""
    mapA = {n: (x, y, z) for (n, x, y, z) in pA}
    mapB = {n: (x, y, z) for (n, x, y, z) in pB}
    common = sorted(set(mapA).intersection(mapB))
    a = [(n, *mapA[n]) for n in common]
    b = [(n, *mapB[n]) for n in common]
    return a, b

def compare_embeddings(points2d: Points, points3d: Points) -> Dict[str, float]:
    """Vergleicht zwei Einbettungen (z. B. 2D vs 3D) über Distanzkorrelation und Procrustes."""
    a, b = _align_names(points2d, points3d)
    if len(a) < 3:
        return {"common": float(len(a))}
    A = _coords_array(a)
    B = _coords_array(b)
    out: Dict[str, float] = {"common": float(len(a))}
    # Mantel-ähnliche Korrelation zwischen Distanzvektoren
    try:
        from scipy.spatial.distance import pdist  # type: ignore
        from scipy.stats import spearmanr  # type: ignore
        dA = pdist(A)
        dB = pdist(B)
        res = spearmanr(dA, dB)
        # spearmanr may return a namedtuple with .correlation/.pvalue or a tuple/array-like
        rho = float("nan")
        p = float("nan")
        try:
            # Prefer attribute access (namedtuple or object)
            corr = getattr(res, "correlation", None)
            pv = getattr(res, "pvalue", None)
            if corr is not None:
                rho = float(corr)
            if pv is not None:
                p = float(pv)

            # Fallback: try array/sequence indexing if attributes missing or NaN
            if (corr is None or (isinstance(rho, float) and np.isnan(rho))):
                # Determine size of `res` safely without passing arbitrary types directly to np.size
                size_res = 0
                try:
                    if hasattr(res, "__len__"):
                        # Prefer built-in len() for sequence-like objects
                        size_res = len(res)  # type: ignore[arg-type]
                    else:
                        # Fallback: convert to array then ask for its size
                        size_res = int(np.size(np.asarray(res)))
                except Exception:
                    size_res = 0

                if size_res > 0:
                    arr = np.asarray(res)
                    if arr.size >= 1:
                        rho = float(arr.flat[0])
                    if arr.size >= 2:
                        p = float(arr.flat[1])
        except Exception:
            try:
                arr = np.asarray(res)
                if arr.size >= 2:
                    rho = float(arr.flat[0])
                    p = float(arr.flat[1])
                elif arr.size == 1:
                    rho = float(arr.flat[0])
                    p = float("nan")
            except Exception:
                rho = float("nan")
                p = float("nan")
        out.update({"rank_corr_dist": rho, "rank_corr_p": p})
    except Exception:
        out.update({"rank_corr_dist": float("nan"), "rank_corr_p": float("nan")})
    # Procrustes-Ähnlichkeit
    try:
        from scipy.spatial import procrustes  # type: ignore
        m1, m2, disparity = procrustes(A, B)
        out.update({"procrustes_disparity": float(disparity)})
        rmse = float(np.sqrt(np.mean((m1 - m2) ** 2)))
        out.update({"aligned_rmse": rmse})
    except Exception:
        out.update({"procrustes_disparity": float("nan"), "aligned_rmse": float("nan")})
    # k-NN Überlappung (Neighborhood Preservation)
    def knn_labels(X: np.ndarray, k: int) -> List[set]:
        from sklearn.neighbors import NearestNeighbors  # type: ignore
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(X)))
        nn.fit(X)
        idx = nn.kneighbors(return_distance=False)
        # Zeile i: [i, nn1, nn2,...] → Nachbarn ohne i
        return [set(row[1:]) for row in idx]
    try:
        ks = [3, 5]
        from numpy import mean
        for k in ks:
            if len(A) > k:
                na = knn_labels(A, k)
                nb = knn_labels(B, k)
                overlap = [len(na[i].intersection(nb[i])) / float(k) for i in range(len(A))]
                out[f"knn_overlap_k{k}"] = float(np.mean(overlap))
    except Exception:
        for k in (3, 5):
            out[f"knn_overlap_k{k}"] = float("nan")
    return out

# ------------------------------ Vergleichs-Plots ------------------------------

def shepard_plot(pointsA: Points, pointsB: Points, title: str, out_path: Path) -> Path:
    """Shepard-Plot: paarweise Distanzen A vs. B mit y=x Referenz."""
    from scipy.spatial.distance import pdist
    A = _coords_array(pointsA)
    B = _coords_array(pointsB)
    dA = pdist(A)
    dB = pdist(B)
    fig = plt.figure(figsize=(6, 5), dpi=120)
    ax = fig.add_subplot(111)
    ax.scatter(dA, dB, s=14, alpha=0.7)
    lim = [min(dA.min(), dB.min()), max(dA.max(), dB.max())]
    ax.plot(lim, lim, linestyle='--', linewidth=1)
    ax.set_xlabel('Distanz A')
    ax.set_ylabel('Distanz B')
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

def pca_bar(pointsA: Points, pointsB: Optional[Points], titleA: str, titleB: Optional[str], out_path: Path) -> Path:
    """Balkendiagramm der PCA-Varianzanteile (bis zu 3 Komponenten)."""
    def evr(points: Points) -> List[float]:
        try:
            from sklearn.decomposition import PCA
            C = _coords_array(points)
            p = PCA(n_components=min(3, C.shape[1])).fit(C)
            r = list(p.explained_variance_ratio_)
            return r + [0.0] * (3 - len(r))
        except Exception:
            return [np.nan, np.nan, np.nan]
    rA = evr(pointsA)
    rB = evr(pointsB) if pointsB is not None else None
    x = np.arange(3)
    w = 0.35
    fig = plt.figure(figsize=(6.5, 4.2), dpi=120)
    ax = fig.add_subplot(111)
    ax.bar(x - (w/2 if rB is not None else 0), rA, width=w, label=titleA)
    if rB is not None:
        ax.bar(x + w/2, rB, width=w, label=(titleB or "B"))
    ax.set_xticks(x); ax.set_xticklabels(["PC1","PC2","PC3"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Erklärte Varianz")
    ax.set_title("PCA-Varianzanteile")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

# ------------------------------ 3D Plot ------------------------------

def plot_points_3d(points: List[Tuple[str, float, float, float]],
                   title: str = "inverse-MDS Ergebnisse",
                   save_path: Optional[Path] = None,
                   show: bool = True,
                   axis_labels: Optional[Tuple[str, str, str]] = None) -> Path:
    """Erzeugt einen 3D-Scatter-Plot. Speichert optional ein PNG und zeigt das Fenster an."""
    if not points:
        raise ValueError("Keine Punkte zum Plotten übergeben.")

    names = [p[0] for p in points]
    X = [p[1] for p in points]
    Y = [p[2] for p in points]
    Z = [p[3] for p in points]

    fig = plt.figure(figsize=(7.5, 6.5), dpi=120)
    ax = fig.add_subplot(111, projection='3d')

    # Scatter (nutze ax.scatter; 3D-Achse akzeptiert (X,Y,Z); pyright/VSCode meckert fälschlich bei 'zs')
    ax.scatter(X, Y, Z, s=40, depthshade=True) # pyright: ignore[reportArgumentType]

    # Labels an die Punkte
    for (name, x, y, z) in points:
        ax.text(x, y, z, name, fontsize=8)

    # Achsen stylen
    ax.set_title(title)
    if axis_labels:
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_zlabel(axis_labels[2])
    else:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    # Achsenbegrenzungen fest auf [-1, 1] setzen
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    # dezentes Gitter
    ax.grid(True, linestyle=':')

    fig.tight_layout()

    out_path = save_path or (RESULTS_DIR / "plot.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)

    if show and matplotlib.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)

    return out_path


# ------------------------------ CLI ------------------------------

def main():
    parser = argparse.ArgumentParser(description="3D-Visualisierung & Analysen der inverse-MDS Ergebnisse")
    parser.add_argument("--file", type=str, default=None, help="Pfad zur CSV; Standard: neueste CSV aus results/")
    parser.add_argument("--file2", type=str, default=None, help="Zweite CSV (z. B. 2D vs 3D) für Vergleich")
    parser.add_argument("--save", type=str, default=None, help="Plot als PNG speichern (Pfad)")
    parser.add_argument("--show", action="store_true", help="Plotfenster anzeigen (falls GUI verfügbar)")
    parser.add_argument("--compare", action="store_true", help="Vergleichsmetriken & -plots erzeugen (wenn --file2 gesetzt)")
    args = parser.parse_args()

    if args.file:
        csv_path = Path(args.file)
    else:
        csv_path = latest_csv()
        if not csv_path:
            raise SystemExit("Keine CSV in results/ gefunden. Exportiere zuerst Ergebnisse.")

    points, labels = load_points_from_csv(csv_path)
    title = f"inverse-MDS — {csv_path.name}"
    save_path = Path(args.save) if args.save else None
    out = plot_points_3d(points, title=title, save_path=save_path, show=args.show, axis_labels=labels)
    print(f"Plot gespeichert unter: {out}")

    # Single-metrics
    m1 = metrics_single(points)
    print("Metriken (Set 1):", m1)

    # Optional: Vergleich
    if args.file2:
        csv_path2 = Path(args.file2)
        points2, labels2 = load_points_from_csv(csv_path2)
        if args.compare:
            comp = compare_embeddings(points, points2)
            print("Vergleich 1↔2:", comp)
            # Plots
            a1, b1 = _align_names(points, points2)
            shepard_plot(a1, b1, title="Distanzen: Set1 vs Set2", out_path=RESULTS_DIR / "shepard_1_vs_2.png")
            pca_bar(points, points2, titleA=csv_path.name, titleB=csv_path2.name, out_path=RESULTS_DIR / "pca_1_vs_2.png")
        else:
            # Falls kein --compare gesetzt ist, nur zweiten Plot rendern
            plot_points_3d(points2, title=f"inverse-MDS — {csv_path2.name}", save_path=RESULTS_DIR / "plot2.png", show=args.show, axis_labels=labels2)


if __name__ == "__main__":
    main()