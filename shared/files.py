import os
import random
import pathlib
from datetime import datetime
import csv
from PySide6.QtGui import QPixmap, Qt

import shared.scene_axes as scene_axes
import shared.state as state
import shared.tokens as token
import shared.log as log 
from shared.const import *

def pictures_dir() -> str:
    """Return the 'pictures' subfolder path relative to this script."""
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "pictures")

def load_images_for_categories():
    """
    Load images from the pictures/ folder and populate IMAGES_ORIG and IMAGES_BY_CAT.
    Scales thumbnails for overlay usage.
    """
    global IMAGES_BY_CAT, IMAGES_ORIG
    folder = pictures_dir()
    exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'}
    paths = []
    try:
        for p in pathlib.Path(folder).iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                paths.append(str(p))
    except FileNotFoundError:
        IMAGES_BY_CAT = {}
        return
    random.shuffle(paths)
    pixmaps = []
    for path in paths:
        pm = QPixmap(path)
        if not pm.isNull():
            pixmaps.append(pm)
    if not pixmaps:
        IMAGES_BY_CAT = {}
        return
    cats = token.token_categories()
    if len(pixmaps) > len(cats):
        pixmaps = pixmaps[:len(cats)]
    state.IMAGES_BY_CAT.clear()
    state.IMAGES_ORIG.clear()
    for i, cat in enumerate(cats):
        pm_orig = pixmaps[i % len(pixmaps)]
        state.IMAGES_ORIG[cat] = pm_orig
        pm_scaled = pm_orig.scaled(IMAGE_MAX_WH, IMAGE_MAX_WH,
                                   Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
        state.IMAGES_BY_CAT[cat] = pm_scaled

def export_results(btn_start, btn_submit,
                   edit_x, edit_y, edit_z,
                   combo_x_inline, combo_y_inline, combo_z_inline,
                   header_label, app):
    """Export combined points to CSV if all pairs are combined."""
    log.log_session_event("submitted, finished experiment")
    if not token.all_pairs_combined():
        token.update_submit_state(btn_submit, btn_start)
        return
    btn_start.setEnabled(True)
    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base, "results")
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    x_name = scene_axes.axis_display_name(edit_x, combo_x_inline, "X")
    y_name = scene_axes.axis_display_name(edit_y, combo_y_inline, "Y")
    z_name = scene_axes.axis_display_name(edit_z, combo_z_inline, "Z")
    rows = token.collect_combined_points_norm()
    csv_path = os.path.join(out_dir, f"embedding_{ts}.csv")
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Name", x_name, y_name, z_name])
            elapsed_seconds = None
            if log.start_time is not None:
                elapsed_seconds = (datetime.now() - log.start_time).total_seconds()
            for name, xn, yn, zn in rows:
                w.writerow([name, f"{xn:.6f}", f"{yn:.6f}", f"{zn:.6f}"])
            if elapsed_seconds is not None:
                w.writerow([])
                w.writerow(["Elapsed Time (2D-Experiment) (s)", f"{elapsed_seconds:.2f}"])
    except Exception:
        csv_path = os.path.join(base, f"embedding_{ts}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Name", x_name, y_name, z_name])
            elapsed_seconds = None
            if log.start_time is not None:
                elapsed_seconds = (datetime.now() - log.start_time).total_seconds()
            for name, xn, yn, zn in rows:
                w.writerow([name, f"{xn:.6f}", f"{yn:.6f}", f"{zn:.6f}"])
            if elapsed_seconds is not None:
                w.writerow([])
                w.writerow(["Elapsed Time (2D-Experiment) (s)", f"{elapsed_seconds:.2f}"])
    try:
        header_label.setText(f"Saved CSV: {os.path.basename(csv_path)}")
    except Exception:
        pass
    app.quit()
    # if EXPERIMENT_MODE_2D is True:
    #     base = os.path.dirname(os.path.abspath(__file__))
    #     path_3d = os.path.join(base, "experiment3d.py")

    #     try:
    #         win.close()
    #         subprocess.Popen([sys.executable, path_3d])
    #     except Exception as e:
    #         log.log_to_console(f"Failed to start 3D experiment: {e}")