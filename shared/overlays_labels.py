from typing import Optional
from PySide6.QtWidgets import QLabel, QPushButton
from PySide6.QtCore import Qt, QObject

import shared.geometry as geometry
import shared.state as state
from shared.const import *
import shared.overlays_utils as overlays_utils

def ensure_point_label(pid: str, label_filter) -> Optional[QLabel]:
    """Ensure a QLabel exists as overlay label for pid and return it."""
    if label_filter is None: 
        return
    if pid in state.point_labels:
        return state.point_labels[pid]
    lab = QLabel(pid, parent=state.VIEW)
    lab.setProperty('pid', pid)
    lab.installEventFilter(label_filter)
    lab.setStyleSheet("""
        color: #ffffff;
        background: rgba(0,0,0,140);
        border: 1px solid #666;
        border-radius: 4px;
        padding: 1px 4px;
        font-size: 12px;
        font-weight: 600;
    """)
    lab.setTextFormat(Qt.TextFormat.RichText)
    lab.setText(ALIGN_BAD_HTML.format(partner="?"))
    lab.hide()
    state.point_labels[pid] = lab
    return lab

def ensure_image_label(pid: str) -> Optional[QLabel]:
    """Return or create an image overlay label for a token id (Img#.1 / Img#.2)."""
    global VIEW
    if state.VIEW is None:
        return None
    cat = state.category_of(pid) if '.' in pid else pid
    pm = state.IMAGES_BY_CAT.get(cat)
    if pm is None or pm.isNull():
        return None
    if pid in state.image_labels:
        lab = state.image_labels[pid]
        lab.setPixmap(pm)
        return lab
    lab = QLabel(parent=state.VIEW)
    lab.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
    lab.setPixmap(pm)
    lab.hide()
    lab.lower()
    state.image_labels[pid] = lab
    return lab

def update_image_label(pid: str):
    """Position the per-token image overlay above the placed 3D point (debug-only)."""
    global VIEW
    if state.VIEW is None:
        return
    if pid not in state.placed_points:
        return
    lab = ensure_image_label(pid)
    if lab is None:
        return
    _, coords = state.placed_points[pid]
    if not geometry._is_point_visible_world(coords):
        lab.hide()
        return
    pr = geometry.project_point(coords) 
    if pr is None:
        lab.hide()
        return
    px, py = pr
    lab.adjustSize()
    lab.move(int(px - lab.width() // 2), int(py - lab.height() - IMAGE_OVER_POINT_MARGIN))
    lab.show()
    lab.lower()

def update_point_label(pid: str, label_filter: QObject | None = None):
    """Update overlay label text and position for a placed point id."""
    global VIEW
    if state.VIEW is None:
        return
    if pid not in state.placed_points:
        return
    lab = ensure_point_label(pid, label_filter)
    if lab is None:
        return
    item, coords = state.placed_points[pid]
    if not geometry._is_point_visible_world(coords):
        lab.hide()
        return
    pr = geometry.project_point(coords)
    if pr is None:
        lab.hide()
        return
    px, py = pr
    lab.setText(overlays_utils.alignment_indicator_for(pid))
    lab.adjustSize()
    lab.move(int(px - lab.width() // 2), int(py - lab.height() - 18))
    if '.' not in pid:
        lab.setToolTip("Rechtsklick: Paar aufteilen (zu .1 / .2)")
    lab.show()
    lab.raise_()

def update_all_point_labels(label_filter: QObject | None = None):
    for pid in list(state.placed_points.keys()):
        update_point_label(pid, label_filter)

    if getattr(state, "show_images", False):
        for pid in list(state.placed_points.keys()):
            update_image_label(pid)
    else:
        for pid in list(state.placed_points.keys()):
            lab = state.image_labels.get(pid)
            if lab:
                lab.hide()
                
def position_axis_labels(axis_label_x: QLabel, axis_label_y: QLabel, axis_label_z: QLabel):
    """Position the overlay axis labels near the projected axis midpoints."""
    if axis_label_z and axis_label_y and axis_label_z is None: 
        return  
    pts = {
        'x': (AXIS_LEN * 0.5, 0.0, 0.0),
        'y': (0.0, AXIS_LEN * 0.5, 0.0),
        'z': (0.0, 0.0, AXIS_LEN * 0.5),
    }
    labs = {'x': axis_label_x, 'y': axis_label_y, 'z': axis_label_z}
    for k in ('x', 'y', 'z'):
        res = geometry.project_point(pts[k])
        if res is None:
            continue
        x_px, y_px = res
        lab = labs[k]
        lab.adjustSize()
        if k == 'z':
            OFFSET_X = 10
            lab.move(x_px + OFFSET_X, y_px - lab.height() // 2)
        else:
            lab.move(x_px - lab.width() // 2, max(0, y_px - lab.height() - AXIS_LABEL_MARGIN))
        lab.raise_()
        
def apply_axis_labels(edit_x, edit_y, edit_z,
                      axis_label_x, axis_label_y, axis_label_z,
                      header_label):
    """Apply axis labels from edit fields to axis overlays + header."""
    tx = edit_x.text().strip()
    ty = edit_y.text().strip()
    tz = edit_z.text().strip()

    axis_label_x.setText(tx)
    axis_label_y.setText(ty)
    axis_label_z.setText(tz)

    for lab in (axis_label_x, axis_label_y, axis_label_z):
        lab.show()
        lab.raise_()

    header_label.setText(f"X: {tx}    Y: {ty}    Z: {tz}")
    position_axis_labels(axis_label_x, axis_label_y, axis_label_z)
