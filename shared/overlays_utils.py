import numpy as np
from typing import Optional
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame

from shared.const import *
import shared.state as state

def alignment_indicator_for(pid: str) -> str:
    """Return HTML alignment indicator for a point and its partner."""
    if pid not in state.placed_points or '.' not in pid:
        return pid
    partner = state.partner_of(pid)
    if not partner or partner not in state.placed_points:
        return ALIGN_BAD_HTML.format(partner=partner or "?")
    _, c_self = state.placed_points[pid]
    _, c_part = state.placed_points[partner]
    tol = state.get_z_tolerance()
    if abs(float(c_self[2]) - float(c_part[2])) <= tol:
        return ALIGN_OK_HTML.format(partner=partner)
    else:
        return ALIGN_BAD_HTML.format(partner=partner)

def update_point_color(pid: str):
    """Set both points of a category to green when aligned in Z, else yellow."""
    if pid not in state.placed_points or '.' not in pid:
        return
    partner = state.partner_of(pid)
    if not partner or partner not in state.placed_points:
        return

    _, c_self = state.placed_points[pid]
    _, c_part = state.placed_points[partner]
    tol = state.get_z_tolerance()

    color = np.array([[1.0, 1.0, 0.0, 1.0]])  # yellow
    if abs(float(c_self[2]) - float(c_part[2])) <= tol:
        color = np.array([[0.0, 1.0, 0.0, 1.0]])  # green
    item_self, _ = state.placed_points[pid]
    item_partner, _ = state.placed_points[partner]
    item_self.setData(color=color)
    item_partner.setData(color=color)
    
def _ensure_hover_preview() -> QLabel:
    """Ensure and return the global hover preview label."""
    global HOVER_PREVIEW_LABEL
    if state.HOVER_PREVIEW_LABEL is None:
        lab = QLabel(parent=state.VIEW)
        lab.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        lab.hide()
        HOVER_PREVIEW_LABEL = lab
    return HOVER_PREVIEW_LABEL

# This function previews the current category image near the mouse cursor in the 3D view.
# def _show_hover_preview(cat: str, view_pos: QPoint):
#     """Show small hover thumbnail near view_pos inside the scene."""
#     pm = IMAGES_BY_CAT.get(cat)
#     if pm is None or pm.isNull():
#         _hide_hover_preview()
#         return
#     lab = _ensure_hover_preview()
#     lab.setPixmap(pm)
#     lab.adjustSize()
#     x = int(view_pos.x() - lab.width() // 2)
#     y = int(view_pos.y() - lab.height() - IMAGE_OVER_POINT_MARGIN)
#     lab.move(x, y)
#     lab.show()
#     lab.lower()

def _show_hover_preview_over_dock(cat: str, point_dock: QFrame | None = None):
    """Show larger preview above the token dock for a category."""
    if point_dock is None:
        point_dock = state.POINT_DOCK if hasattr(state, "POINT_DOCK") else None
    if point_dock is None:
        return
    pm_orig = state.IMAGES_ORIG.get(cat) or state.IMAGES_BY_CAT.get(cat)
    if pm_orig is None or pm_orig.isNull():
        _hide_hover_preview()
        return
    lab = _ensure_hover_preview()
    pm_big = pm_orig.scaled(IMAGE_CONTAINER_WH, IMAGE_CONTAINER_WH,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)
    lab.setPixmap(pm_big)
    lab.adjustSize()
    dock_center_x = point_dock.x() + point_dock.width() // 2
    top_y = point_dock.y()
    x = int(dock_center_x - lab.width() // 2)
    y = int(top_y - lab.height() - IMAGE_OVER_POINT_MARGIN)
    lab.move(x, y)
    lab.show()
    lab.lower()

def _hide_hover_preview():
    if state.HOVER_PREVIEW_LABEL is not None:
        try:
            state.HOVER_PREVIEW_LABEL.hide()
        except Exception:
            pass
        
def set_preview_for_category(cat: Optional[str], preview_box: QLabel):
    """Update the right-side preview box with a scaled image for category."""
    if not cat:
        preview_box.setText("Image Preview")
        preview_box.setPixmap(QPixmap())
        return
    pm_orig = state.IMAGES_ORIG.get(cat) or state.IMAGES_BY_CAT.get(cat)
    if pm_orig is None or pm_orig.isNull():
        preview_box.setText("Image Preview")
        preview_box.setPixmap(QPixmap())
        return
    pm_scaled = pm_orig.scaled(preview_box.width()-12, preview_box.height()-12,
                               Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
    preview_box.setPixmap(pm_scaled)