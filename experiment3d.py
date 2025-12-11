import sys
import os
import random
from glob import glob
import csv
from datetime import datetime
from typing import Dict, Optional

import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QFrame, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QCheckBox, QMainWindow, QSizePolicy, QGridLayout, QScrollArea, QPlainTextEdit, QSlider
)
from PySide6.QtGui import QVector3D, QDrag, QCursor, QPixmap, QFont, QKeySequence, QShortcut
from PySide6.QtCore import Qt, QTimer, QMimeData, QPoint, QObject
from pyqtgraph.opengl import GLViewWidget, GLLinePlotItem, GLGridItem

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Constant definitions
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

HAS_GLTEXT = False
AXIS_LEN = 10.0
TICK_STEP = 1.0
TICK_SIZE = 0.10
GAP_LEN = 0.15
DASH_LEN = 0.30
LABEL_WIDTH = 80
INPUT_WIDTH = 120
POINT_SIZE = 16
Z_ALIGN_EPS = 0.5
LABEL_OVER_POINT_MARGIN = 8
RANGE_LABEL_MARGIN = 10
AXIS_LABEL_MARGIN = 15
CUBE_WIDTH = 1
LATTICE_WIDTH = 1
BTN_SPACING = 8
BTN_MARGIN = 10
PARAM_PANEL_GAP = -5
COMBO_WIDTH = 150
ROW_SPACING = 15
CONTROL_H = 28
GAP_H = 12
PREVIEW_TOP_OFFSET = 22
ACTIONS_TOP_OFFSET = -16
SCENE_TOP_OFFSET = 22
SCENE_FIXED_HEIGHT = 760
HOVER_PREVIEW_MARGIN = 30
BL_MARGIN_X, BL_MARGIN_Y = 10, 10
BR_MARGIN_X, BR_MARGIN_Y = 12, 12
POINT_COLOR = np.array([[1.0, 1.0, 0.0, 1.0]])
CUBE_COLOR = (120, 120, 120, 0.2)
LATTICE_COLOR = (120, 120, 120, 0.2)
PLANE_OFFSETS = {'xy': 0.0, 'xz': 0.0, 'yz': 0.0}

# Fixed size for token container + scroll area
TOKEN_CONTAINER_W = 140
TOKEN_CONTAINER_H = 345

IMAGE_OVER_POINT_MARGIN = 32
IMAGE_MAX_WH = 40
IMAGE_CONTAINER_WH = 120
LABEL_SCREEN_MARGIN = 24
VIS_DOT_THRESHOLD = 0.0

ALIGN_OK_HTML = "<span style='color:#7CFC00'>✅ {partner}</span>"
ALIGN_BAD_HTML = "<span style='color:#ff6666'>❌ {partner}</span>"

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Global state variables
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

working: Dict[str, Optional[float]] = {'x': None, 'y': None, 'z': None}
placed_points: Dict[str, tuple] = {}
point_labels: Dict[str, QLabel] = {}
pair_lines: Dict[str, GLLinePlotItem] = {}
helper_lines = {}
points = []
cube_items = []
point_tokens = []
lattice_items = []
current_plane = 'xy'
placement_phase = 1
start_time = None
LOCK_CAMERA = False

IMAGES_BY_CAT: Dict[str, QPixmap] = {}
IMAGES_ORIG: Dict[str, QPixmap] = {}
image_labels: Dict[str, QLabel] = {}
HOVER_PREVIEW_LABEL: Optional[QLabel] = None
IS_DOCK_DRAG = False

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

def log_to_console(text):
    text = f"[{datetime.now().strftime('%H:%M:%S')}] {text}"
    console_box.appendPlainText(text)
    
def write_log_to_file(log_path: str, text: str):
    try:   
        with open(log_path, 'a', encoding='utf-8', newline='') as f:
            f.write(text + '\n')
    except Exception as e:
        log_to_console(f"Failed to write log: {e}")
    
def log_session_event(event: str):
    if not start_time:
        return
    elapsed = (datetime.now() - start_time).total_seconds()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    base = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"session_log_{start_time.strftime('%Y%m%d_%H%M%S')}.csv")
    write_log_to_file(log_path, f"{timestamp},{elapsed:.2f},{event}")
    log_to_console(event)

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# File-system helpers and image loading
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

def _pictures_dir() -> str:
    """Return the 'pictures' subfolder path relative to this script."""
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "pictures")

def _token_categories() -> list[str]:
    """Return category names derived from point token ids or default list."""
    cats = []
    try:
        for t in point_tokens: 
            cat = t.pid.split('.')[0]
            if cat not in cats:
                cats.append(cat)
    except Exception:
        pass
    if not cats:
        cats = [f"{i}. image" for i in range(1, 11)]
    return cats

def _load_images_for_categories():
    """
    Load images from the pictures/ folder and populate IMAGES_ORIG and IMAGES_BY_CAT.
    Scales thumbnails for overlay usage.
    """
    import pathlib
    global IMAGES_BY_CAT, IMAGES_ORIG
    folder = _pictures_dir()
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
    cats = _token_categories()
    if len(pixmaps) > len(cats):
        pixmaps = pixmaps[:len(cats)]
    IMAGES_BY_CAT.clear()
    IMAGES_ORIG.clear()
    for i, cat in enumerate(cats):
        pm_orig = pixmaps[i % len(pixmaps)]
        IMAGES_ORIG[cat] = pm_orig
        pm_scaled = pm_orig.scaled(IMAGE_MAX_WH, IMAGE_MAX_WH,
                                   Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
        IMAGES_BY_CAT[cat] = pm_scaled

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Event filter and small widget classes
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

class _PointLabelFilter(QObject):
    """Event filter for overlay point labels."""

    def eventFilter(self, obj, ev):
        try:
            pass
        except Exception:
            pass
        return False

POINT_LABEL_FILTER = _PointLabelFilter()

class SceneView(GLViewWidget):
    """GL view widget with drag/drop, picking and hover preview handling."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.dragging_pid = None
        self.selected_pid = None
        self.drag_offset = (0.0, 0.0)
        self.drag_pending = False
        self.drag_z0 = None
        self.drag_plane_origin = None
        self.drag_plane_normal = None
        # --- Freeze XY after scroll support ---
        self.freeze_xy_after_scroll = False
        self.freeze_xy_pos = None
        self.hover_pid = None

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasFormat('application/x-point-id'):
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dragMoveEvent(self, ev):
        if ev.mimeData().hasFormat('application/x-point-id'):
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dropEvent(self, ev):
        if not ev.mimeData().hasFormat('application/x-point-id'):
            ev.ignore()
            return
        try:
            point_id = ev.mimeData().data('application/x-point-id').data().decode('utf-8')
        except Exception:
            ev.ignore()
            return

        pos = ev.position().toPoint()
        px, py = int(pos.x()), int(pos.y())
        p0, p1 = screen_to_world_ray(px, py)
        if p0 is None or p1 is None:
            ev.ignore()
            return

        # intersect ray with mid-plane z = AXIS_LEN * 0.5
        zmid = AXIS_LEN * 0.5
        dir = QVector3D(p1.x() - p0.x(), p1.y() - p0.y(), p1.z() - p0.z())
        if abs(dir.z()) < 1e-9:
            ev.ignore()
            return
        t = (zmid - p0.z()) / dir.z()
        if t < 0:
            ev.ignore()
            return
        hit = p0 + dir * t
        x, y, z = clamp_to_cube(float(hit.x()), float(hit.y()), float(hit.z()))

        _set_point_position(point_id, (x, y, z))
        _update_helper_lines(point_id)
        _mark_token_placed(point_id)
        ev.acceptProposedAction()

    def mousePressEvent(self, ev):
        """
        Support picking of projected 2D tokens to initiate drag of an already-placed token.
        Otherwise forward to base implementation (camera interaction).
        """
        pos = ev.position().toPoint()
        mx, my = pos.x(), pos.y()

        # Find nearest placed point via 2D projection
        hit_pid = None
        best_d2 = 1e9
        for pid, (_item, coords) in placed_points.items():
            pr = project_point(coords)
            if pr is None:
                continue
            px, py = pr
            dx = float(px - mx)
            dy = float(py - my)
            d2 = dx*dx + dy*dy
            if d2 < best_d2:
                best_d2 = d2
                hit_pid = pid

        pick_r = float(max(16.0, POINT_SIZE + 6.0))
        # Left-click drag logic (previously right-click drag logic)
        if ev.button() == Qt.MouseButton.LeftButton:
            if hit_pid is not None and best_d2 <= (pick_r * pick_r):
                self.dragging_pid = hit_pid
                # _highlight_token(hit_pid)
                _update_helper_lines(hit_pid)
                # No pending, no offset, no snap — purely drag on movement
                self.drag_pending = False
                self.drag_offset = (0.0, 0.0)
                item, coords_old = placed_points.get(hit_pid, (None, [0,0,AXIS_LEN*0.5]))
                self.drag_z0 = float(coords_old[2])
                _update_helper_lines(hit_pid)
                # Reset freeze XY state for new drag
                self.freeze_xy_after_scroll = False
                self.freeze_xy_pos = None
                self.drag_offset = (mx, my)
                # --- RAISE IMAGE + LABEL WHEN DRAGGING A PLACED POINT ---
                try:
                    # Raise point label
                    if hit_pid in point_labels:
                        point_labels[hit_pid].raise_()
                except Exception:
                    pass

                try:
                    # Raise image overlay
                    if hit_pid in image_labels:
                        image_labels[hit_pid].raise_()
                except Exception:
                    pass

                try:
                    # Raise both partner images if .1/.2
                    cat = hit_pid.split('.')[0]
                    for suffix in ("", ".1", ".2"):
                        key = f"{cat}{suffix}"
                        if key in image_labels:
                            image_labels[key].raise_()
                        if key in point_labels:
                            point_labels[key].raise_()
                except Exception:
                    pass
                # ---------------------------------------------------------
                return
        if not LOCK_CAMERA:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self.dragging_pid is not None:
            pos = ev.position().toPoint()
            mx, my = pos.x(), pos.y()
            # If XY is frozen after scroll, keep XY and update only if user moves significantly
            if self.freeze_xy_after_scroll:
                dx = abs(mx - self.drag_offset[0])
                dy = abs(my - self.drag_offset[1])
                if dx < 3 and dy < 3:
                    if self.freeze_xy_pos is not None:
                        fx, fy = self.freeze_xy_pos
                        zplane = float(self.drag_z0 if self.drag_z0 is not None else AXIS_LEN * 0.5)
                        _set_point_position(self.dragging_pid, (fx, fy, zplane))
                    # Keep preview active while dragging
                    if self.dragging_pid is not None:
                        cat = self.dragging_pid.split('.')[0]
                        _set_preview_for_category(cat)
                    return
                else:
                    self.freeze_xy_after_scroll = False
                    self.freeze_xy_pos = None
            px = mx
            py = my
            p0, p1 = screen_to_world_ray(int(px), int(py))
            if p0 is not None and p1 is not None:
                zplane = float(self.drag_z0 if self.drag_z0 is not None else AXIS_LEN * 0.5)
                dir = QVector3D(float(p1.x()-p0.x()), float(p1.y()-p0.y()), float(p1.z()-p0.z()))
                if abs(dir.z()) > 1e-9:
                    t = (zplane - p0.z()) / dir.z()
                    if t >= 0:
                        hit = p0 + dir * t
                        x_new, y_new, _ = clamp_to_cube(float(hit.x()), float(hit.y()), zplane)
                        _set_point_position(self.dragging_pid, (x_new, y_new, zplane))
            # Keep preview active while dragging
            if self.dragging_pid is not None:
                cat = self.dragging_pid.split('.')[0]
                _set_preview_for_category(cat)
            return
        # Hover picking: show hover preview for nearest visible point if within radius.
        try:
            pos = ev.position().toPoint()
            mx, my = pos.x(), pos.y()
            hit_pid = None
            best_d2 = 1e9
            best_proj = None
            for pid, (_item, coords) in placed_points.items():
                if not _is_point_visible_world(coords):
                    continue
                proj = project_point(coords)
                if proj is None:
                    continue
                px, py = proj
                dx = float(px - mx)
                dy = float(py - my)
                d2 = dx*dx + dy*dy
                if d2 < best_d2:
                    best_d2 = d2
                    hit_pid = pid
                    best_proj = (px, py)
            pick_r = float(max(16.0, POINT_SIZE + 6.0))
            prev_hover = self.hover_pid
            self.hover_pid = hit_pid if (hit_pid is not None and best_d2 <= (pick_r * pick_r)) else None
            # Only rebuild helper lines if hover pid changes
            if self.hover_pid != prev_hover:
                if prev_hover is not None:
                    _update_helper_lines(prev_hover)
                if self.hover_pid is not None:
                    _update_helper_lines(self.hover_pid)
            if self.hover_pid is not None:
                cat = self.hover_pid.split('.')[0]
                _set_preview_for_category(cat)
                # Token appearance on hover
                item, _ = placed_points[self.hover_pid]
                item.setData(color=np.array([[1,1,1,1]]), size=POINT_SIZE+6)
                # Reset others
                for pid, (itm, _) in placed_points.items():
                    if pid != self.hover_pid:
                        itm.setData(color=POINT_COLOR, size=POINT_SIZE)
            else:
                # Reset all token appearance
                for pid, (item, _) in placed_points.items():
                    item.setData(color=POINT_COLOR, size=POINT_SIZE)
        except Exception:
            pass
        if LOCK_CAMERA:
            return
        super().mouseMoveEvent(ev)

    def wheelEvent(self, ev):
        pid = self.dragging_pid or self.hover_pid
        if pid is None:
            return
        delta = ev.angleDelta().y() / 120.0
        item, coords = placed_points.get(pid, (None, None))
        if coords is None:
            return
        x, y, z = coords
        z += float(delta) * 0.3
        x, y, z = clamp_to_cube(x, y, z)
        # Update drag plane origin z so dragging plane moves with the point
        if self.drag_plane_origin is not None:
            self.drag_plane_origin.setZ(z)
        # Update drag_z0 so dragging logic continues from new height
        self.drag_z0 = z
        # Freeze XY after scroll
        self.freeze_xy_after_scroll = True
        self.freeze_xy_pos = (x, y)
        _set_point_position(pid, (x, y, z))
        _update_helper_lines(pid)

    def keyPressEvent(self, ev):
        if self.selected_pid is None:
            super().keyPressEvent(ev)
            return
        # W/S key movement disabled; only wheelEvent changes Z.
        super().keyPressEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self.drag_pending = False
            self.drag_z0 = None
            # Clear freeze XY state on release
            self.freeze_xy_after_scroll = False
            self.freeze_xy_pos = None
            if self.dragging_pid in helper_lines:
                for it in helper_lines[self.dragging_pid]:
                    try:
                        view.removeItem(it)
                    except Exception:
                        pass
                helper_lines.pop(self.dragging_pid, None)
            self.dragging_pid = None
            return
        super().mouseReleaseEvent(ev)

    def leaveEvent(self, ev):
        # Do NOT hide preview on leaving scene — keep last preview active
        self.hover_pid = None
        super().leaveEvent(ev)

class DraggableToken(QLabel):
    """Small QLabel that acts as draggable token in the left dock."""

    def __init__(self, pid: str, parent=None):
        display = pid
        super().__init__(display, parent)
        self.pid = pid
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        self.setStyleSheet(_token_style_mode('disabled'))
        self.setFixedSize(110, 18)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)

    def enterEvent(self, ev):
        # Show preview as soon as we hover over a token
        try:
            cat = self.pid.split('.')[0]
            _show_hover_preview_over_dock(cat)
        except Exception:
            pass
        try:
            check_hover_cb.setChecked(True)
            self.raise_()
        except Exception:
            pass
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        # Do NOT hide preview on leaving token — keep last preview active
        super().leaveEvent(ev)

    def mouseMoveEvent(self, ev):
        """
        Keep preview ALWAYS visible while hovering or dragging.
        Only the category of this token should drive the preview.
        """
        try:
            cat = self.pid.split('.')[0]
            _show_hover_preview_over_dock(cat)   # <- forces preview to stay alive
        except Exception:
            pass
        try:
            self.raise_()
        except Exception:
            pass
        super().mouseMoveEvent(ev)

    def mousePressEvent(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        if not self.isEnabled():
            return

        global IS_DOCK_DRAG
        IS_DOCK_DRAG = True   # start dock-drag mode

        # FIX 1: always use correct category ("1", "2", ...)
        cat = self.pid.split('.')[0]

        # FIX 2: keep dock preview alive while clicking
        _show_hover_preview_over_dock(cat)

        # ---- BRING THIS TOKEN AND OVERLAYS TO FRONT ----
        try:
            self.raise_()
        except Exception:
            pass
        try:
            # Bring image overlay to front
            cat = self.pid.split('.')[0]
            for suffix in ("", ".1", ".2"):
                key = f"{cat}{suffix}"
                if key in image_labels:
                    image_labels[key].raise_()
                if key in point_labels:
                    point_labels[key].raise_()
        except Exception:
            pass
        # -----------------------------------------------

        drag = QDrag(self)
        mime = QMimeData()
        mime.setData('application/x-point-id', self.pid.encode('utf-8'))
        drag.setMimeData(mime)

        # Drag pixmap
        pm = IMAGES_BY_CAT.get(cat)
        if pm is not None and not pm.isNull():
            drag.setPixmap(pm)
            hs = QPoint(pm.width() // 2, int(pm.height() * 0.8))
            drag.setHotSpot(hs)

        # DO NOT CALL _set_preview_for_category() here – wrong preview!

        drag.exec(Qt.DropAction.MoveAction)

    IS_DOCK_DRAG = False

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# UI initialization
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

app = QApplication(sys.argv)
try:
    app.setFont(QFont("SF Pro Text", 12))
except Exception:
    try:
        app.setFont(QFont(".SF NS Text", 12))
    except Exception:
        pass

win = QMainWindow()
# win.setWindowTitle("3D inverse-MDS for embedding data")

central = QWidget()
root = QVBoxLayout(central)
root.setContentsMargins(10, 10, 10, 10)
root.setSpacing(10)
win.setCentralWidget(central)

main_row = QHBoxLayout()
main_row.setContentsMargins(0, 0, 0, 0)
main_row.setSpacing(GAP_H)
root.addLayout(main_row, 1)

LEFT_W = 120
left_col = QFrame()
left_col.setFrameShape(QFrame.Shape.StyledPanel)
left_col.setStyleSheet("QFrame { border: 0px solid #666; border-radius: 8px; }")
left_col.setMinimumWidth(LEFT_W)
left_v = QVBoxLayout(left_col)
left_v.setContentsMargins(10, 10, 10, 10)
left_v.setSpacing(10)

actions_row = QWidget(left_col)
actions_h = QHBoxLayout(actions_row)
actions_h.setContentsMargins(0, 0, 0, 0)
actions_h.setSpacing(8)
actions_row = QWidget(left_col)
actions_h = QHBoxLayout(actions_row)
actions_h.setContentsMargins(0, 0, 0, 0)
actions_h.setSpacing(8)

preview_box = QLabel()
preview_box.setFixedSize(230, 210)
preview_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
preview_box.setStyleSheet("""
    QLabel {
        background: rgba(255,255,255,0.1);
        color: #ddd;
        border: 1px solid #888;
        border-radius: 6px;
    }
""")
preview_box.setText("Image Preview")

preview_label = QLabel("Preview:")
preview_label.setStyleSheet("color: #fff; font-size: 14px; font-weight: 600; background: transparent;")
preview_label.adjustSize()
preview_label.show()

main_row.addWidget(left_col, 0)

view = SceneView()
view.setBackgroundColor('k')
view.setCameraPosition(distance=8)
view.setCameraParams(fov=60)

view_wrap = QWidget()
vw_lay = QVBoxLayout(view_wrap)
vw_lay.setContentsMargins(0, SCENE_TOP_OFFSET, 10, 0)
vw_lay.setSpacing(0)
vw_lay.addWidget(view)
view_wrap.setFixedHeight(SCENE_FIXED_HEIGHT)
main_row.addWidget(view_wrap, 1, Qt.AlignmentFlag.AlignTop)

#######################################################
#######################################################
#######################################################

# Add a top-left title label to the 3D view
title_label = QLabel("Inverse MDS", parent=win)
title_label.setStyleSheet("color: white; font-size: 26px; font-weight: 700; background: transparent;")
title_label.adjustSize()
title_label.move(20, 30)
title_label.raise_()

condition_label = QLabel("3D Condition", parent=view)
condition_label.setStyleSheet("color: white; font-size: 16px; background: transparent; font-weight: bold;")
condition_label.adjustSize()
condition_label.move(10, 10)
condition_label.raise_()

line_top = QFrame(parent=win)
line_top.setFrameShape(QFrame.Shape.HLine)
line_top.setFrameShadow(QFrame.Shadow.Plain)
line_top.setStyleSheet("color: gray; background-color: transparent;")
line_top.setFixedWidth(370)
line_top.move(20, 60)

#######################################################
#######################################################
#######################################################

# Participant name input (First + Last Name)
name_label = QLabel("Participant:", parent=win)
name_label.setStyleSheet("color: white; font-size: 16px; background: transparent; font-weight: bold;")
name_label.adjustSize()
name_label.move(20, 90)
name_label.raise_()

name_input = QLineEdit(parent=win)
name_input.setPlaceholderText("First name, Last name")
name_input.setFixedWidth(200)
name_input.setStyleSheet(
"""color: #000000;
    background: #f5f5f5;
    border: 1px solid black;
    border-radius: 6px;
    padding: 4px 8px;
""")
name_input.move(20, 115)
name_input.raise_()

# Age selection slider
age_label = QLabel("Age:", parent=win)
age_label.setStyleSheet("color: white; font-size: 16px; background: transparent; font-weight: bold;")
age_label.adjustSize()
age_label.move(230, 90)
age_label.raise_()

age_slider = QSlider(Qt.Orientation.Horizontal, parent=win)
age_slider.setMinimum(18)
age_slider.setMaximum(90)
age_slider.setValue(24)
age_slider.setFixedWidth(160)
age_slider.move(230, 115)
age_slider.raise_()

age_value = QLabel("24", parent=win)
age_value.setStyleSheet("color: white; font-size: 16px; background: transparent;")
age_value.adjustSize()
age_value.move(270, 90)
age_value.raise_()

middle_line = QFrame(parent=win)
middle_line.setFrameShape(QFrame.Shape.HLine)
middle_line.setFrameShadow(QFrame.Shadow.Plain)
middle_line.setStyleSheet("color: gray; background-color: transparent;")
middle_line.setFixedWidth(370)
middle_line.move(20, 150)

#######################################################
#######################################################
#######################################################

collapse_btn = QPushButton("▼", parent=win)
collapse_btn.setFixedSize(20, 20)
collapse_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
collapse_btn.setStyleSheet("""
    QPushButton {
        color: white;
        background: transparent;
        border: none;
        font-size: 12px;
        padding: 0px;
    }
    QPushButton:hover {
        background: rgba(255,255,255,0.1);
    }
""")
collapse_btn.move(105, 180)
collapse_btn.raise_()

is_collapsed = [False]  # Use list to allow modification in nested function

def _toggle_checklist():
    if is_collapsed[0]:
        # Expand
        collapse_btn.setText("▼")
        status_label.show()
        set_name_label.show()
        name_cb.show()
        check_hover_label.show()
        check_hover_cb.show()
        xyz_rotate_label.show()
        xyz_cb.show()
        stimuli_drag_label.show()
        stimuli_cb.show()
        start_label.show()
        start_cb.show()
        counter_label.show()
        line_bottom.move(20, 350)
        is_collapsed[0] = False
    else:
        # Collapse
        collapse_btn.setText("▶")
        status_label.show()
        set_name_label.hide()
        name_cb.hide()
        check_hover_label.hide()
        check_hover_cb.hide()
        xyz_rotate_label.hide()
        xyz_cb.hide()
        stimuli_drag_label.hide()
        stimuli_cb.hide()
        start_label.hide()
        start_cb.hide()
        counter_label.hide()
        line_bottom.move(20, 200)
        is_collapsed[0] = True

collapse_btn.clicked.connect(_toggle_checklist)

status_label = QLabel("Checklist:", parent=win)
status_label.setStyleSheet("color: white; font-size: 16px; background: transparent; font-weight: bold; border: none;")
status_label.adjustSize()
status_label.move(20, 180)
status_label.raise_()

set_name_label = QLabel("1. Set Name and Age:", parent=win)
set_name_label.setStyleSheet("color: lightgray; font-size: 14px; background: transparent;")
set_name_label.adjustSize()
set_name_label.move(20, 210)
set_name_label.raise_()

name_cb = QCheckBox("", parent=win)
name_cb.setStyleSheet("color: lightgray; font-size: 14px; background: transparent;")
name_cb.setEnabled(False)
name_cb.move(340, 205)
name_cb.raise_()

check_hover_label = QLabel("2. Hover over Stimuli to preview images:", parent=win)
check_hover_label.setStyleSheet("color: lightgray; font-size: 14px; background: transparent;")
check_hover_label.adjustSize()
check_hover_label.move(20, 240)
check_hover_label.raise_()

check_hover_cb = QCheckBox("", parent=win)
check_hover_cb.setStyleSheet("color: lightgray; font-size: 14px; background: transparent;")
check_hover_cb.setEnabled(False)
check_hover_cb.move(340, 235)
check_hover_cb.raise_()

xyz_rotate_label = QLabel("3. Rotate xyz axis or set default view (^D):", parent=win)
xyz_rotate_label.setStyleSheet("color: lightgray; font-size: 14px; background: transparent;")
xyz_rotate_label.adjustSize()
xyz_rotate_label.move(20, 270)
xyz_rotate_label.raise_()

xyz_cb = QCheckBox("", parent=win) 
xyz_cb.setStyleSheet("color: lightgray; font-size: 14px; background: transparent;")
xyz_cb.setEnabled(False)
xyz_cb.move(340, 265)
xyz_cb.raise_()

stimuli_drag_label = QLabel("4. Drag stimuli into the 3D space and adjust:", parent=win)
stimuli_drag_label.setStyleSheet("color: lightgray; font-size: 14px; background: transparent;")
stimuli_drag_label.adjustSize()
stimuli_drag_label.move(20, 300)
stimuli_drag_label.raise_()

stimuli_cb = QCheckBox("", parent=win)  
stimuli_cb.setStyleSheet("color: lightgray; font-size: 14px; background: transparent;")
stimuli_cb.setEnabled(False)
stimuli_cb.move(340, 295)
stimuli_cb.raise_()

start_label = QLabel("5. Press \"Start\" to start the Experiment:", parent=win)
start_label.setStyleSheet("color: lightgray; font-size: 14px; background: transparent;")
start_label.adjustSize()
start_label.move(20, 330)
start_label.raise_()

start_cb = QCheckBox("", parent=win)
start_cb.setStyleSheet("color: lightgray; font-size: 14px; background: transparent;")
start_cb.setEnabled(False)
start_cb.move(340, 325)
start_cb.raise_()

counter_label = QLabel("(0/5)", parent=win)
counter_label.setStyleSheet("color: lightgray; font-size: 12px; background: transparent;")
counter_label.adjustSize()
counter_label.move(365, 332)
counter_label.raise_()

line_bottom = QFrame(parent=win)
line_bottom.setFrameShape(QFrame.Shape.HLine)
line_bottom.setFrameShadow(QFrame.Shadow.Plain)
line_bottom.setStyleSheet("color: gray; background-color: transparent;")
line_bottom.setFixedWidth(370)
line_bottom.move(20, 350)

# Update displayed age dynamically
age_slider.valueChanged.connect(lambda v: age_value.setText(str(v)))
name_input.textChanged.connect(lambda: (
    name_cb.setChecked(name_input.text().strip() != ""),
))

#######################################################
#######################################################
#######################################################

def _set_preview_for_category(cat: Optional[str]):
    """Update the right-side preview box with a scaled image for category."""
    if not cat:
        preview_box.setText("Image Preview")
        preview_box.setPixmap(QPixmap())
        return
    pm_orig = IMAGES_ORIG.get(cat) or IMAGES_BY_CAT.get(cat)
    if pm_orig is None or pm_orig.isNull():
        preview_box.setText("Image Preview")
        preview_box.setPixmap(QPixmap())
        return
    pm_scaled = pm_orig.scaled(preview_box.width()-12, preview_box.height()-12,
                               Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
    preview_box.setPixmap(pm_scaled)

win.setFixedSize(1400, 800)

def _center_on_screen():
    """Center main window on primary available geometry (macOS-friendly)."""
    try:
        screen = win.screen() or app.primaryScreen()
        geo = screen.availableGeometry()
        x = geo.x() + (geo.width() - win.width()) // 2
        y = geo.y() + (geo.height() - win.height()) // 2
        win.move(x, y)
    except Exception:
        try:
            fg = win.frameGeometry()
            center = app.primaryScreen().availableGeometry().center()
            fg.moveCenter(center)
            win.move(fg.topLeft())
        except Exception:
            pass

win.show()
_center_on_screen()
try:
    if win.windowHandle():
        win.windowHandle().screenChanged.connect(lambda *_: _center_on_screen())
except Exception:
    pass

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Geometry / projection / raycasting / time helpers
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

def _start_experiment():
    global start_time
    
    start_time = datetime.now()
    btn_start.setEnabled(False)
    start_cb.setChecked(True)
    cb_lock.setChecked(True)
    name_input.setDisabled(True)
    age_slider.setDisabled(True)

    bool(_toggle_checklist() == 0)
    set_view_default()
    _update_progress_counter()
    log_session_event("Experiment started...")

def project_point(p):
    """Project a world 3D point to 2D view pixel coordinates (x, y)."""
    try:
        vm = view.viewMatrix()
        pm = view.projectionMatrix()
        m = pm * vm
        v = QVector3D(float(p[0]), float(p[1]), float(p[2]))
        ndc = m.map(v)
        ndc_x, ndc_y = float(ndc.x()), float(ndc.y())
        px = int((ndc_x + 1.0) * 0.5 * view.width())
        py = int((1.0 - (ndc_y + 1.0) * 0.5) * view.height())
        return px, py
    except Exception:
        return None

def _camera_position_vec3():
    """Return camera position as numpy array or None if unavailable."""
    try:
        p = view.cameraPosition()
        return np.array([float(p.x()), float(p.y()), float(p.z())], dtype=float)
    except Exception:
        return None

def _camera_forward_vec3():
    """Compute normalized forward vector from camera to view center, or None."""
    try:
        cp = _camera_position_vec3()
        ctr = view.opts.get('center')
        if cp is None or ctr is None:
            return None
        c = np.array([float(ctr.x()), float(ctr.y()), float(ctr.z())], dtype=float)
        f = c - cp
        n = np.linalg.norm(f)
        if n <= 1e-9:
            return None
        return f / n
    except Exception:
        return None

def _is_point_visible_world(coords):
    """Return False if a world point is behind camera or projected outside screen margins."""
    try:
        cp = _camera_position_vec3()
        fwd = _camera_forward_vec3()
        pr = project_point(coords)
        if pr is None:
            return False
        x, y = pr
        w, h = view.width(), view.height()
        m = LABEL_SCREEN_MARGIN
        within = (m <= x <= w - m) and (m <= y <= h - m)
        if cp is None or fwd is None:
            return within
        pt = np.array([float(coords[0]), float(coords[1]), float(coords[2])], dtype=float)
        v = pt - cp
        if float(np.dot(v, fwd)) <= VIS_DOT_THRESHOLD:
            return False
        return within
    except Exception:
        return True

def screen_to_world_ray(px: int, py: int):
    """Return (near_world_qvec, far_world_qvec) for a screen pixel (px,py)."""
    w = max(1, view.width())
    h = max(1, view.height())
    nx = 2.0 * px / w - 1.0
    ny = 1.0 - 2.0 * py / h
    vm = view.viewMatrix()
    pm = view.projectionMatrix()
    m = pm * vm
    try:
        inv = m.inverted()[0]
    except Exception:
        return None, None
    near_ndc = QVector3D(nx, ny, -1.0)
    far_ndc = QVector3D(nx, ny, 1.0)
    near_w = inv.map(near_ndc)
    far_w = inv.map(far_ndc)
    return near_w, far_w

def intersect_with_plane(p0: QVector3D, p1: QVector3D, plane: str):
    """Intersect ray (p0 -> p1) with one of 'xy','xz','yz' planes. Return (x,y,z) or None."""
    dir = QVector3D(float(p1.x()-p0.x()), float(p1.y()-p0.y()), float(p1.z()-p0.z()))
    if plane == 'xy':
        if abs(dir.z()) < 1e-9:
            return None
        z0 = float(PLANE_OFFSETS['xy'])
        t = (z0 - p0.z()) / dir.z()
    elif plane == 'xz':
        if abs(dir.y()) < 1e-9:
            return None
        y0 = float(PLANE_OFFSETS['xz'])
        t = (y0 - p0.y()) / dir.y()
    else:  # 'yz'
        if abs(dir.x()) < 1e-9:
            return None
        x0 = float(PLANE_OFFSETS['yz'])
        t = (x0 - p0.x()) / dir.x()
    if t < 0:
        return None
    hit = p0 + dir * t
    return float(hit.x()), float(hit.y()), float(hit.z())

def intersect_with_plane_t(p0: QVector3D, p1: QVector3D, plane: str):
    """Like intersect_with_plane but returns (t, (x,y,z)) to allow nearest-candidate selection."""
    dir = QVector3D(float(p1.x()-p0.x()), float(p1.y()-p0.y()), float(p1.z()-p0.z()))
    if plane == 'xy':
        if abs(dir.z()) < 1e-9:
            return None
        z0 = float(PLANE_OFFSETS['xy'])
        t = (z0 - p0.z()) / dir.z()
    elif plane == 'xz':
        if abs(dir.y()) < 1e-9:
            return None
        y0 = float(PLANE_OFFSETS['xz'])
        t = (y0 - p0.y()) / dir.y()
    else:
        if abs(dir.x()) < 1e-9:
            return None
        x0 = float(PLANE_OFFSETS['yz'])
        t = (x0 - p0.x()) / dir.x()
    if t < 0:
        return None
    hit = p0 + dir * t
    return float(t), (float(hit.x()), float(hit.y()), float(hit.z()))

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Axes, ticks and grid helpers
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

def _axis_segment(p0, p1, color=(1,1,1,1), width=3):
    pts = np.array([p0, p1], dtype=float)
    return GLLinePlotItem(pos=pts, color=color, width=width, mode='lines')

def _auto_tick_step(L: float) -> float:
    """Choose a readable tick step based on axis length L."""
    if L <= 10:
        return 1.0
    if L <= 20:
        return 2.0
    return max(1.0, round(L / 10.0, 1))

def _build_axis_solid(axis: str, L: float, color=(1,1,1,1), width=3):
    if axis == 'x':
        return [_axis_segment((0,0,0), (L,0,0), color=color, width=width)]
    elif axis == 'y':
        return [_axis_segment((0,0,0), (0,L,0), color=color, width=width)]
    else:
        return [_axis_segment((0,0,0), (0,0,L), color=color, width=width)]

def _build_axis_dashed(axis: str, L: float, color=(1,1,1,1), width=3,
                       dash_len: float=DASH_LEN, gap_len: float=GAP_LEN):
    items = []
    s = 0.0
    step = max(1e-6, float(dash_len + gap_len))
    while s < L - 1e-9:
        e = min(L, s + dash_len)
        if axis == 'x':
            items.append(_axis_segment((s,0,0), (e,0,0), color=color, width=width))
        elif axis == 'y':
            items.append(_axis_segment((0,s,0), (0,e,0), color=color, width=width))
        else:
            items.append(_axis_segment((0,0,s), (0,0,e), color=color, width=width))
        s += step
    return items

def _screen_perp_tick_on_z(z_val: float, screen_len_px: int = 12):
    """
    Compute two world points that correspond to a short screen-perpendicular tick centered
    on the projected Z-axis location at z_val. Returns ((xL,yL,z),(xR,yR,z)) or None.
    """
    Img = (0.0, 0.0, float(z_val))
    pr = project_point(Img)
    if pr is None:
        return None
    px, py = pr
    dz = 0.01
    pr2 = project_point((0.0, 0.0, float(z_val) + dz))
    if pr2 is None:
        return None
    vx = float(pr2[0] - px)
    vy = float(pr2[1] - py)
    nx, ny = -vy, vx
    nlen = (nx*nx + ny*ny) ** 0.5 or 1.0
    nx /= nlen
    ny /= nlen
    half = screen_len_px * 0.5
    pL = (int(px - nx * half), int(py - ny * half))
    pR = (int(px + nx * half), int(py + ny * half))
    p0L, p1L = screen_to_world_ray(pL[0], pL[1])
    p0R, p1R = screen_to_world_ray(pR[0], pR[1])
    if p0L is None or p1L is None or p0R is None or p1R is None:
        return None
    hitL = intersect_with_plane(p0L, p1L, 'xy')
    hitR = intersect_with_plane(p0R, p1R, 'xy')
    if hitL is None or hitR is None:
        return None
    xL, yL, _ = hitL
    xR, yR, _ = hitR
    return (xL, yL, float(z_val)), (xR, yR, float(z_val))

def _build_axis_ticks(axis: str, L: float, tick_step: float=0,
                      tick_size: float=TICK_SIZE, color=(1,1,1,0.9), width=2):
    items = []
    if tick_step is None:
        tick_step = _auto_tick_step(L)
    t = float(tick_step)
    while t < L + 1e-9:
        if axis == 'x':
            p0 = (t, 0.0, 0.0)
            p1 = (t, 0.0, tick_size)
            items.append(_axis_segment(p0, p1, color=color, width=width))
        elif axis == 'y':
            p0 = (0.0, t, 0.0)
            p1 = (0.0, t, tick_size)
            items.append(_axis_segment(p0, p1, color=color, width=width))
        else:
            leg = tick_size
            items.append(_axis_segment((0.0, 0.0, t), (leg, 0.0, t), color=color, width=width))
            items.append(_axis_segment((0.0, 0.0, t), (0.0, leg, t), color=color, width=width))
        t += tick_step
    return items

def _build_axes_with_ticks(L: float):
    items = []
    step = _auto_tick_step(L)
    items += _build_axis_solid('x', L, color=(1,0,0,1), width=3)
    items += _build_axis_ticks('x', L, tick_step=step, color=(1,0,0,0.9), width=2)
    items += _build_axis_solid('y', L, color=(0,1,0,1), width=3)
    items += _build_axis_ticks('y', L, tick_step=step, color=(0,1,0,0.9), width=2)
    items += _build_axis_solid('z', L, color=(0,0,1,1), width=3)
    items += _build_axis_ticks('z', L, tick_step=step, color=(0,0,1,0.9), width=2)
    return items

axis_items = _build_axes_with_ticks(AXIS_LEN)
for it in axis_items:
    view.addItem(it)

yz_grid = GLGridItem()
yz_grid.setSize(x=AXIS_LEN, y=AXIS_LEN)
yz_grid.rotate(90, 0, 1, 0)
yz_grid.translate(PLANE_OFFSETS['yz'], AXIS_LEN * 0.5, AXIS_LEN * 0.5)

xz_grid = GLGridItem()
xz_grid.setSize(x=AXIS_LEN, y=AXIS_LEN)
xz_grid.rotate(90, 1, 0, 0)
xz_grid.translate(AXIS_LEN * 0.5, PLANE_OFFSETS['xz'], AXIS_LEN * 0.5)

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Overlays: labels, images, hover preview
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

def show_plane_grids():
    try:
        view.addItem(yz_grid)
    except Exception:
        pass
    try:
        view.addItem(xz_grid)
    except Exception:
        pass

def hide_plane_grids():
    try:
        view.removeItem(yz_grid)
    except Exception:
        pass
    try:
        view.removeItem(xz_grid)
    except Exception:
        pass

# Dedicated toggle for plane grids (YZ/XZ) independent of Debug
def _toggle_plane_grids_ui(checked: bool):
    """Toggle only the YZ/XZ plane grids via the UI checkbox, independent of Debug."""
    if checked:
        show_plane_grids()
    else:
        hide_plane_grids()

def _debug_on() -> bool:
    return bool(cb_stimuli.isChecked())

def _ensure_point_label(pid: str) -> QLabel:
    """Ensure a QLabel exists as overlay label for pid and return it."""
    if pid in point_labels:
        return point_labels[pid]
    lab = QLabel(pid, parent=view)
    lab.setProperty('pid', pid)
    lab.installEventFilter(POINT_LABEL_FILTER)
    lab.setStyleSheet("color: #ffffff; background: rgba(0,0,0,140); border: 1px solid #666; border-radius: 4px; padding: 1px 4px; font-size: 12px; font-weight: 600;")
    lab.setTextFormat(Qt.TextFormat.RichText)
    lab.hide()
    point_labels[pid] = lab
    return lab

def _ensure_image_label(pid: str) -> Optional[QLabel]:
    """Return or create an image overlay label for token id (Img#.1 / Img#.2)."""
    cat = _category_of(pid) if '.' in pid else pid
    pm = IMAGES_BY_CAT.get(cat)
    if pm is None or pm.isNull():
        return None
    if pid in image_labels:
        lab = image_labels[pid]
        lab.setPixmap(pm)
        return lab
    lab = QLabel(parent=view)
    lab.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
    lab.setPixmap(pm)
    lab.hide()
    lab.lower()
    image_labels[pid] = lab
    return lab

def _update_image_label(pid: str):
    """Position the per-token image overlay above the placed 3D point (debug-only)."""
    if not _debug_on():
        if pid in image_labels:
            try:
                image_labels[pid].hide()
            except Exception:
                pass
        return
    if pid not in placed_points:
        if pid in image_labels:
            try:
                image_labels[pid].hide()
            except Exception:
                pass
        return
    lab = _ensure_image_label(pid)
    if lab is None:
        return
    _, coords = placed_points[pid]
    if not _is_point_visible_world(coords):
        lab2 = image_labels.get(pid)
        if lab2 is not None:
            try:
                lab2.hide()
            except Exception:
                pass
        return
    pr = project_point(coords)
    if pr is None:
        lab.hide()
        return
    px, py = pr
    lab.adjustSize()
    lab.move(int(px - lab.width() // 2), int(py - lab.height() - IMAGE_OVER_POINT_MARGIN))
    lab.show()


def _ensure_hover_preview() -> QLabel:
    global HOVER_PREVIEW_LABEL
    if HOVER_PREVIEW_LABEL is None:
        lab = QLabel(parent=view)
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

def _show_hover_preview_over_dock(cat: str):
    """Show larger preview above the token dock for a category."""
    pm_orig = IMAGES_ORIG.get(cat) or IMAGES_BY_CAT.get(cat)
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
    # Do NOT hide preview during drag OR when not hovering another token
    return

def _alignment_indicator_for(pid: str) -> str:
    """Return HTML alignment indicator for a point and its partner."""
    if pid not in placed_points or '.' not in pid:
        return pid
    partner = _partner_of(pid)
    if not partner or partner not in placed_points:
        return ALIGN_BAD_HTML.format(partner=partner or "?")
    _, c_self = placed_points[pid]
    _, c_part = placed_points[partner]
    tol = _current_z_tol()
    if abs(float(c_self[2]) - float(c_part[2])) <= tol:
        return ALIGN_OK_HTML.format(partner=partner)
    else:
        return ALIGN_BAD_HTML.format(partner=partner)

# Neue/angepasste Funktion: _update_point_color
def _update_point_color(pid: str):
    """Setzt beide Punkte einer Kategorie auf grün, wenn sie in Z übereinstimmen."""
    if pid not in placed_points or '.' not in pid:
        return
    partner = _partner_of(pid)
    if not partner or partner not in placed_points:
        return

    _, c_self = placed_points[pid]
    _, c_part = placed_points[partner]
    tol = _current_z_tol()

    color = np.array([[1.0, 1.0, 0.0, 1.0]])  # Standard: gelb
    if abs(float(c_self[2]) - float(c_part[2])) <= tol:
        color = np.array([[0.0, 1.0, 0.0, 1.0]])  # grün bei Übereinstimmung

    item_self, _ = placed_points[pid]
    item_self.setData(color=color)
    item_partner, _ = placed_points[partner]
    item_partner.setData(color=color)

def _update_point_label(pid: str):
    """Update overlay label text and position for a placed point id."""
    if pid not in placed_points:
        return
    _ = _ensure_point_label(pid)
    item, coords = placed_points[pid]
    if not _is_point_visible_world(coords):
        lab = point_labels[pid]
        try:
            lab.hide()
        except Exception:
            pass
        return
    pr = project_point(coords)
    lab = point_labels[pid]
    if pr is None:
        lab.hide()
        return
    px, py = pr
    # Cube label = just leading number
    try:
        num = pid.split('.')[0]
    except Exception:
        num = pid
    lab.setText(num)
    lab.adjustSize()
    lab.move(int(px - lab.width() // 2), int(py - lab.height() - LABEL_OVER_POINT_MARGIN))
    lab.show()

def _update_all_point_labels():
    for pid in list(placed_points.keys()):
        _update_point_label(pid)
    for pid in list(placed_points.keys()):
        _update_image_label(pid)

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Point sprite handling and cube/lattice rendering
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

def _ensure_point_item(pid: str):
    """Create or return a GLScatterPlotItem used to render a point sprite."""
    from pyqtgraph.opengl import GLScatterPlotItem
    if pid in placed_points:
        return placed_points[pid][0]
    item = GLScatterPlotItem(pos=np.array([[0.0, 0.0, 0.0]]), size=POINT_SIZE, color=POINT_COLOR, pxMode=True)
    view.addItem(item)
    placed_points[pid] = (item, [0.0, 0.0, 0.0])
    return item

def _set_point_position(pid: str, coords):
    """Set world position of a point sprite and update overlays/lines."""
    item = _ensure_point_item(pid)
    x, y, z = map(float, coords)
    pos = np.array([[x, y, z]], dtype=float)
    item.setData(pos=pos)
    placed_points[pid] = (item, [x, y, z])
    _update_point_label(pid)
    _update_image_label(pid)
    _update_point_color(pid)
    _update_helper_lines(pid)
    if '.' in pid:
        _update_pair_line(_category_of(pid))
    try:
        stimuli_cb.setChecked(True)
    except Exception:
        pass


# --- Helper lines overlay ---
def _update_helper_lines(pid):
    # Clear all helper lines except the hovered PID
    hover = getattr(view, "hover_pid", None)
    for other_pid in list(helper_lines.keys()):
        if other_pid != hover:
            for it in helper_lines[other_pid]:
                try: view.removeItem(it)
                except: pass
            helper_lines.pop(other_pid, None)

    if pid is None:
        return

    if hover != pid:
        return

    if pid not in placed_points:
        return

    # Remove old
    if pid in helper_lines:
        for it in helper_lines[pid]:
            try:
                view.removeItem(it)
            except:
                pass
        helper_lines.pop(pid, None)

    _, (x, y, z) = placed_points[pid]

    # --- Proper dashed line generator ---
    def dashed(p0, p1, dash=0.4, gap=0.35):
        import numpy as _np
        v = _np.array(p1) - _np.array(p0)
        L = float(_np.linalg.norm(v))
        if L <= 1e-9:
            return []

        d = v / L
        out = []
        s = 0.0

        while s < L:
            e = min(L, s + dash)
            a = _np.array(p0) + d * s
            b = _np.array(p0) + d * e
            pts = _np.vstack([a, b])

            item = GLLinePlotItem(
                pos=pts,
                color=(1, 1, 1, 0.4),
                width=1,
                mode='lines'
            )
            out.append(item)
            s += dash + gap

        return out

    segs = []

    # 1) Vertical line to floor (z = 0)
    segs += dashed((x, y, z), (x, y, 0.0))

    # 2) Horizontal line to Y‑axis (x = 0)
    segs += dashed((x, y, z), (0.0, y, z))

    # 3) Horizontal line to X‑axis (y = 0)
    segs += dashed((x, y, z), (x, 0.0, z))

    # New secondary helper lines:
    # From (0, y, z) → (0, 0, z)  (Y-axis hit → X-axis)
    segs += dashed((0.0, y, z), (0.0, 0.0, z))

    # From (0, y, z) → (0, y, 0)  (Y-axis hit → floor)
    segs += dashed((0.0, y, z), (0.0, y, 0.0))

    # From (x, 0, z) → (0, 0, z)  (X-axis hit → Y-axis)
    segs += dashed((x, 0.0, z), (0.0, 0.0, z))

    # From (x, 0, z) → (x, 0, 0)  (X-axis hit → floor)
    segs += dashed((x, 0.0, z), (x, 0.0, 0.0))

    helper_lines[pid] = segs
    for it in segs:
        try:
            view.addItem(it)
        except:
            pass

def _reset_all_points():
    """Remove all point sprites, overlays and reset token states."""
    for pid, (it, _) in list(placed_points.items()):
        try:
            view.removeItem(it)
        except Exception:
            pass
    placed_points.clear()
    for cat, line in list(pair_lines.items()):
        try:
            view.removeItem(line)
        except Exception:
            pass
    pair_lines.clear()
    for pid, lab in list(point_labels.items()):
        try:
            lab.hide()
            lab.deleteLater()
        except Exception:
            pass
    point_labels.clear()
    for pid, lab in list(image_labels.items()):
        try:
            lab.hide()
            lab.deleteLater()
        except Exception:
            pass
    image_labels.clear()
    _hide_hover_preview()
    for t in point_tokens:
        t.setProperty('placed', False)
        t.setStyleSheet(_token_style(False))
        t.show()
    globals()['placement_phase'] = 1
    _update_token_states()
    _update_submit_state()
    log_session_event("Reset all placed points")


def _add_or_update_point(coords):
    """Add a temporary GL point for debugging/visualization."""
    from pyqtgraph.opengl import GLScatterPlotItem
    x, y, z = coords
    if any(c is None for c in (x, y, z)):
        return
    pos = np.array([[float(x), float(y), float(z)]])
    item = GLScatterPlotItem(pos=pos, size=POINT_SIZE, color=POINT_COLOR, pxMode=True)
    view.addItem(item)
    points.append((item, [float(x), float(y), float(z)]))

def _make_edge(p0, p1, color=CUBE_COLOR, width=CUBE_WIDTH):
    pts = np.array([p0, p1], dtype=float)
    return GLLinePlotItem(pos=pts, color=color, width=width, mode='lines')

def build_cube_wireframe(L: float):
    """Return line items that form a wireframe cube of size L."""
    items = []
    vals = [0.0, float(L)]
    for y in vals:
        for z in vals:
            items.append(_make_edge((0, y, z), (L, y, z)))
    for x in vals:
        for z in vals:
            items.append(_make_edge((x, 0, z), (x, L, z)))
    for x in vals:
        for y in vals:
            items.append(_make_edge((x, y, 0), (x, y, L)))
    return items

def show_cube():
    global cube_items
    if not cube_items:
        cube_items = build_cube_wireframe(AXIS_LEN)
    for it in cube_items:
        try:
            view.addItem(it)
        except Exception:
            pass

def hide_cube():
    for it in cube_items:
        try:
            view.removeItem(it)
        except Exception:
            pass

def build_lattice_grid(L: float, step: float):
    """Return GLLinePlotItem list for a 3D lattice grid with given step."""
    step = max(1e-3, float(step))
    ticks = np.arange(0.0, float(L) + 1e-9, step, dtype=float)
    items = []
    for y in ticks:
        for z in ticks:
            items.append(_make_edge((0.0, y, z), (L, y, z), color=LATTICE_COLOR, width=LATTICE_WIDTH))
    for x in ticks:
        for z in ticks:
            items.append(_make_edge((x, 0.0, z), (x, L, z), color=LATTICE_COLOR, width=LATTICE_WIDTH))
    for x in ticks:
        for y in ticks:
            items.append(_make_edge((x, y, 0.0), (x, y, L), color=LATTICE_COLOR, width=LATTICE_WIDTH))
    return items

def show_lattice(step: float):
    global lattice_items
    hide_lattice()
    lattice_items = build_lattice_grid(AXIS_LEN, step)
    for it in lattice_items:
        try:
            view.addItem(it)
        except Exception:
            pass

def hide_lattice():
    global lattice_items
    for it in lattice_items:
        try:
            view.removeItem(it)
        except Exception:
            pass
    lattice_items = []

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

def _cube_center():
    L = float(AXIS_LEN)
    return QVector3D(L/2.0, L/2.0, L/2.0)

def _fit_distance_for_extent(extent: float, margin: float = 2) -> float:
    """Compute camera distance so a given extent is visible respecting FOV."""
    w = max(1, view.width())
    h = max(1, view.height())
    vfov_deg = float(view.opts.get('fov', 60))
    vfov = np.deg2rad(vfov_deg)
    aspect = w / h
    hfov = 2.0 * np.arctan(np.tan(vfov/2.0) * aspect)
    half = extent / 2.0
    d_v = half / np.tan(vfov/2.0)
    d_h = half / np.tan(hfov/2.0)
    return float(max(d_v, d_h) * margin)

view.opts['center'] = _cube_center()
view.setCameraPosition(distance=_fit_distance_for_extent(13), elevation=35.3, azimuth=45)

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Header / axis label overlays and tick labels
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

header_label = QLabel("", parent=view)
header_label.setStyleSheet("color: #ffffff; font-size: 18px; font-weight: 600; background: transparent;")
header_label.raise_()

def _position_header():
    header_label.adjustSize()
    x = (view.width() - header_label.width()) // 2
    y = 12
    header_label.move(x, y)

axis_label_x = QLabel("", parent=view)
axis_label_y = QLabel("", parent=view)
axis_label_z = QLabel("", parent=view)
for lab, col in [(axis_label_x, "#d33"), (axis_label_y, "#0a0"), (axis_label_z, "#33d")]:
    lab.setStyleSheet(f"color: {col}; font-size: 16px; font-weight: 500; background: transparent;")
    lab.raise_()
for lab in (axis_label_x, axis_label_y, axis_label_z):
    lab.show()

axis_tick_labels = {'x': [], 'y': [], 'z': []}
TICK_WORLD_POS = [0.0, AXIS_LEN * 0.5, AXIS_LEN]

def _ensure_axis_tick_labels():
    """Ensure 3 QLabel tick overlays per axis exist (for debug mode)."""
    for k in ('x','y','z'):
        if len(axis_tick_labels[k]) == 3:
            continue
        for lab in axis_tick_labels[k]:
            try:
                lab.hide(); lab.deleteLater()
            except Exception:
                pass
        axis_tick_labels[k] = []
        for _ in range(3):
            lab = QLabel(parent=view)
            lab.setStyleSheet("color: #cccccc; font-size: 12px; background: rgba(0,0,0,120); border: 1px solid #555; border-radius: 3px; padding: 0px 3px;")
            lab.hide()
            axis_tick_labels[k].append(lab)

def _update_axis_tick_labels():
    """Position small axis value labels when debug mode is enabled."""
    for k in ('x','y','z'):
        for lab in axis_tick_labels[k]:
            lab.hide()
    return

def _show_axis_tick_labels(show: bool):
    _ensure_axis_tick_labels()
    for k in ('x','y','z'):
        for lab in axis_tick_labels[k]:
            (lab.show() if show else lab.hide())

def choose_plane_and_hit(px: int, py: int):
    """Choose the best plane hit (yz/xz/xy) for the given screen pixel and return (plane, hit)."""
    p0, p1 = screen_to_world_ray(px, py)
    if p0 is None or p1 is None:
        return None
    ortho = bool(view.opts.get('ortho', False))
    if ortho:
        if current_plane in ('yz', 'xz'):
            candidates = []
            for pl in ('yz', 'xz'):
                r = intersect_with_plane_t(p0, p1, pl)
                if r is not None:
                    t, hit = r
                    if pl == current_plane:
                        t = t - 1e-9
                    candidates.append((t, pl, hit))
            if not candidates:
                return None
            candidates.sort(key=lambda x: x[0])
            _, pl, hit = candidates[0]
            return pl, hit
        else:
            res = intersect_with_plane_t(p0, p1, current_plane)
            if res is None:
                return None
            _, hit = res
            return current_plane, hit
    candidates = []
    for pl in ('yz', 'xz'):
        r = intersect_with_plane_t(p0, p1, pl)
        if r is not None:
            t, hit = r
            candidates.append((t, pl, hit))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    _, pl, hit = candidates[0]
    return pl, hit

def clamp_to_cube(x, y, z):
    """Clamp a world coordinate to the cube [0, AXIS_LEN]."""
    L = AXIS_LEN
    return max(0, min(L, x)), max(0, min(L, y)), max(0, min(L, z))

def position_axis_labels():
    """Position axis labels at the visible ends of the three world axes."""
    axis_pts = {
        'x': (AXIS_LEN, 0.0, 0.0),
        'y': (0.0, AXIS_LEN, 0.0),
        'z': (0.0, 0.0, AXIS_LEN),
    }

    label_map = {
        'x': axis_label_x,
        'y': axis_label_y,
        'z': axis_label_z,
    }

    for axis, world_pt in axis_pts.items():
        pr = project_point(world_pt)
        if pr is None:
            continue

        px, py = pr
        lab = label_map[axis]
        lab.adjustSize()

        # For X and Y we place labels slightly away from axis end
        if axis == 'x':
            lab.move(px - lab.width() // 2 - 20, py - lab.height() + 7)
        elif axis == 'y':
            lab.move(px - lab.width() // 2 + 20, py - lab.height() + 7)
        else:  # Z axis
            # Offset to the side to avoid overlapping the axis line
            lab.move(px - 6, py - 25)

        lab.raise_()
        
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Panel / controls / params
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

panel = QFrame(parent=left_col)
panel.setFrameShape(QFrame.Shape.StyledPanel)
panel.setStyleSheet(
    """
    QFrame { background: none; border-radius: 8px; }
    QLabel { color: white;}
    QLineEdit { color: #000000; background: lightgray; border: 1px solid black; border-radius: 4px; padding: 1px 6px; font-size: 12px; }
    QPushButton { color: #000000; background: lightgray; border: 1px solid black; border-radius: 4px; padding: 4px 8px; }
    QPushButton:pressed { background: #e5e5e5; }
    """
)
panel_layout = QVBoxLayout(panel)
panel_layout.setContentsMargins(0, 8, 0, 10)
panel_layout.setSpacing(ROW_SPACING)

def make_row(caption: str, default_text: str):
    """Create a labeled row with a QLineEdit for the main axis inputs."""
    row = QWidget(panel)
    h = QHBoxLayout(row)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(ROW_SPACING)
    lab = QLabel(caption, row)
    lab.setFixedWidth(LABEL_WIDTH)
    lab.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    edit = QLineEdit(row)
    edit.setText(default_text)
    edit.setFixedHeight(CONTROL_H)
    h.addWidget(lab)
    h.addWidget(edit)
    h.addStretch(1)
    return row, edit, lab, h

row_x, edit_x, lab_x, row_x_layout = make_row("X:", "X")
edit_x.setReadOnly(True)
edit_x.setCursor(QCursor(Qt.CursorShape.ForbiddenCursor))
row_y, edit_y, lab_y, row_y_layout = make_row("Y:", "Y")
edit_y.setReadOnly(True)
edit_y.setCursor(QCursor(Qt.CursorShape.ForbiddenCursor))
row_z, edit_z, lab_z, row_z_layout = make_row("Z:", "Z")
edit_z.setReadOnly(True)
edit_z.setCursor(QCursor(Qt.CursorShape.ForbiddenCursor))
row_x_layout.setSpacing(ROW_SPACING)
row_y_layout.setSpacing(ROW_SPACING)
row_z_layout.setSpacing(ROW_SPACING)

# Disable axis input fields and labels

row_x.hide()
row_y.hide()
row_z.hide()


for lab in (lab_x, lab_y, lab_z):
    lab.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

edit_x.setFixedWidth(INPUT_WIDTH)
edit_y.setFixedWidth(INPUT_WIDTH)
edit_z.setFixedWidth(INPUT_WIDTH)
edit_x.setFixedHeight(CONTROL_H)
edit_y.setFixedHeight(CONTROL_H)
edit_z.setFixedHeight(CONTROL_H)


# Move Lock View checkbox to the scene view (bottom-right)
cb_lock = QCheckBox("Lock View (^L)", parent=view)
cb_lock.setStyleSheet("QCheckBox { color: #ffffff; background: rgba(0,0,0,120);}")
cb_lock.setCursor(Qt.CursorShape.PointingHandCursor)
cb_lock.setChecked(True)
cb_lock.show()
cb_lock.raise_()

# Elegante direkte Lambda-Lösung:
orig_resize = view.resizeEvent
view.resizeEvent = lambda ev: (
    orig_resize(ev),
    cb_lock.move(view.width() - cb_lock.width() - 20,
                 view.height() - cb_lock.height() - 20),
    cb_lock.raise_()
)

def _toggle_lock(checked: bool):
    globals()['LOCK_CAMERA'] = bool(checked)

cb_lock.toggled.connect(_toggle_lock)

cb_stimuli = QCheckBox("Show Stimuli (^B)", parent=view)
cb_stimuli.setStyleSheet("QCheckBox { color: #ffffff; background: rgba(0,0,0,120);}")
cb_stimuli.setCursor(Qt.CursorShape.PointingHandCursor)
cb_stimuli.setChecked(True)
cb_stimuli.show()
cb_stimuli.raise_()

orig_resize2 = view.resizeEvent
view.resizeEvent = lambda ev: (
    orig_resize2(ev),
    cb_stimuli.move(20,
                 view.height() - cb_stimuli.height() - 20),
    cb_stimuli.raise_()
)

def _toggle_grid(checked: bool):
    if checked:
        show_cube()
        show_lattice(10)
        _update_all_point_labels()
    else:
        hide_cube()
        hide_lattice()
        for pid, lab in list(image_labels.items()):
            try:
                lab.hide()
            except Exception:
                pass

# Provide fixed tolerance function (no UI)
def _current_z_tol():
    return Z_ALIGN_EPS

hint_row = QWidget(panel)
hint_box = QHBoxLayout(hint_row)
hint_box.setContentsMargins(0, 0, 0, 0)
hint_box.setSpacing(ROW_SPACING)
spacer = QWidget(hint_row)
spacer.setFixedWidth(LABEL_WIDTH)
hint_box.addWidget(spacer)
hint = QLabel("Press ⏎ to apply", hint_row)
hint_box.addWidget(hint)
hint_box.addStretch(1)
panel_layout.addWidget(hint_row)

hint_row.hide()

def _set_view_fitted(elevation=0, azimuth=0, zoom=1.0):
    """Set camera to view cube center with optional elevation/azimuth and zoom multiplier."""
    extent = float(AXIS_LEN)
    dist = _fit_distance_for_extent(extent)
    dist *= float(zoom)
    view.opts['center'] = _cube_center()
    view.setCameraPosition(distance=dist, elevation=elevation, azimuth=azimuth)

def set_view_xy():
    """Switch to orthographic view focused on YZ plane (Z-Y)."""
    global current_plane
    current_plane = 'yz'
    view.opts['ortho'] = True
    log_session_event("Rotated View XY")
    _set_view_fitted(elevation=0, azimuth=90, zoom=0.03)

def set_view_yz():
    """Switch to orthographic view focused on XZ plane (Z-X)."""
    global current_plane
    current_plane = 'xz'
    view.opts['ortho'] = True
    log_session_event("Rotated View XZ")
    _set_view_fitted(elevation=0, azimuth=0, zoom=0.03)

# Camera helpers
def set_view_default():
    """Reset camera to the program's initial perspective view."""
    try:
        view.opts['ortho'] = False  # ensure perspective
    except Exception:
        pass
    # Match the startup camera center and angles
    view.opts['center'] = _cube_center()
    view.setCameraPosition(distance=_fit_distance_for_extent(13), elevation=35.3, azimuth=45)
    try:
        xyz_cb.setChecked(True)
    except Exception:
        pass

# btn_xy.clicked.connect(set_view_xy)
# btn_yz.clicked.connect(set_view_yz)

panel.adjustSize()
panel.setMinimumSize(150, 250)
panel.show()

# Add panel to left layout if not already present
left_v.addWidget(panel)

# ---------------------------------------------------------------------
# Tokens / token dock
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

tokens_label = QLabel("Stimuli:")
tokens_label.setStyleSheet("color: #fff; font-size: 14px; font-weight: 600; background: transparent;")
tokens_label.adjustSize()
tokens_label.show()

point_dock = QFrame()
point_dock.setFrameShape(QFrame.Shape.StyledPanel)
point_dock.setStyleSheet("QFrame { background: rgba(0,0,0,120); border: 1px solid #777; border-radius: 6px; }")

point_dock_layout = QGridLayout(point_dock)
point_dock_layout.setContentsMargins(8, 8, 8, 8)
point_dock_layout.setHorizontalSpacing(6)
point_dock_layout.setVerticalSpacing(6)

point_dock.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
point_dock_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

point_tokens = []

def _token_style(placed: bool) -> str:
    """Return CSS style for token labels depending on placed state."""
    if placed:
        return "QLabel { color: #111; background: #7CFC00; border: 1px solid #3a3; border-radius: 4px; padding: 2px 6px; font-weight: 600; }"
    return "QLabel { color: #eee; background: #444; border: 1px solid #999; border-radius: 4px; padding: 2px 6px; } QLabel:hover { background: #555; }"

def _token_style_mode(mode: str) -> str:
    """Return CSS for named token modes: 'placed', 'disabled', default active."""
    if mode == 'placed':
        return "QLabel { color: #111; background: #7CFC00; border: 1px solid #3a3; border-radius: 4px; padding: 2px 6px; font-weight: 600; }"
    if mode == 'disabled':
        return "QLabel { color: #aaa; background: #333; border: 1px solid #666; border-radius: 4px; padding: 2px 6px; }"
    return "QLabel { color: #eee; background: #444; border: 1px solid #999; border-radius: 4px; padding: 2px 6px; } QLabel:hover { background: #555; }"


def _update_token_states():
    """Update enabled/disabled state and styles of tokens according to placed flag."""
    for t in point_tokens:
        placed = bool(t.property('placed'))
        if placed:
            t.setStyleSheet(_token_style_mode('placed'))
            t.setEnabled(False)
        else:
            t.setEnabled(True)
            t.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
            t.setStyleSheet(_token_style_mode('active'))
    _hide_hover_preview()

def _category_of(pid: str) -> str:
    return pid.split('.')[0]

def _partner_of(pid: str) -> Optional[str]:
    if '.' not in pid:
        return None
    cat = _category_of(pid)
    return f"{cat}.2" if pid.endswith('.1') else f"{cat}.1"

def _remove_placed_point(pid: str):
    """Remove a placed point's sprite and overlays."""
    if pid in placed_points:
        try:
            it, _ = placed_points[pid]
            view.removeItem(it)
        except Exception:
            pass
        placed_points.pop(pid, None)
    if pid in point_labels:
        try:
            point_labels[pid].hide()
            point_labels[pid].deleteLater()
        except Exception:
            pass
        point_labels.pop(pid, None)
    if pid in image_labels:
        try:
            image_labels[pid].hide()
            image_labels[pid].deleteLater()
        except Exception:
            pass
        image_labels.pop(pid, None)
    _update_pair_line(_category_of(pid))
    _update_submit_state()

def _update_pair_line(cat: str):
    """Draw or update debugging line between .1 and .2 points of a category."""
    if not _debug_on():
        if cat in pair_lines:
            line = pair_lines[cat]
            try:
                view.removeItem(line)
            except Exception:
                pass
            try:
                line.setData(pos=np.empty((0, 3), dtype=float))
            except Exception:
                pass
            del pair_lines[cat]
        return
    pid1 = f"{cat}.1"
    pid2 = f"{cat}.2"
    has1 = pid1 in placed_points
    has2 = pid2 in placed_points
    if has1 and has2:
        _, c1 = placed_points[pid1]
        _, c2 = placed_points[pid2]
        pts = np.array([c1, c2], dtype=float)
        if cat in pair_lines:
            line = pair_lines[cat]
            line.setData(pos=pts)
        else:
            line = GLLinePlotItem(pos=pts, color=(1, 1, 1, 1), width=1, mode='lines')
            pair_lines[cat] = line
        try:
            view.addItem(line)
        except Exception:
            pass
    else:
        if cat in pair_lines:
            line = pair_lines[cat]
            try:
                view.removeItem(line)
            except Exception:
                pass
            try:
                line.setData(pos=np.empty((0, 3), dtype=float))
            except Exception:
                pass
            del pair_lines[cat]

for i in range(1, 13):
    t = DraggableToken(f"{i}. Stimulus", parent=point_dock)
    t.setMinimumWidth(110)
    t.setFixedHeight(27)
    point_tokens.append(t)
    row = i - 1
    point_dock_layout.addWidget(t, row, 0)
    _update_token_states()

# def _all_pairs_combined() -> bool:
#     """Return True if all categories have a combined point (ImgN) placed and no .1/.2 remain."""
#     cats = _token_categories()
#     if not cats:
#         return False
#     for c in cats:
#         if c not in placed_points:
#             return False
#         if f"{c}.1" in placed_points or f"{c}.2" in placed_points:
#             return False
#     return True

def _update_submit_state():
    try:
        total = len(point_tokens)
        placed = len(placed_points)
        started = bool(start_cb.isChecked())
        btn_submit.setEnabled(placed == total and started)
    except Exception:
        pass

def _collect_combined_points_norm():
    """Collect combined points and normalize world coords from [0,AXIS_LEN] to [-1,1]."""
    data = []
    L = float(AXIS_LEN)
    half = L * 0.5
    for name, (_item, coords) in placed_points.items():
        if '.' in name:
            continue
        x, y, z = map(float, coords)
        xn = (x - half) / half
        yn = (y - half) / half
        zn = (z - half) / half
        data.append((name, xn, yn, zn))
    try:
        data.sort(key=lambda t: (int(''.join(ch for ch in t[0] if ch.isdigit())) if any(ch.isdigit() for ch in t[0]) else t[0]))
    except Exception:
        data.sort(key=lambda t: t[0])
    return data

def _axis_display_name(edit_widget: QLineEdit, combo_widget, fallback: str) -> str:
    """Return axis label display text from edit_widget, fallback if empty."""
    try:
        label = (edit_widget.text() or "").strip() or fallback
        return f"{label}"
    except Exception:
        return f"{fallback}"

def _export_results():
    """Export combined points to CSV if all pairs are combined."""
    log_session_event("submitted, finished experiment")
    # if not _all_pairs_combined():
    #     _update_submit_state()
    #     return
    btn_start.setEnabled(True)
    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base, "results")
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    x_name = _axis_display_name(edit_x, None, "X")
    y_name = _axis_display_name(edit_y, None, "Y")
    z_name = _axis_display_name(edit_z, None, "Z")
    rows = _collect_combined_points_norm()
    csv_path = os.path.join(out_dir, f"embedding_{ts}.csv")
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Name", x_name, y_name, z_name])
            # Zeitmessung hinzufügen
            elapsed_seconds = None
            global start_time
            if start_time is not None:
                elapsed_seconds = (datetime.now() - start_time).total_seconds()
            for name, xn, yn, zn in rows:
                w.writerow([name, f"{xn:.6f}", f"{yn:.6f}", f"{zn:.6f}"])
            if elapsed_seconds is not None:
                w.writerow([])
                w.writerow(["Elapsed Time (s)", f"{elapsed_seconds:.2f}"])
    except Exception:
        csv_path = os.path.join(base, f"embedding_{ts}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Name", x_name, y_name, z_name])
            elapsed_seconds = None
            if start_time is not None:
                elapsed_seconds = (datetime.now() - start_time).total_seconds()
            for name, xn, yn, zn in rows:
                w.writerow([name, f"{xn:.6f}", f"{yn:.6f}", f"{zn:.6f}"])
            if elapsed_seconds is not None:
                w.writerow([])
                w.writerow(["Elapsed Time (s)", f"{elapsed_seconds:.2f}"])
    try:
        header_label.setText(f"Saved CSV: {os.path.basename(csv_path)}")
        _position_header()
    except Exception:
        pass


point_dock.adjustSize()
point_dock.show()

btn_reset = QPushButton("Reset  (^R)", left_col)
btn_submit = QPushButton("Submit  (^↩)", left_col)
btn_start = QPushButton("Start", left_col)
btn_start.setDisabled(True)
def _update_start_button_state():
    """Update Start button enabled state based on all conditions."""
    btn_start.setEnabled(
        name_input.text().strip() != "" and 
        check_hover_cb.isChecked() and
        xyz_cb.isChecked() and
        stimuli_cb.isChecked()
    )

def _update_progress_counter():
    completed = 0
    if name_cb.isChecked(): completed += 1
    if check_hover_cb.isChecked(): completed += 1
    if xyz_cb.isChecked(): completed += 1
    if stimuli_cb.isChecked(): completed += 1
    if start_cb.isChecked(): completed += 1
    counter_label.setText(f"({completed}/5)")
    counter_label.adjustSize()

_update_start_button_state()
_update_progress_counter()
name_input.textChanged.connect(_update_start_button_state)
check_hover_cb.toggled.connect(_update_start_button_state)
xyz_cb.toggled.connect(_update_start_button_state)
stimuli_cb.toggled.connect(_update_start_button_state)

name_cb.toggled.connect(_update_progress_counter)
check_hover_cb.toggled.connect(_update_progress_counter)
xyz_cb.toggled.connect(_update_progress_counter)
stimuli_cb.toggled.connect(_update_progress_counter)
start_cb.toggled.connect(_update_progress_counter)
btn_grid = QPushButton("Set View (^D)", left_col)
btn_reset.setFixedSize(110, 25)
btn_submit.setFixedSize(110, 25)
btn_start.setFixedSize(110, 25)
btn_grid.setFixedSize(110, 25)

btn_reset.clicked.connect(_reset_all_points)
btn_start.clicked.connect(_start_experiment)
btn_start.clicked.connect(_reset_all_points)
btn_grid.toggled.connect(set_view_default)
btn_grid.clicked.connect(lambda *_: xyz_cb.setChecked(True))

btn_grid.setCheckable(True)
btn_submit.setEnabled(False)
btn_submit.clicked.connect(lambda: _export_results())

btn_reset.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
btn_submit.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
btn_start.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
btn_grid.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

btn_reset.setStyleSheet("""
QPushButton {
    color: #000000;
    background: #f5f5f5;
    border: 1px solid black;
    border-radius: 6px;
    padding: 4px 8px;
}

/* gedrückt */
QPushButton:pressed {
    background: grey;
}

/* deaktiviert */
QPushButton:disabled {
    background: #e0e0e0;
    color: #888888;
    border: 1px solid black;
}
""")
btn_submit.setStyleSheet(btn_reset.styleSheet())
btn_start.setStyleSheet(btn_reset.styleSheet() + """QPushButton { background: #00cc66; border: solid lightgray;}""")
btn_grid.setStyleSheet(btn_reset.styleSheet())

try:
    sc_border = QShortcut(QKeySequence("Meta+B"), win)
    sc_border.activated.connect(lambda: cb_stimuli.setChecked(not cb_stimuli.isChecked()))
    sc_reset = QShortcut(QKeySequence("Meta+R"), win)
    sc_reset.activated.connect(_reset_all_points)
    sc_debug = QShortcut(QKeySequence("Meta+D"), win)
    sc_submit = QShortcut(QKeySequence("Meta+Return"), win)
    sc_submit.activated.connect(lambda: _export_results())
    sc_debug.activated.connect(lambda: btn_grid.setChecked(not btn_grid.isChecked()))
    sc_lock = QShortcut(QKeySequence("Meta+L"), win)
    sc_lock.activated.connect(lambda: cb_lock.setChecked(not cb_lock.isChecked()))
except Exception:
    pass
_update_submit_state()

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Layout: image strip, preview + actions
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

image_row = QWidget(left_col)
image_h = QHBoxLayout(image_row)
image_h.setContentsMargins(0, 0, 0, 0)
image_h.setSpacing(GAP_H)

token_col = QWidget(image_row)
token_v = QVBoxLayout(token_col)
token_v.setContentsMargins(0, 0, 0, 0) 
token_v.setSpacing(6)
token_v.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

token_scroll = QScrollArea(parent=token_col)
token_scroll.setWidget(point_dock)
token_scroll.setWidgetResizable(False)
token_scroll.setFrameShape(QFrame.Shape.NoFrame)
token_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
token_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
# token_scroll.setFixedSize(TOKEN_CONTAINER_W, TOKEN_CONTAINER_H)
token_scroll.setFixedWidth(TOKEN_CONTAINER_W)
token_scroll.setFixedHeight(342)
# Ensure the dock is at least as wide as the viewport so no horizontal scrollbar appears spuriously
token_scroll.setStyleSheet("QScrollArea { background: transparent; } QScrollBar:vertical { background: #222; width: 8px; margin: 0px 0px 0px 0px; } QScrollBar::handle:vertical { background: #555; min-height: 10px; border-radius: 4px; } QScrollBar::handle:vertical:hover { background: #777; } QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; } QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }")
token_scroll.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

token_v.addWidget(tokens_label, 0, Qt.AlignmentFlag.AlignLeft)
token_v.addWidget(token_scroll, 0, Qt.AlignmentFlag.AlignLeft)

preview_col = QWidget(image_row)
preview_v = QVBoxLayout(preview_col)
preview_v.setContentsMargins(0, PREVIEW_TOP_OFFSET, 0, 0)
preview_v.setSpacing(6)

preview_label.setParent(preview_col)
preview_label.move(0, 0)

preview_box.setParent(preview_col)
preview_v.addWidget(preview_box, 0, Qt.AlignmentFlag.AlignTop)

preview_v.addSpacing(ACTIONS_TOP_OFFSET)

actions_row.setParent(preview_col)
preview_v.addWidget(actions_row, 0)

# btn_combine.setParent(preview_col)
# btn_combine.move(0, 240)
btn_reset.setParent(preview_col)
btn_reset.move(120, 240)
btn_submit.setParent(preview_col)
btn_submit.move(120, 275)
btn_start.setParent(preview_col)
btn_start.move(0, 240)
btn_grid.setParent(preview_col)
btn_grid.move(0, 275)

console_box = QPlainTextEdit(preview_col)
console_box.setReadOnly(True)
console_box.setStyleSheet("""
    QPlainTextEdit {
        background: rgba(255,255,255,0.1);
        color: #ddd;
        border-radius: 6px;
        font-size: 12px;
    }
""")
console_box.setFixedHeight(55)
console_box.setFixedWidth(preview_box.width())
console_box.setParent(preview_col)
console_box.move(0, 310) 

image_h.addWidget(token_col, 0)
image_h.addWidget(preview_col, 0)
left_v.addWidget(image_row, 0)

def _mark_token_placed(pid: str):
    """Update a token as placed (hide/green) and update UI state."""
    for t in point_tokens:
        if t.pid == pid:
            t.setProperty('placed', True)
            t.setStyleSheet(_token_style_mode('placed'))
            t.hide()
            _ensure_image_label(pid)
            break
    _update_token_states()
    _update_submit_state()

def _mark_token_unplaced(pid: str):
    """Mark a token as unplaced and update UI state."""
    for t in point_tokens:
        if t.pid == pid:
            t.setProperty('placed', False)
            t.show()
            break
    _update_token_states()

    # Clear helper lines when no selection
    if pid is None:
        for pid2, segs in list(helper_lines.items()):
            for it in segs:
                try:
                    view.removeItem(it)
                except Exception:
                    pass
        helper_lines.clear()
        return

    # Highlight selected point
    if pid in placed_points:
        it, _ = placed_points[pid]
        it.setData(size=POINT_SIZE + 10)
        it.setData(color=np.array([[1.0, 1.0, 1.0, 1.0]]))

btn_grid.setChecked(False)
cb_lock.setChecked(False)

def apply_labels():
    """Apply axis labels from edit fields to the overlay labels and header."""
    tx, ty, tz = edit_x.text().strip(), edit_y.text().strip(), edit_z.text().strip()
    axis_label_x.setText(tx)
    axis_label_y.setText(tz)   # swapped
    axis_label_z.setText(ty)   # swapped
    for lab in (axis_label_x, axis_label_y, axis_label_z):
        lab.show()
        lab.raise_()
    header_label.setText(f"X: {tx}    Y: {tz}    Z: {ty}")
    _position_header()
    position_axis_labels()

for e in (edit_x, edit_y, edit_z):
    e.returnPressed.connect(apply_labels)

_load_images_for_categories()
apply_labels()
position_axis_labels()
_position_header()
_update_token_states()
_ensure_axis_tick_labels()
_update_submit_state()
# _position_debug_button()

view.sigMouseMoved = getattr(view, 'sigMouseMoved', None)

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Runtime: timers, resize handlers, shortcuts
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

_timer = QTimer()
_timer.setInterval(1000 // 60)
_timer.timeout.connect(position_axis_labels)
_timer.timeout.connect(_update_all_point_labels)
_timer.timeout.connect(_update_axis_tick_labels)
_timer.start()

_old_win_resize = win.resizeEvent
def _win_resize(ev):
    if _old_win_resize:
        _old_win_resize(ev)
    _position_header()
    _update_token_states()
    position_axis_labels()
    _update_all_point_labels()
    cb_lock.move(view.width() - cb_lock.width() - 20, view.height() - cb_lock.height() - 20)

win.resizeEvent = _win_resize

try:
    _old__show_hover_preview_over_dock = _show_hover_preview_over_dock
    def _show_hover_preview_over_dock(cat: str):
        _old__show_hover_preview_over_dock(cat)
        try:
            _set_preview_for_category(cat)
        except Exception:
            pass
except Exception:
    pass

sys.exit(app.exec())