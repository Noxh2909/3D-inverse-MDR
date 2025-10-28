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
    QLabel, QComboBox, QCheckBox, QMenu, QMainWindow, QSizePolicy, QGridLayout, QScrollArea, QPlainTextEdit
)
from PySide6.QtGui import QVector3D, QDrag, QCursor, QPixmap, QFont, QKeySequence, QShortcut
from PySide6.QtCore import Qt, QTimer, QMimeData, QPoint, QObject, QEvent
from pyqtgraph.opengl import GLViewWidget, GLLinePlotItem, GLGridItem

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Constants / Global configuration
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
IMAGE_MAX_WH = 80
IMAGE_CONTAINER_WH = 130
LABEL_SCREEN_MARGIN = 24
VIS_DOT_THRESHOLD = 0.0

ALIGN_OK_HTML = "<span style='color:#7CFC00'>aligned with {partner}</span>"
ALIGN_BAD_HTML = "<span style='color:#ff6666'>not aligned with {partner}</span>"

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Runtime state (globals)
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

working: Dict[str, Optional[float]] = {'x': None, 'y': None, 'z': None}
placed_points: Dict[str, tuple] = {}
point_labels: Dict[str, QLabel] = {}
pair_lines: Dict[str, GLLinePlotItem] = {}
points = []
cube_items = []
lattice_items = []
current_plane = 'xy'
placement_phase = 1
start_time = None
LOCK_CAMERA = False

IMAGES_BY_CAT: Dict[str, QPixmap] = {}
IMAGES_ORIG: Dict[str, QPixmap] = {}
image_labels: Dict[str, QLabel] = {}
HOVER_PREVIEW_LABEL: Optional[QLabel] = None

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Debug / logging helpers
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
        cats = [f"{i}" for i in range(1, 11)]
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
    """Event filter for overlay point labels (right-click context menu)."""

    def eventFilter(self, obj, ev):
        try:
            if ev.type() == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.RightButton:
                pid = obj.property('pid')
                if pid and isinstance(pid, str) and '.' not in pid:
                    _show_revert_menu(obj, pid)
                    return True
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
        """
        On drop: choose plane & hit point, clamp to cube and set point position.
        Also updates placement phase and token state.
        """
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
        chosen = choose_plane_and_hit(px, py)
        if chosen is None:
            ev.ignore()
            return
        plane_used, hit = chosen
        x, y, z = clamp_to_cube(*hit)
        if plane_used == 'xy':
            z = PLANE_OFFSETS['xy']
        elif plane_used == 'xz':
            y = PLANE_OFFSETS['xz']
        else:
            x = PLANE_OFFSETS['yz']
        _set_point_position(point_id, (x, y, z))
        globals()['current_plane'] = plane_used
        _mark_token_placed(point_id)
        ev.acceptProposedAction()

    def mousePressEvent(self, ev):
        """
        Support picking of projected 2D tokens to initiate drag of an already-placed token.
        Otherwise forward to base implementation (camera interaction).
        """
        if ev.button() == Qt.MouseButton.LeftButton:
            pos = ev.position().toPoint()
            mx, my = pos.x(), pos.y()
            hit_pid = None
            best_d2 = 1e9
            for pid, (_item, coords) in placed_points.items():
                if '.' not in pid:
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
            pick_r = float(max(10.0, POINT_SIZE))
            if hit_pid is not None and best_d2 <= (pick_r * pick_r):
                drag = QDrag(self)
                mime = QMimeData()
                mime.setData('application/x-point-id', hit_pid.encode('utf-8'))
                drag.setMimeData(mime)
                drag.exec(Qt.DropAction.MoveAction)
                return
        if not LOCK_CAMERA:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        """Hover picking: show hover preview for nearest visible point if within radius."""
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
            if hit_pid is not None and best_d2 <= (pick_r * pick_r):
                cat = hit_pid.split('.')[0]
                _set_preview_for_category(cat)  
                if best_proj is not None:
                    # _show_hover_preview(cat, QPoint(int(best_proj[0]), int(best_proj[1])))
                    pass
            else:
                _hide_hover_preview()
        except Exception:
            pass

        if LOCK_CAMERA:
            return
        super().mouseMoveEvent(ev)

    def wheelEvent(self, ev):
        """Allow wheel zoom/behavior unless camera is locked."""
        if LOCK_CAMERA:
            return
        super().wheelEvent(ev)

    def leaveEvent(self, ev):
        try:
            _hide_hover_preview()
        except Exception:
            pass
        super().leaveEvent(ev)


class DraggableToken(QLabel):
    """Small QLabel that acts as draggable token in the left dock."""

    def __init__(self, pid: str, parent=None):
        super().__init__(pid, parent)
        self.pid = pid
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        self.setStyleSheet(_token_style_mode('disabled'))
        # self.setMinimumWidth(80)
        self.setFixedSize(55, 18)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)

    def enterEvent(self, ev):
        try:
            cat = self.pid.split('.')[0]
            _show_hover_preview_over_dock(cat)
        except Exception:
            pass
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        _hide_hover_preview()
        super().leaveEvent(ev)

    def mouseMoveEvent(self, ev):
        try:
            cat = self.pid.split('.')[0]
            _show_hover_preview_over_dock(cat)
        except Exception:
            pass
        super().mouseMoveEvent(ev)

    def mousePressEvent(self, ev):
        """Initiate a drag with the token id in mime data if allowed in current phase."""
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        if not _is_token_draggable(self.pid) or not self.isEnabled():
            return
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData('application/x-point-id', self.pid.encode('utf-8'))
        drag.setMimeData(mime)
        cat = self.pid.split('.')[0]
        try:
            pm = IMAGES_BY_CAT.get(cat)
            if pm is not None and not pm.isNull():
                drag.setPixmap(pm)
                hs = QPoint(pm.width() // 2, int(pm.height() * 0.8))
                drag.setHotSpot(hs)
        except Exception:
            pass
        _set_preview_for_category(cat)
        _hide_hover_preview()
        drag.exec(Qt.DropAction.MoveAction)

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
win.setWindowTitle("3D inverse-MDS for embedding data")

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
try:
    view.setCameraParams(fov=110)
except Exception:
    view.opts['fov'] = 75
    view.update()

view_wrap = QWidget()
vw_lay = QVBoxLayout(view_wrap)
vw_lay.setContentsMargins(0, SCENE_TOP_OFFSET, 10, 0)
vw_lay.setSpacing(0)
vw_lay.addWidget(view)
view_wrap.setFixedHeight(SCENE_FIXED_HEIGHT)
main_row.addWidget(view_wrap, 1, Qt.AlignmentFlag.AlignTop)

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
    log_session_event("Experiment started...")
    cb_lock.setChecked(True)
    set_view_xy()
    _position_header()

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
    return bool(btn_grid.isChecked())

def show_pair_lines():
    cats = set()
    for pid in placed_points.keys():
        if '.' in pid:
            cats.add(pid.split('.')[0])
    for cat in cats:
        _update_pair_line(cat)

def hide_pair_lines():
    for cat, line in list(pair_lines.items()):
        try:
            view.removeItem(line)
        except Exception:
            pass
        try:
            line.setData(pos=np.empty((0, 3), dtype=float))
        except Exception:
            pass

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
    lab.lower()

def _show_revert_menu(widget: QLabel, pid: str):
    """Show context menu to revert a combined point back to pair .1/.2."""
    try:
        if '.' in pid:
            return
        m = QMenu(widget)
        act = m.addAction(f"Revert pair '{pid}' to {pid}.1 / {pid}.2")
        chosen = m.exec(widget.mapToGlobal(widget.rect().bottomLeft()))
        if chosen == act:
            _revert_combined(pid)
            _update_submit_state()
    except Exception:
        pass

def _ensure_hover_preview() -> QLabel:
    global HOVER_PREVIEW_LABEL
    if HOVER_PREVIEW_LABEL is None:
        lab = QLabel(parent=view)
        lab.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        lab.hide()
        HOVER_PREVIEW_LABEL = lab
    return HOVER_PREVIEW_LABEL

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
    if HOVER_PREVIEW_LABEL is not None:
        try:
            HOVER_PREVIEW_LABEL.hide()
        except Exception:
            pass

def _alignment_indicator_for(pid: str) -> str:
    """Return token id only (no HTML indicator)."""
    return pid

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
    lab.setText(pid)
    lab.adjustSize()
    lab.move(int(px - lab.width() // 2), int(py - lab.height() - LABEL_OVER_POINT_MARGIN))
    if '.' not in pid:
        try:
            lab.setToolTip("Rechtsklick: Paar aufteilen (zu .1 / .2)")
        except Exception:
            pass
    lab.show()
    lab.raise_()

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
    if '.' in pid:
        _update_pair_line(_category_of(pid))

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
view.setCameraPosition(distance=_fit_distance_for_extent(AXIS_LEN), elevation=20, azimuth=45)

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

def _position_debug_button():
    """Place debug toggle button bottom-right inside the scene view area."""
    try:
        btn_grid.adjustSize()
        x = view.width() - btn_grid.width() - BR_MARGIN_X
        y = view.height() - btn_grid.height() - BR_MARGIN_Y
        btn_grid.move(max(0, x), max(0, y))
        btn_grid.raise_()
    except Exception:
        pass

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
    show = _debug_on()
    if not show:
        for k in ('x','y','z'):
            for lab in axis_tick_labels[k]:
                lab.hide()
        return
    _ensure_axis_tick_labels()
    def disp_text(t):
        v = t - (AXIS_LEN * 0.5)
        if abs(t - 0.0) < 1e-6:
            return "-1"
        elif abs(t - AXIS_LEN * 0.5) < 1e-6:
            return "0"
        elif abs(t - AXIS_LEN) < 1e-6:
            return "+1"
        else:
            return f"{t:.2f}"  # fallback
    for idx, t in enumerate(TICK_WORLD_POS):
        pr = project_point((t, 0.0, 0.0))
        lab = axis_tick_labels['x'][idx]
        if pr is None:
            lab.hide(); continue
        px, py = pr
        lab.setText(disp_text(t))
        lab.adjustSize()
        lab.move(int(px - lab.width()//2), max(0, py - lab.height() - RANGE_LABEL_MARGIN))
        lab.show(); lab.raise_()
    for idx, t in enumerate(TICK_WORLD_POS):
        if idx == 0:
            axis_tick_labels['y'][idx].hide()
            continue
        pr = project_point((0.0, t, 0.0))
        lab = axis_tick_labels['y'][idx]
        if pr is None:
            lab.hide(); continue
        px, py = pr
        lab.setText(disp_text(t))
        lab.adjustSize()
        lab.move(int(px - lab.width()//2), max(0, py - lab.height() - RANGE_LABEL_MARGIN))
        lab.show(); lab.raise_()
    for idx, t in enumerate(TICK_WORLD_POS):
        if idx == 0:
            axis_tick_labels['z'][idx].hide()
            continue
        pr = project_point((0.0, 0.0, t))
        lab = axis_tick_labels['z'][idx]
        if pr is None:
            lab.hide(); continue
        px, py = pr
        lab.setText(disp_text(t))
        lab.adjustSize()
        OFFSET_X = 16
        lab.move(int(px + OFFSET_X), int(py - lab.height()//2))
        lab.show(); lab.raise_()

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
    """Position the overlay axis labels near the projected axis midpoints."""
    pts = {
        'x': (AXIS_LEN * 0.5, 0.0, 0.0),
        'y': (0.0, AXIS_LEN * 0.5, 0.0),
        'z': (0.0, 0.0, AXIS_LEN * 0.5),
    }
    labs = {'x': axis_label_x, 'y': axis_label_y, 'z': axis_label_z}
    for k in ('x', 'y', 'z'):
        res = project_point(pts[k])
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
        
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Panel / controls / params
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

panel = QFrame(parent=view)
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
panel.move(10, 10)
panel_layout = QVBoxLayout(panel)
panel_layout.setContentsMargins(10, 8, 10, 10)
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
    return row, edit, lab, h

def make_row_combo(caption: str, items: list[str], current_index: int = 0):
    """Create a labeled row with a QComboBox (unused inlined in this layout)."""
    row = QWidget(panel)
    h = QHBoxLayout(row)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(ROW_SPACING)
    lab = QLabel(caption, row)
    lab.setFixedWidth(LABEL_WIDTH)
    lab.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    combo = QComboBox(row)
    combo.addItems(items)
    if 0 <= current_index < combo.count():
        combo.setCurrentIndex(current_index)
    combo.setFixedHeight(CONTROL_H)
    h.addWidget(lab)
    h.addWidget(combo)
    return row, combo, lab, h

row_x, edit_x, lab_x, row_x_layout = make_row("X:", "set x-axis")
row_y, edit_y, lab_y, row_y_layout = make_row("Y:", "set y-axis")
row_z, edit_z, lab_z, row_z_layout = make_row("Z:", "set z-axis")
row_x_layout.setSpacing(ROW_SPACING)
row_y_layout.setSpacing(ROW_SPACING)
row_z_layout.setSpacing(ROW_SPACING)

panel_layout.addWidget(row_x)
panel_layout.addWidget(row_y)
panel_layout.addWidget(row_z)

params_panel = QFrame(parent=view)
params_panel.setFrameShape(QFrame.Shape.NoFrame)
params_panel.setStyleSheet("QFrame { background: none; }")
params_layout = QVBoxLayout(params_panel)
params_layout.setContentsMargins(0, 8, 0, 10)
params_layout.setSpacing(6)

combo_x_params = QComboBox(params_panel)
combo_x_params.addItems([
    "none","big-small","low-high","slow–fast","light–heavy","narrow–wide",
    "dark–bright","soft–sharp","rough–smooth","warm–cool",
    "safe–risky","positive–negative","active–passive","old–young",
])
combo_x_params.setFixedWidth(COMBO_WIDTH)
params_layout.addWidget(combo_x_params)

combo_y_params = QComboBox(params_panel)
combo_y_params.addItems([
    "none","big-small","low-high","slow–fast","light–heavy","narrow–wide",
    "dark–bright","soft–sharp","rough–smooth","warm–cool",
    "safe–risky","positive–negative","active–passive","old–young",
])
combo_y_params.setFixedWidth(COMBO_WIDTH)
params_layout.addWidget(combo_y_params)

combo_z_params = QComboBox(params_panel)
combo_z_params.addItems([
    "none","big-small","low-high","slow–fast","light–heavy","narrow–wide",
    "dark–bright","soft–sharp","rough–smooth","warm–cool",
    "safe–risky","positive–negative","active–passive","old–young",
])
combo_z_params.setFixedWidth(COMBO_WIDTH)
params_layout.addWidget(combo_z_params)

params_panel.adjustSize()
params_panel.show()

panel.setParent(left_col)
left_v.insertWidget(0, panel, 0)
try:
    params_panel.hide()
except Exception:
    pass

for lab in (lab_x, lab_y, lab_z):
    lab.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

edit_x.setFixedWidth(INPUT_WIDTH)
edit_y.setFixedWidth(INPUT_WIDTH)
edit_z.setFixedWidth(INPUT_WIDTH)
edit_x.setFixedHeight(CONTROL_H)
edit_y.setFixedHeight(CONTROL_H)
edit_z.setFixedHeight(CONTROL_H)

_PARAM_ITEMS = [
    "none","big-small","low-high","slow–fast","light–heavy","narrow–wide",
    "dark–bright","soft–sharp","rough–smooth","warm–cool",
    "safe–risky","positive–negative","active–passive","old–young",
]

combo_x_inline = QComboBox(row_x)
combo_x_inline.addItems(_PARAM_ITEMS)
combo_x_inline.setFixedWidth(COMBO_WIDTH)
combo_x_inline.setFixedHeight(CONTROL_H)
row_x_layout.addWidget(combo_x_inline)
row_x_layout.addStretch(1)

combo_y_inline = QComboBox(row_y)
combo_y_inline.addItems(_PARAM_ITEMS)
combo_y_inline.setFixedWidth(COMBO_WIDTH)
combo_y_inline.setFixedHeight(CONTROL_H)
row_y_layout.addWidget(combo_y_inline)
row_y_layout.addStretch(1)

combo_z_inline = QComboBox(row_z)
combo_z_inline.addItems(_PARAM_ITEMS)
combo_z_inline.setFixedWidth(COMBO_WIDTH)
combo_z_inline.setFixedHeight(CONTROL_H)
row_z_layout.addWidget(combo_z_inline)
row_z_layout.addStretch(1)

plane_row = QWidget(panel)
plane_box = QHBoxLayout(plane_row)
plane_box.setContentsMargins(0, 0, 0, 0)
plane_box.setSpacing(ROW_SPACING)
plane_lab = QLabel("Axes Focus:", plane_row)
plane_lab.setFixedWidth(LABEL_WIDTH)
plane_lab.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
btn_xy = QPushButton("Z-Y", plane_row)
btn_xy.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
btn_xy.setStyleSheet("QPushButton:pressed { background: grey; }")
btn_yz = QPushButton("Z-X", plane_row)
btn_yz.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
btn_yz.setStyleSheet("QPushButton:pressed { background: grey; }")
for b in (btn_xy, btn_yz):
    b.setMinimumWidth(52)
btn_xy.setFixedHeight(CONTROL_H)
btn_yz.setFixedHeight(CONTROL_H)
plane_box.addWidget(plane_lab)
plane_box.addWidget(btn_xy)
plane_box.addWidget(btn_yz)

cb_lock = QCheckBox("Lock View (^L)", plane_row)
cb_lock.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
cb_lock.setChecked(False)
cb_lock.setFixedHeight(CONTROL_H)
cb_lock.setStyleSheet("QCheckBox { color: #ffffff; }")
cb_lock.show()
plane_box.addWidget(cb_lock, 0, Qt.AlignmentFlag.AlignVCenter)
plane_box.addStretch(1)

lat_row = QWidget(panel)
lat_box = QHBoxLayout(lat_row)
lat_box.setContentsMargins(0, 0, 0, 0)
lat_box.setSpacing(ROW_SPACING)
lat_lab = QLabel("Grid-Steps:", lat_row)
lat_lab.setFixedWidth(LABEL_WIDTH)
lat_lab.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
lat_edit = QLineEdit(lat_row)
lat_edit.setText(str(AXIS_LEN))
lat_edit.setFixedWidth(INPUT_WIDTH)
lat_edit.setFixedHeight(CONTROL_H)
lat_edit.setReadOnly(True)
lat_edit.setCursor(QCursor(Qt.CursorShape.ForbiddenCursor))

grid_enable = QCheckBox("Enable Grid (^G)", lat_row)
grid_enable.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
grid_enable.setChecked(False)
grid_enable.setFixedHeight(CONTROL_H)
grid_enable.setStyleSheet("QCheckBox { color: #ffffff; }")
grid_enable.show()

lat_box.addWidget(lat_lab)
lat_box.addWidget(lat_edit)
lat_box.addWidget(grid_enable, 0, Qt.AlignmentFlag.AlignVCenter)
lat_box.addStretch(1)
panel_layout.addWidget(lat_row)

# Wire the plane grid checkbox toggle and set initial state
grid_enable.toggled.connect(_toggle_plane_grids_ui)
_toggle_plane_grids_ui(grid_enable.isChecked())

ztol_row = QWidget(panel)
ztol_box = QHBoxLayout(ztol_row)
ztol_box.setContentsMargins(0, 0, 0, 0)
ztol_box.setSpacing(ROW_SPACING)
ztol_lab = QLabel("Tolerance:", ztol_row)
ztol_lab.setFixedWidth(LABEL_WIDTH)
ztol_lab.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
ztol_edit = QLineEdit(ztol_row)
ztol_edit.setText(str(Z_ALIGN_EPS))
ztol_edit.setFixedWidth(INPUT_WIDTH)
ztol_edit.setFixedHeight(CONTROL_H)
ztol_edit.setReadOnly(True)
ztol_edit.setCursor(QCursor(Qt.CursorShape.ForbiddenCursor))
ztol_box.addWidget(ztol_lab)
ztol_box.addWidget(ztol_edit)
ztol_box.addStretch(1)
panel_layout.addWidget(ztol_row)

panel_layout.addWidget(plane_row)

def _current_z_tol():
    """Return current z alignment tolerance as float, fallback to constant on parse error."""
    try:
        return float(ztol_edit.text().strip())
    except Exception:
        ztol_edit.setText(str(Z_ALIGN_EPS))
        return Z_ALIGN_EPS

ztol_edit.returnPressed.connect(_update_all_point_labels)

btn_grid = QPushButton("Enable Debug (^D)", view)
btn_grid.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
btn_grid.setCheckable(True)
btn_grid.show()
_position_debug_button()

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

def _current_step():
    """Parse and return grid step value from UI, with safe fallback."""
    try:
        return float(lat_edit.text().strip())
    except Exception:
        lat_edit.setText("0.5")
        return 0.5

def _toggle_grid(checked: bool):
    """Enable/disable debug overlays (cube, lattice, lines, labels)."""
    if checked:
        show_cube()
        show_lattice(_current_step())
        show_pair_lines()
        _show_axis_tick_labels(True)
        _update_all_point_labels()
        btn_grid.setText("Disable Debug")
    else:
        hide_cube()
        hide_lattice()
        hide_pair_lines()
        _show_axis_tick_labels(False)
        for pid, lab in list(image_labels.items()):
            try:
                lab.hide()
            except Exception:
                pass
        btn_grid.setText("Enable Debug")

def _toggle_lock(checked: bool):
    globals()['LOCK_CAMERA'] = bool(checked)

btn_grid.toggled.connect(_toggle_grid)
cb_lock.toggled.connect(_toggle_lock)

def _apply_lattice_step():
    step = _current_step()
    if btn_grid.isChecked():
        show_lattice(step)

lat_edit.returnPressed.connect(_apply_lattice_step)

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
    view.setCameraPosition(distance=_fit_distance_for_extent(AXIS_LEN), elevation=20, azimuth=45)

btn_xy.clicked.connect(set_view_xy)
btn_yz.clicked.connect(set_view_yz)

panel.adjustSize()
panel.setMinimumSize(150, 250)
panel.show()

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Tokens / token dock
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

tokens_label = QLabel("Image Tokens:")
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

def _is_token_draggable(pid: str) -> bool:
    """Return whether a token is draggable in the current placement phase."""
    global placement_phase
    if placement_phase == 1:
        return pid.endswith('.1')
    return pid.endswith('.2')

def _update_token_states():
    """Update enabled/disabled state and styles of tokens according to phase and placed flag."""
    for t in point_tokens:
        placed = bool(t.property('placed'))
        if placed:
            t.setStyleSheet(_token_style_mode('placed'))
            t.setEnabled(False)
            continue
        if _is_token_draggable(t.pid):
            t.setEnabled(True)
            t.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
            t.setStyleSheet(_token_style_mode('active'))
            t.show()
        else:
            t.setEnabled(False)
            t.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            t.setStyleSheet(_token_style_mode('disabled'))
            t.show()
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

for i in range(1, 11):
    t1 = DraggableToken(f"{i}.1", parent=point_dock)
    t2 = DraggableToken(f"{i}.2", parent=point_dock)
    for t in (t1, t2):
        t.setMinimumWidth(52)
        t.setFixedHeight(27)
    point_tokens.extend([t1, t2])
    row = i - 1
    point_dock_layout.addWidget(t1, row, 0)
    point_dock_layout.addWidget(t2, row, 1)
    _update_token_states()

def _all_pairs_combined() -> bool:
    """Return True if all categories have a combined point (ImgN) placed and no .1/.2 remain."""
    cats = _token_categories()
    if not cats:
        return False
    for c in cats:
        if c not in placed_points:
            return False
        if f"{c}.1" in placed_points or f"{c}.2" in placed_points:
            return False
    return True

def _update_submit_state():
    try:
        btn_submit.setEnabled(_all_pairs_combined() & (btn_start.isEnabled() == False))
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

def _axis_display_name(edit_widget: QLineEdit, combo_widget: QComboBox, fallback: str) -> str:
    """Compose axis label display text including property/range in parentheses."""
    try:
        label = (edit_widget.text() or "").strip() or fallback
        rng = (combo_widget.currentText() or "").strip()
        if not rng or rng.lower() == "none":
            rng = "none"
        return f"{label}({rng})"
    except Exception:
        return f"{fallback}(none)"

def _export_results():
    """Export combined points to CSV if all pairs are combined."""
    log_session_event("submitted, finished experiment")
    if not _all_pairs_combined():
        _update_submit_state()
        return
    btn_start.setEnabled(True)
    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base, "results")
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    x_name = _axis_display_name(edit_x, combo_x_inline, "X")
    y_name = _axis_display_name(edit_y, combo_y_inline, "Y")
    z_name = _axis_display_name(edit_z, combo_z_inline, "Z")
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

def _revert_combined(cat: str):
    """
    Split a combined point back into ImgN.1 (YZ plane) and ImgN.2 (XZ plane).
    Place points near the combined coordinates and mark tokens as placed.
    """
    try:
        combined_pid = cat
        if '.' in combined_pid:
            return
        if combined_pid not in placed_points:
            return
        _, c = placed_points[combined_pid]
        Xc, Yc, Zc = float(c[0]), float(c[1]), float(c[2])
        p1 = (PLANE_OFFSETS['yz'], Yc, Zc)
        p2 = (Xc, PLANE_OFFSETS['xz'], Zc)
        p1 = clamp_to_cube(*p1)
        p2 = clamp_to_cube(*p2)
        pid1 = f"{cat}.1"
        pid2 = f"{cat}.2"
        _set_point_position(pid1, p1)
        _set_point_position(pid2, p2)
        for t in point_tokens:
            if t.pid in (pid1, pid2):
                t.setProperty('placed', True)
                t.setStyleSheet(_token_style_mode('placed'))
                t.hide()
        _remove_placed_point(combined_pid)
        _update_pair_line(cat)
        _update_token_states()
    except Exception:
        pass
    log_session_event("Reverted Pair")
    _update_submit_state()

def _combine_pairs():
    """Combine each .1 + .2 pair into a single combined point ImgN if both placed."""
    eps = 1e-6
    for i in range(1, 11):
        pid1 = f"{i}.1"
        pid2 = f"{i}.2"
        if pid1 in placed_points and pid2 in placed_points:
            _, c1 = placed_points[pid1]
            _, c2 = placed_points[pid2]
            yz_pt = None
            xz_pt = None
            if abs(float(c1[0]) - float(PLANE_OFFSETS['yz'])) < eps:
                yz_pt = c1
            if abs(float(c2[0]) - float(PLANE_OFFSETS['yz'])) < eps:
                yz_pt = c2
            if abs(float(c1[1]) - float(PLANE_OFFSETS['xz'])) < eps:
                xz_pt = c1
            if abs(float(c2[1]) - float(PLANE_OFFSETS['xz'])) < eps:
                xz_pt = c2
            if yz_pt is not None and xz_pt is not None:
                x = float(xz_pt[0])
                y = float(yz_pt[1])
                z = 0.5 * (float(yz_pt[2]) + float(xz_pt[2]))
            else:
                x = 0.5 * (float(c1[0]) + float(c2[0]))
                y = 0.5 * (float(c1[1]) + float(c2[1]))
                z = 0.5 * (float(c1[2]) + float(c2[2]))
            x, y, z = clamp_to_cube(x, y, z)
            pid_combined = f"{i}"
            _set_point_position(pid_combined, (x, y, z))
            _remove_placed_point(pid1)
            _remove_placed_point(pid2)
            cat = f"{i}"
            if cat in pair_lines:
                try:
                    view.removeItem(pair_lines[cat])
                except Exception:
                    pass
                try:
                    pair_lines[cat].setData(pos=np.empty((0, 3), dtype=float))
                except Exception:
                    pass
                del pair_lines[cat]
    set_view_default()
    log_session_event("Combined Pairs")

point_dock.adjustSize()
point_dock.show()

btn_combine = QPushButton("Combine  (^K)", left_col)
btn_reset = QPushButton("Reset  (^R)", left_col)
btn_submit = QPushButton("Submit  (^↩)", left_col)
btn_start = QPushButton("Start", left_col)

btn_combine.setFixedSize(110, 25)
btn_reset.setFixedSize(110, 25)
btn_submit.setFixedSize(110, 25)
btn_start.setFixedSize(110, 25)

btn_combine.clicked.connect(_combine_pairs)
btn_reset.clicked.connect(_reset_all_points)
btn_start.clicked.connect(_start_experiment)
btn_start.clicked.connect(_reset_all_points)

btn_submit.setEnabled(False)
btn_submit.clicked.connect(lambda: _export_results())

btn_combine.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
btn_reset.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
btn_submit.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
btn_start.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

btn_combine.setStyleSheet("""
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
btn_reset.setStyleSheet(btn_combine.styleSheet())
btn_submit.setStyleSheet(btn_combine.styleSheet())
btn_start.setStyleSheet(btn_combine.styleSheet() + """QPushButton { background: #00cc66;}""")

try:
    sc_combine = QShortcut(QKeySequence("Meta+K"), win)
    sc_combine.activated.connect(_combine_pairs)
    sc_reset = QShortcut(QKeySequence("Meta+R"), win)
    sc_reset.activated.connect(_reset_all_points)
    sc_debug = QShortcut(QKeySequence("Meta+D"), win)
    sc_submit = QShortcut(QKeySequence("Meta+Return"), win)
    sc_submit.activated.connect(lambda: _export_results())
    sc_debug.activated.connect(lambda: btn_grid.setChecked(not btn_grid.isChecked()))
    sc_lock = QShortcut(QKeySequence("Meta+L"), win)
    sc_lock.activated.connect(lambda: cb_lock.setChecked(not cb_lock.isChecked()))
    sc_grid = QShortcut(QKeySequence("Meta+G"), win)
    sc_grid.activated.connect(lambda: grid_enable.setChecked(not grid_enable.isChecked()))
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

btn_combine.setParent(preview_col)
btn_combine.move(0, 240)
btn_reset.setParent(preview_col)
btn_reset.move(120, 240)
btn_submit.setParent(preview_col)
btn_submit.move(120, 275)
btn_start.setParent(preview_col)
btn_start.move(0, 275)

console_box = QPlainTextEdit(preview_col)
console_box.setReadOnly(True)
console_box.setStyleSheet("""
    QPlainTextEdit {
        background: rgba(255,255,255,0.1);
        color: #ddd;
        border-radius: 6px;
        font-family: Consolas, monospace;
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
    """Update a token as placed (hide/green) and advance placement phase if needed."""
    global placement_phase
    for t in point_tokens:
        if t.pid == pid:
            t.setProperty('placed', True)
            t.setStyleSheet(_token_style_mode('placed'))
            t.hide()
            _ensure_image_label(pid)
            break
    if placement_phase == 1:
        all_one_placed = True
        for t in point_tokens:
            if t.pid.endswith('.1') and not bool(t.property('placed')):
                all_one_placed = False
                break
        if all_one_placed:
            placement_phase = 2
            set_view_yz()  
    _update_token_states()

def _mark_token_unplaced(pid: str):
    """Mark a token as unplaced and update UI state."""
    for t in point_tokens:
        if t.pid == pid:
            t.setProperty('placed', False)
            t.show()
            break
    _update_token_states()

btn_grid.setChecked(False)
cb_lock.setChecked(False)

def apply_labels():
    """Apply axis labels from edit fields to the overlay labels and header."""
    tx, ty, tz = edit_x.text().strip(), edit_y.text().strip(), edit_z.text().strip()
    axis_label_x.setText(tx)
    axis_label_y.setText(ty)
    axis_label_z.setText(tz)
    for lab in (axis_label_x, axis_label_y, axis_label_z):
        lab.show()
        lab.raise_()
    header_label.setText(f"X: {tx}    Y: {ty}    Z: {tz}")
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
_position_debug_button()

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
    _position_debug_button()
    _update_token_states()
    position_axis_labels()
    _update_all_point_labels()

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