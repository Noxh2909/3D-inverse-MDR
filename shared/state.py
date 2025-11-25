import sys
from typing import Dict, Optional
from PySide6.QtGui import QGuiApplication, QPixmap
from PySide6.QtWidgets import QLabel, QFrame, QPushButton
from pyqtgraph.opengl import GLLinePlotItem

from shared.const import *

Z_ALIGN_EPS = 0.5
VIEW = None
#
def set_view(v):
    """Set the global VIEW variable for overlay labels."""
    global VIEW
    VIEW = v

def ensure_qt_ready():
    """
    Ensure that a QGuiApplication exists so that QPixmap, QLabel, etc. can
    be safely instantiated, even if no visible QApplication window exists yet.
    """
    if QGuiApplication.instance() is None:
        QGuiApplication(sys.argv)

def set_z_tolerance(val: float):
    """Set Z alignment tolerance value."""
    global Z_ALIGN_EPS
    Z_ALIGN_EPS = float(val)

def get_z_tolerance() -> float:
    """Return current Z alignment tolerance value."""
    return float(Z_ALIGN_EPS)

def category_of(pid: str) -> str:
    """Return the category part of a point id (before the dot)."""
    return pid.split('.')[0]

def partner_of(pid: str) -> Optional[str]:
    """Return the partner point id for a given point id, or None."""
    if '.' not in pid:
        return None
    cat = category_of(pid)
    return f"{cat}.2" if pid.endswith('.1') else f"{cat}.1"

def debug_on(btn: QPushButton) -> bool:
    return bool(btn.isChecked())

# Data structures for experiment state
working: Dict[str, Optional[float]] = {'x': None, 'y': None, 'z': None}
placed_points: Dict[str, tuple] = {}
point_labels: Dict[str, QLabel] = {}
pair_lines: Dict[str, GLLinePlotItem] = {}
axis_tick_labels = {'x': [], 'y': [], 'z': []}
points = []
cube_items = []
point_tokens = []
lattice_items = []
current_plane = 'xy'
placement_phase = 1
show_images = False
LOCK_CAMERA = False

# Image- and GUI-related state
POINT_DOCK: QFrame | None = None
IMAGES_BY_CAT: Dict[str, QPixmap] = {}
IMAGES_ORIG: Dict[str, QPixmap] = {}
HOVER_PREVIEW_LABEL: Optional[QLabel] = None
TICK_WORLD_POS = [0.0, AXIS_LEN * 0.5, AXIS_LEN]
image_labels: Dict[str, QLabel] = {}