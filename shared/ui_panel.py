from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QLineEdit, QFrame, QPushButton
from pyqtgraph.opengl import GLGridItem
from PySide6.QtCore import Qt

from shared.const import *
import shared.scene_axes as scene_axes
import shared.scene_cube as scene_cube
import shared.overlays_labels as overlay_labels
import shared.state as state

def make_row(caption: str, default_text: str, panel: QFrame):
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

def toggle_plane_grids_ui(checked: bool, yz_grid: GLGridItem, xz_grid: GLGridItem):
    """Toggle only the YZ/XZ plane grids via the UI checkbox, independent of Debug."""
    if checked:
        scene_axes.show_plane_grids(state.VIEW, yz_grid, xz_grid)
    else:
        scene_axes.hide_plane_grids(state.VIEW, yz_grid, xz_grid)
    
def toggle_grid(checked: bool, btn: QPushButton):
    state.show_images = checked
    """Enable/disable debug overlays (cube, lattice, lines, labels)."""
    if checked:
        scene_cube.show_cube()
        overlay_labels.update_all_point_labels()
        btn.setText("Hide Image (^D)")
    else:
        scene_cube.hide_cube()
        for pid, lab in list(state.image_labels.items()):
            try:
                lab.hide()
            except Exception:
                pass
        btn.setText("Show Image (^D)")
        
def toggle_lock(checked: bool):
    state.LOCK_CAMERA = bool(checked)