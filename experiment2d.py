import sys
from datetime import datetime

from shared.const import *
import shared.state as state
import shared.tokens as token
import shared.scene_axes as scene_axes
import shared.scene_objects as scene_objects
import shared.scene_cube as scene_cube
import shared.geometry as geometry
import shared.overlays_labels as overlay_labels
import shared.overlays_utils as overlay_utils
import shared.log as log
import shared.files as files
import shared.camera as camera
import shared.ui_panel as ui
import shared.window as window

from PySide6.QtWidgets import (
    QApplication, QWidget, QFrame, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QCheckBox, QMainWindow, QSizePolicy, QGridLayout, QScrollArea, QPlainTextEdit
)
from PySide6.QtGui import QDrag, QCursor, QFont, QKeySequence, QShortcut
from PySide6.QtCore import Qt, QTimer, QMimeData, QPoint, QObject, QEvent
from pyqtgraph.opengl import GLViewWidget, GLGridItem

EXPERIMENT_MODE_2D = True

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
                    token.show_revert_menu(obj, pid, btn_submit, btn_start)
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
        chosen = geometry.choose_plane_and_hit(px, py)
        if chosen is None:
            ev.ignore()
            return
        plane_used, hit = chosen
        x, y, z = geometry.clamp_to_cube(*hit)
        if plane_used == 'xy':
            z = PLANE_OFFSETS['xy']
        elif plane_used == 'xz':
            y = PLANE_OFFSETS['xz']
        else:
            x = PLANE_OFFSETS['yz']
        scene_objects.set_point_position(point_id, (x, y, z))
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
            for pid, (_item, coords) in state.placed_points.items():
                if '.' not in pid:
                    continue
                proj = geometry.project_point(coords)
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
        if not state.LOCK_CAMERA:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        """Hover picking: show hover preview for nearest visible point if within radius."""
        try:
            pos = ev.position().toPoint()
            mx, my = pos.x(), pos.y()
            hit_pid = None
            best_d2 = 1e9
            best_proj = None

            for pid, (_item, coords) in state.placed_points.items():
                if not geometry._is_point_visible_world(coords):
                    continue
                proj = geometry.project_point(coords)
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
                overlay_utils.set_preview_for_category(cat, preview_box)  
                if best_proj is not None:
                    # _show_hover_preview(cat, QPoint(int(best_proj[0]), int(best_proj[1])))
                    pass
            else:
                overlay_utils._hide_hover_preview()
        except Exception:
            pass

        if state.LOCK_CAMERA:
            return
        super().mouseMoveEvent(ev)

    def wheelEvent(self, ev):
        """Allow wheel zoom/behavior unless camera is locked."""
        if state.LOCK_CAMERA:
            return
        super().wheelEvent(ev)

    def leaveEvent(self, ev):
        try:
            overlay_utils._hide_hover_preview()
        except Exception:
            pass
        super().leaveEvent(ev)


class DraggableToken(QLabel):
    """Small QLabel that acts as draggable token in the left dock."""

    def __init__(self, pid: str, parent=None):
        super().__init__(pid, parent)
        self.pid = pid
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        self.setStyleSheet(token.token_style_mode('disabled'))
        # self.setMinimumWidth(80)
        self.setFixedSize(55, 18)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)

    def enterEvent(self, ev):
        try:
            cat = self.pid.split('.')[0]
            _show_hover_preview_over_dock(cat, point_dock)
        except Exception:
            pass
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        overlay_utils._hide_hover_preview()
        super().leaveEvent(ev)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self.drag_start_pos = ev.position().toPoint()

    def mouseMoveEvent(self, ev):
        if not ev.buttons() & Qt.MouseButton.LeftButton:
            return
        if (ev.position().toPoint() - self.drag_start_pos).manhattanLength() < 5:
            return
        if not token.is_token_draggable(self.pid) or not self.isEnabled():
            return

        drag = QDrag(self)
        mime = QMimeData()
        mime.setData('application/x-point-id', self.pid.encode('utf-8'))
        drag.setMimeData(mime)

        cat = self.pid.split('.')[0]
        pm = state.IMAGES_BY_CAT.get(cat)
        if pm and not pm.isNull():
            drag.setPixmap(pm)
            hs = QPoint(pm.width() // 2, int(pm.height() * 0.8))
            drag.setHotSpot(hs)

        overlay_utils.set_preview_for_category(cat, preview_box) 
        overlay_utils._hide_hover_preview()
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

left_col = QFrame()
left_col.setFrameShape(QFrame.Shape.StyledPanel)
left_col.setStyleSheet("QFrame { border: 0px solid #666; border-radius: 8px; }")
left_col.setMinimumWidth(120)
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
state.VIEW = view

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

win.setFixedSize(1400, 800)
win.show()
window.center_on_screen(win)

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Geometry / projection / raycasting / time helpers
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

def _start_experiment():
    btn_start.setEnabled(False)
    cb_lock.setChecked(True)
    
    log.start_time = datetime.now()
    log.log_session_event("Experiment started...")
    camera.set_view_xy()
    # _position_header()
    
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Axes, ticks and grid helpers
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

axis_items = scene_axes.build_axes_with_ticks(AXIS_LEN)
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

view.opts['center'] = camera._cube_center()
view.setCameraPosition(distance=camera._fit_distance_for_extent(AXIS_LEN), elevation=20, azimuth=45)

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Header / axis label overlays and tick labels
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

header_label = QLabel("", parent=view)
header_label.setStyleSheet("color: #ffffff; font-size: 18px; font-weight: 600; background: transparent;")
header_label.raise_()

def position_debug_button():
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

row_x, edit_x, lab_x, row_x_layout = ui.make_row("X:", "set x-axis", panel)
row_y, edit_y, lab_y, row_y_layout = ui.make_row("Y:", "set y-axis", panel)
row_z, edit_z, lab_z, row_z_layout = ui.make_row("Z:", "set z-axis", panel)
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
grid_enable.toggled.connect(
    lambda checked: ui.toggle_plane_grids_ui(checked, yz_grid, xz_grid)
)
ui.toggle_plane_grids_ui(grid_enable.isChecked(), yz_grid, xz_grid)

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
ztol_edit.returnPressed.connect(lambda: overlay_labels.update_all_point_labels(POINT_LABEL_FILTER))

btn_grid = QPushButton("Show Image (^D)", view)
btn_grid.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
btn_grid.setCheckable(True)
btn_grid.show()
position_debug_button()

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
    
btn_grid.toggled.connect(
    lambda checked: ui.toggle_grid(checked, btn_grid))
cb_lock.toggled.connect(ui.toggle_lock)
btn_xy.clicked.connect(camera.set_view_xy)
btn_yz.clicked.connect(camera.set_view_yz)

panel.adjustSize()
panel.setMinimumSize(150, 250)
panel.show()

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Tokens / token dock
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

tokens_label = QLabel("Images:")
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

for i in range(1, 17):
    t1 = DraggableToken(f"{i}.1", parent=point_dock)
    t2 = DraggableToken(f"{i}.2", parent=point_dock)
    for t in (t1, t2):
        t.setMinimumWidth(52)
        t.setFixedHeight(27)
    state.point_tokens.extend([t1, t2])
    row = i - 1
    point_dock_layout.addWidget(t1, row, 0)
    point_dock_layout.addWidget(t2, row, 1)
    token.update_token_states()

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

btn_combine.clicked.connect(lambda: token.combine_pairs(btn_submit, btn_start))
btn_reset.clicked.connect(scene_objects.reset_all_points)
btn_start.clicked.connect(_start_experiment)
btn_start.clicked.connect(scene_objects.reset_all_points)

btn_submit.setEnabled(False)
btn_submit.clicked.connect(
    lambda: files.export_results(
        btn_start, btn_submit,
        edit_x, edit_y, edit_z,
        combo_x_inline, combo_y_inline, combo_z_inline,
        header_label, app
    )
)

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
btn_start.setStyleSheet(btn_combine.styleSheet() + """QPushButton { background: #00cc66; border: solid lightgray;}""")

try:
    sc_combine = QShortcut(QKeySequence("Meta+K"), win)
    sc_combine.activated.connect(token.combine_pairs(btn_submit, btn_start))
    sc_reset = QShortcut(QKeySequence("Meta+R"), win)
    sc_reset.activated.connect(scene_objects.reset_all_points)
    sc_debug = QShortcut(QKeySequence("Meta+D"), win)
    sc_submit = QShortcut(QKeySequence("Meta+Return"), win)
    sc_submit.activated.connect(lambda: files.export_results(
            btn_start, btn_submit,
            edit_x, edit_y, edit_z,
            combo_x_inline, combo_y_inline, combo_z_inline,
            header_label, app
        )
    )
    sc_debug.activated.connect(lambda: btn_grid.setChecked(not btn_grid.isChecked()))
    sc_lock = QShortcut(QKeySequence("Meta+L"), win)
    sc_lock.activated.connect(lambda: cb_lock.setChecked(not cb_lock.isChecked()))
    sc_grid = QShortcut(QKeySequence("Meta+G"), win)
    sc_grid.activated.connect(lambda: grid_enable.setChecked(not grid_enable.isChecked()))
except Exception:
    pass
token.update_submit_state(btn_submit, btn_start)

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

# --- Console Output Box (experiment log display) ---
from shared.log import set_console_box

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
console_box.show()       
console_box.raise_()      

set_console_box(console_box)
image_h.addWidget(token_col, 0)
image_h.addWidget(preview_col, 0)
left_v.addWidget(image_row, 0)

def _mark_token_placed(pid: str):
    """Update a token as placed (hide/green) and advance placement phase if needed."""
    global placement_phase
    for t in state.point_tokens:
        if t.pid == pid:
            t.setProperty('placed', True)
            t.setStyleSheet(token.token_style_mode('placed'))
            t.hide()
            overlay_labels.ensure_image_label(pid)
            break
    if state.placement_phase == 1:
        all_one_placed = True
        for t in state.point_tokens:
            if t.pid.endswith('.1') and not bool(t.property('placed')):
                all_one_placed = False
                break
        if all_one_placed:
            state.placement_phase = 2
            camera.set_view_yz()
    token.update_token_states()

def _mark_token_unplaced(pid: str):
    """Mark a token as unplaced and update UI state."""
    for t in state.point_tokens:
        if t.pid == pid:
            t.setProperty('placed', False)
            t.show()
            break
    token.update_token_states()

btn_grid.setChecked(False)
cb_lock.setChecked(False)

for e in (edit_x, edit_y, edit_z):
    e.returnPressed.connect(
        lambda: overlay_labels.apply_axis_labels(
        edit_x, edit_y, edit_z,
        axis_label_x, axis_label_y, axis_label_z,
        header_label
    ))

files.load_images_for_categories()
overlay_labels.apply_axis_labels(
        edit_x, edit_y, edit_z,
        axis_label_x, axis_label_y, axis_label_z,
        header_label
    )
token.update_token_states()
token.update_submit_state(btn_submit, btn_start)
position_debug_button()

view.sigMouseMoved = getattr(view, 'sigMouseMoved', None)

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Runtime: timers, resize handlers, shortcuts
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

timer = QTimer()
timer.setInterval(1000 // 60)
timer.timeout.connect(
    lambda: overlay_labels.position_axis_labels(axis_label_x, axis_label_y, axis_label_z)
)
timer.timeout.connect(
    lambda: overlay_labels.update_all_point_labels(POINT_LABEL_FILTER))
timer.start()

old_win_resize = win.resizeEvent
def win_resize(ev):
    if old_win_resize:
        old_win_resize(ev)
    position_debug_button()
    token.update_token_states()
    overlay_labels.position_axis_labels(axis_label_x, axis_label_y, axis_label_z)
    overlay_labels.update_all_point_labels(POINT_LABEL_FILTER)

win.resizeEvent = win_resize

"""TODO"""
try:
    _old__show_hover_preview_over_dock = overlay_utils._show_hover_preview_over_dock
    def _show_hover_preview_over_dock(cat: str, point_dock: QFrame | None = None):
        _old__show_hover_preview_over_dock(cat, point_dock)
        try:
            overlay_utils.set_preview_for_category(cat, preview_box)
        except Exception:
            pass
except Exception:
    pass
sys.exit(app.exec())