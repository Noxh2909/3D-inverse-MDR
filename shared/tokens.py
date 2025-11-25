from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QPushButton, QLabel, QMenu
from PySide6.QtCore import Qt
from shared.overlays_utils import _hide_hover_preview

import shared.state as state
import shared.log as log
import shared.geometry as geometry
import shared.scene_objects as scene_objects
import shared.tokens as token
import shared.camera as camera
from shared.const import *


def token_style(placed: bool) -> str:
    """Return CSS style for token labels depending on placed state."""
    if placed:
        return "QLabel { color: #111; background: #7CFC00; border: 1px solid #3a3; border-radius: 4px; padding: 2px 6px; font-weight: 600; }"
    return "QLabel { color: #eee; background: #444; border: 1px solid #999; border-radius: 4px; padding: 2px 6px; } QLabel:hover { background: #555; }"

def token_style_mode(mode: str) -> str:
    """Return CSS for named token modes: 'placed', 'disabled', default active."""
    if mode == 'placed':
        return "QLabel { color: #111; background: #7CFC00; border: 1px solid #3a3; border-radius: 4px; padding: 2px 6px; font-weight: 600; }"
    if mode == 'disabled':
        return "QLabel { color: #aaa; background: #333; border: 1px solid #666; border-radius: 4px; padding: 2px 6px; }"
    return "QLabel { color: #eee; background: #444; border: 1px solid #999; border-radius: 4px; padding: 2px 6px; } QLabel:hover { background: #555; }"

def is_token_draggable(pid: str) -> bool:
    """Return whether a token is draggable in the current placement phase."""
    global placement_phase
    if state.placement_phase == 1:
        return pid.endswith('.1')
    return pid.endswith('.2')

def update_token_states():
    """Update enabled/disabled state and styles of tokens according to phase and placed flag."""
    for t in state.point_tokens:
        placed = bool(t.property('placed'))
        if placed:
            t.setStyleSheet(token_style_mode('placed'))
            t.setEnabled(False)
            continue
        if is_token_draggable(t.pid):
            t.setEnabled(True)
            t.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
            t.setStyleSheet(token_style_mode('active'))
            t.show()
        else:
            t.setEnabled(False)
            t.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            t.setStyleSheet(token_style_mode('disabled'))
            t.show()
    _hide_hover_preview()
    
def update_submit_state(submit_button: QPushButton | None = None, start_button: QPushButton | None = None):
    """Aktiviere den Submit-Button nur, wenn das Experiment gestartet wurde UND alle Paare kombiniert sind."""
    if submit_button is None or start_button is None:
        return
    try:
        submit_button.setEnabled(all_pairs_combined() & (start_button.isEnabled() == False))
    except Exception:
        pass
    
def remove_placed_point(pid: str):
    """Remove a placed point's sprite and overlays."""
    if pid in state.placed_points:
        try:
            it, _ = state.placed_points[pid]
            state.VIEW.removeItem(it)
        except Exception:
            pass
        state.placed_points.pop(pid, None)
    if pid in state.point_labels:
        try:
            state.point_labels[pid].hide()
            state.point_labels[pid].deleteLater()
        except Exception:
            pass
        state.point_labels.pop(pid, None)
    if pid in state.image_labels:
        try:
            state.image_labels[pid].hide()
            state.image_labels[pid].deleteLater()
        except Exception:
            pass
        state.image_labels.pop(pid, None)
    update_submit_state()
    
def token_categories() -> list[str]:
    """Return category names derived from point token ids or default list."""
    cats = []
    try:
        for t in state.point_tokens: 
            cat = t.pid.split('.')[0]
            if cat not in cats:
                cats.append(cat)
    except Exception:
        pass
    if not cats:
        cats = [f"{i}" for i in range(1, 16)]
    return cats
    
def all_pairs_combined() -> bool:
    """Return True if all categories have a combined point (ImgN) placed and no .1/.2 remain."""
    cats = token_categories()
    if not cats:
        return False
    for c in cats:
        if c not in state.placed_points:
            return False
        if f"{c}.1" in state.placed_points or f"{c}.2" in state.placed_points:
            return False
    return True

def collect_combined_points_norm():
    """Collect combined points and normalize world coords from [0,AXIS_LEN] to [-1,1]."""
    data = []
    L = float(AXIS_LEN)
    half = L * 0.5
    for name, (_item, coords) in state.placed_points.items():
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

def revert_combined(cat: str, btn1: QPushButton, btn2: QPushButton):
    """
    Split a combined point back into ImgN.1 (YZ plane) and ImgN.2 (XZ plane).
    Place points near the combined coordinates and mark tokens as placed.
    """
    try:
        combined_pid = cat
        if '.' in combined_pid:
            return
        if combined_pid not in state.placed_points:
            return
        _, c = state.placed_points[combined_pid]
        Xc, Yc, Zc = float(c[0]), float(c[1]), float(c[2])
        p1 = (PLANE_OFFSETS['yz'], Yc, Zc)
        p2 = (Xc, PLANE_OFFSETS['xz'], Zc)
        p1 = geometry.clamp_to_cube(*p1)
        p2 = geometry.clamp_to_cube(*p2)
        pid1 = f"{cat}.1"
        pid2 = f"{cat}.2"
        scene_objects.set_point_position(pid1, p1)
        scene_objects.set_point_position(pid2, p2)
        for t in state.point_tokens:
            if t.pid in (pid1, pid2):
                t.setProperty('placed', True)
                t.setStyleSheet(token.token_style_mode('placed'))
                t.hide()
        token.remove_placed_point(combined_pid)
        token.update_token_states()
    except Exception:
        pass
    log.log_session_event("Reverted Pair")
    token.update_submit_state(btn1, btn2)
    for lab in state.image_labels.values():
        lab.hide()
    state.show_images = False

def combine_pairs(btn1: QPushButton, btn2: QPushButton):
    """Combine each .1 + .2 pair into a single combined point ImgN if both placed."""
    eps = 1e-6
    for i in range(1, 17):
        pid1 = f"{i}.1"
        pid2 = f"{i}.2"
        if pid1 in state.placed_points and pid2 in state.placed_points:
            _, c1 = state.placed_points[pid1]
            _, c2 = state.placed_points[pid2]
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
            x, y, z = geometry.clamp_to_cube(x, y, z)
            pid_combined = f"{i}"
            scene_objects.set_point_position(pid_combined, (x, y, z))
            token.remove_placed_point(pid1)
            token.remove_placed_point(pid2)
            cat = f"{i}"
            if cat in state.pair_lines:
                try:
                    state.VIEW.removeItem(state.pair_lines[cat])
                except Exception:
                    pass
                try:
                    state.pair_lines[cat].setData(pos=np.empty((0, 3), dtype=float))
                except Exception:
                    pass
                del state.pair_lines[cat]
    state.VIEW.setCameraPosition(distance=camera._fit_distance_for_extent(AXIS_LEN),elevation=20, azimuth=45)
    token.update_submit_state(btn1, btn2)
    log.log_session_event("Combined Pairs")
    for lab in state.image_labels.values():
        lab.hide()
    state.show_images = False
    
def show_revert_menu(widget: QLabel, pid: str, btn1: QPushButton, btn2: QPushButton):
    """Show context menu to revert a combined point back to pair .1/.2."""
    try:
        if '.' in pid:
            return
        m = QMenu(widget)
        act = m.addAction(f"Revert pair '{pid}' to {pid}.1 / {pid}.2")
        chosen = m.exec(widget.mapToGlobal(widget.rect().bottomLeft()))
        if chosen == act:
            token.revert_combined(pid, btn1, btn2)
            token.update_submit_state(btn1, btn2)
    except Exception:
        pass