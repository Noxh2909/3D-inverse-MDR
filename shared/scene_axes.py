from pyqtgraph.opengl import GLLinePlotItem
from PySide6.QtWidgets import QLineEdit, QComboBox
import numpy as np
from shared.const import *

def axis_segment(p0, p1, color=(1,1,1,1), width=3):
    pts = np.array([p0, p1], dtype=float)
    return GLLinePlotItem(pos=pts, color=color, width=width, mode='lines')

def auto_tick_step(L: float) -> float:
    """Choose a readable tick step based on axis length L."""
    if L <= 10:
        return 1.0
    if L <= 20:
        return 2.0
    return max(1.0, round(L / 10.0, 1))

def build_axis_solid(axis: str, L: float, color=(1,1,1,1), width=3):
    if axis == 'x':
        return [axis_segment((0,0,0), (L,0,0), color=color, width=width)]
    elif axis == 'y':
        return [axis_segment((0,0,0), (0,L,0), color=color, width=width)]
    else:
        return [axis_segment((0,0,0), (0,0,L), color=color, width=width)]

def build_axis_dashed(axis: str, L: float, color=(1,1,1,1), width=3,
                       dash_len: float=DASH_LEN, gap_len: float=GAP_LEN):
    items = []
    s = 0.0
    step = max(1e-6, float(dash_len + gap_len))
    while s < L - 1e-9:
        e = min(L, s + dash_len)
        if axis == 'x':
            items.append(axis_segment((s,0,0), (e,0,0), color=color, width=width))
        elif axis == 'y':
            items.append(axis_segment((0,s,0), (0,e,0), color=color, width=width))
        else:
            items.append(axis_segment((0,0,s), (0,0,e), color=color, width=width))
        s += step
    return items

def build_axis_ticks(axis: str, L: float, tick_step: float=0,
                      tick_size: float=TICK_SIZE, color=(1,1,1,0.9), width=2):
    items = []
    if tick_step is None:
        tick_step = auto_tick_step(L)
    t = float(tick_step)
    while t < L + 1e-9:
        if axis == 'x':
            p0 = (t, 0.0, 0.0)
            p1 = (t, 0.0, tick_size)
            items.append(axis_segment(p0, p1, color=color, width=width))
        elif axis == 'y':
            p0 = (0.0, t, 0.0)
            p1 = (0.0, t, tick_size)
            items.append(axis_segment(p0, p1, color=color, width=width))
        else:
            leg = tick_size
            items.append(axis_segment((0.0, 0.0, t), (leg, 0.0, t), color=color, width=width))
            items.append(axis_segment((0.0, 0.0, t), (0.0, leg, t), color=color, width=width))
        t += tick_step
    return items

def build_axes_with_ticks(L: float):
    items = []
    step = auto_tick_step(L)
    items += build_axis_solid('x', L, color=(1,0,0,1), width=3)
    items += build_axis_ticks('x', L, tick_step=step, color=(1,0,0,0.9), width=2)
    items += build_axis_solid('y', L, color=(0,1,0,1), width=3)
    items += build_axis_ticks('y', L, tick_step=step, color=(0,1,0,0.9), width=2)
    items += build_axis_solid('z', L, color=(0,0,1,1), width=3)
    items += build_axis_ticks('z', L, tick_step=step, color=(0,0,1,0.9), width=2)
    return items

def axis_display_name(edit_widget: QLineEdit, combo_widget: QComboBox, fallback: str) -> str:
    """Compose axis label display text including property/range in parentheses."""
    try:
        label = (edit_widget.text() or "").strip() or fallback
        rng = (combo_widget.currentText() or "").strip()
        if not rng or rng.lower() == "none":
            rng = "none"
        return f"{label}({rng})"
    except Exception:
        return f"{fallback}(none)"

def show_plane_grids(view, yz_grid, xz_grid):
    """
    Add the YZ and XZ grid items to the given scene view.
    This function is UI-independent and should be called by the UI when grids are to be shown.
    """
    try:
        view.addItem(yz_grid)
    except Exception:
        pass
    try:
        view.addItem(xz_grid)
    except Exception:
        pass

def hide_plane_grids(view, yz_grid, xz_grid):
    """
    Remove the YZ and XZ grid items from the given scene view.
    This function is UI-independent and should be called by the UI when grids are to be hidden.
    """
    try:
        view.removeItem(yz_grid)
    except Exception:
        pass
    try:
        view.removeItem(xz_grid)
    except Exception:
        pass