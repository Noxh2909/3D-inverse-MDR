import numpy as np
from pyqtgraph.opengl import GLLinePlotItem

from shared.const import *
import shared.state as state

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
    if not state.cube_items:
        cube_items = build_cube_wireframe(AXIS_LEN)
    for it in cube_items:
        try:
            state.VIEW.addItem(it)
        except Exception:
            pass

def hide_cube():
    for it in cube_items:
        try:
            state.VIEW.removeItem(it)
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