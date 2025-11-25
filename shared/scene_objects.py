from pyqtgraph.opengl import GLScatterPlotItem
import numpy as np

from shared.const import *
import shared.state as state
import shared.overlays_labels as overlays_labels
import shared.overlays_utils as overlay_utils
import shared.tokens as token
import shared.log as log

def ensure_point_item(pid: str):
    """Create or return a GLScatterPlotItem used to render a point sprite."""
    from pyqtgraph.opengl import GLScatterPlotItem
    if pid in state.placed_points:
        return state.placed_points[pid][0]
    item = GLScatterPlotItem(pos=np.array([[0.0, 0.0, 0.0]]), size=POINT_SIZE, color=POINT_COLOR, pxMode=True)
    state.VIEW.addItem(item)
    state.placed_points[pid] = (item, [0.0, 0.0, 0.0])
    return item

def set_point_position(pid: str, coords):
    """Set world position of a point sprite and update overlays/lines."""
    item = ensure_point_item(pid)
    x, y, z = map(float, coords)
    pos = np.array([[x, y, z]], dtype=float)
    item.setData(pos=pos)
    state.placed_points[pid] = (item, [x, y, z])
    overlays_labels.update_point_label(pid)
    overlays_labels.update_image_label(pid)
    overlay_utils.update_point_color(pid)
    
def reset_all_points():
    """Remove all point sprites, overlays and reset token states."""
    for pid, (it, _) in list(state.placed_points.items()):
        try:
            state.VIEW.removeItem(it)
        except Exception:
            pass
    state.placed_points.clear()
    for cat, line in list(state.pair_lines.items()):
        try:
            state.VIEW.removeItem(line)
        except Exception:
            pass
    state.pair_lines.clear()
    for pid, lab in list(state.point_labels.items()):
        try:
            lab.hide()
            lab.deleteLater()
        except Exception:
            pass
    state.point_labels.clear()
    for pid, lab in list(state.image_labels.items()):
        try:
            lab.hide()
            lab.deleteLater()
        except Exception:
            pass
    state.image_labels.clear()
    overlay_utils._hide_hover_preview()
    for t in state.point_tokens:
        t.setProperty('placed', False)
        t.setStyleSheet(token.token_style_mode('active'))
        t.setEnabled(True)
        t.show()
    state.placement_phase = 1
    token.update_token_states()
    token.update_submit_state()
    log.log_session_event("Reset all placed points")
