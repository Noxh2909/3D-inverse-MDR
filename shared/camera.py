
from PySide6.QtGui import QVector3D

from shared.const import *
import shared.state as state
import shared.log as log

def _cube_center():
    L = float(AXIS_LEN)
    return QVector3D(L/2.0, L/2.0, L/2.0)

def _fit_distance_for_extent(extent: float, margin: float = 2) -> float:
    """Compute camera distance so a given extent is visible respecting FOV."""
    w = max(1, state.VIEW.width())
    h = max(1, state.VIEW.height())
    vfov_deg = float(state.VIEW.opts.get('fov', 60))
    vfov = np.deg2rad(vfov_deg)
    aspect = w / h
    hfov = 2.0 * np.arctan(np.tan(vfov/2.0) * aspect)
    half = extent / 2.0
    d_v = half / np.tan(vfov/2.0)
    d_h = half / np.tan(hfov/2.0)
    return float(max(d_v, d_h) * margin)

def _set_view_fitted(elevation=0, azimuth=0, zoom=1.0):
    """Set camera to state.VIEW cube center with optional elevation/azimuth and zoom multiplier."""
    
    extent = float(AXIS_LEN)
    dist = _fit_distance_for_extent(extent)
    dist *= float(zoom)
    state.VIEW.opts['center'] = _cube_center()
    state.VIEW.setCameraPosition(distance=dist, elevation=elevation, azimuth=azimuth)
    
def set_view_xy():
    """Switch to orthographic state.VIEW focused on YZ plane (Z-Y)."""
    global current_plane
    current_plane = 'yz'
    state.VIEW.opts['ortho'] = True
    log.log_session_event("Rotated view XY")
    _set_view_fitted(elevation=0, azimuth=90, zoom=0.03)

def set_view_yz():
    """Switch to orthographic state.VIEW focused on XZ plane (Z-X)."""
    global current_plane
    current_plane = 'xz'
    state.VIEW.opts['ortho'] = True
    log.log_session_event("Rotated view XZ")
    _set_view_fitted(elevation=0, azimuth=0, zoom=0.03)
