from PySide6.QtGui import QVector3D
import numpy as np
import shared.state as state
import shared.camera as camera
from shared.const import *


def project_point(p):
    """Project a world 3D point to 2D view pixel coordinates (x, y)."""
    view = state.VIEW
    if view is None:
        return None
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
    view = state.VIEW
    if view is None:
        return None
    try:
        p = view.cameraPosition()
        return np.array([float(p.x()), float(p.y()), float(p.z())], dtype=float)
    except Exception:
        return None

def _camera_forward_vec3():
    """Compute normalized forward vector from camera to view center, or None."""
    view = state.VIEW
    if view is None:
        return None
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
    """Return False if a world point is behind camera or projected off-screen."""
    view = state.VIEW
    if view is None:
        return False
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
    """Return (near_world_qvec, far_world_qvec) for a screen pixel."""
    view = state.VIEW
    if view is None:
        return None, None
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
    """Intersect ray (p0 -> p1) with plane."""
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
    """Return (t, (x,y,z))."""
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

def _fit_distance_for_extent(extent: float, margin: float = 2) -> float:
    """Compute camera distance needed for a given extent."""
    view = state.VIEW
    if view is None:
        return extent
    try:
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
    except Exception:
        return extent

def _screen_perp_tick_on_z(z_val: float, screen_len_px: int = 12):
    """Compute two world points for a small screen-perpendicular Z tick."""
    view = state.VIEW
    if view is None:
        return None

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

def choose_plane_and_hit(px: int, py: int):
    """Choose the best plane hit (yz/xz/xy) for the given screen pixel and return (plane, hit)."""
    p0, p1 = screen_to_world_ray(px, py)
    if p0 is None or p1 is None:
        return None
    ortho = bool(state.VIEW.opts.get('ortho', False))
    if ortho:
        if camera.current_plane in ('yz', 'xz'):
            candidates = []
            for pl in ('yz', 'xz'):
                r = intersect_with_plane_t(p0, p1, pl)
                if r is not None:
                    t, hit = r
                    if pl == camera.current_plane:
                        t = t - 1e-9
                    candidates.append((t, pl, hit))
            if not candidates:
                return None
            candidates.sort(key=lambda x: x[0])
            _, pl, hit = candidates[0]
            return pl, hit
        else:
            res = intersect_with_plane_t(p0, p1, camera.current_plane)
            if res is None:
                return None
            _, hit = res
            return camera.current_plane, hit
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