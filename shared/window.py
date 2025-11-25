from PySide6.QtWidgets import QApplication, QMainWindow

def center_on_screen(win: QMainWindow):
    """Center main window on primary available geometry (macOS-friendly)."""
    try:
        screen = win.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        x = geo.x() + (geo.width() - win.width()) // 2
        y = geo.y() + (geo.height() - win.height()) // 2
        win.move(x, y)
    except Exception:
        try:
            fg = win.frameGeometry()
            center = QApplication.primaryScreen().availableGeometry().center()
            fg.moveCenter(center)
            win.move(fg.topLeft())
        except Exception:
            pass