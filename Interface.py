from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QSizePolicy, QGraphicsDropShadowEffect, QProgressBar
from PySide6.QtGui import QGuiApplication, QColor, QIcon, QCursor
from PySide6.QtCore import Qt, QTimer, QSize, QEvent

from PySide6.QtCore import QUrl
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget

import subprocess
import sys
import os 
from typing import Optional

class Interface(QWidget):
    
    """ GLOBAL VARIABLES """
    DEFAULT_HEADING_FONT = "Sans Serif"
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D inverse MDS (Prototype)")

        # MAIN WINDOW
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(20, 100, 20, 20)
        self.setLayout(layout)
        self.setStyleSheet("background-color: white;")
        self.setFixedSize(1200, 700)
        self.center_on_screen()

        self._sim_proc: Optional[subprocess.Popen] = None
        
        ##############################################################
        ########################## ICONS #############################
        ##############################################################
        
        self.light_btn = QPushButton("☀", self)  # U+2600 Sun symbol
        self.light_btn.setStyleSheet("font-size: 22px; border: none; background: transparent; color: black;")
        self.light_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        # self.dark_btn = QPushButton("☾", self)  # U+263E Moon symbol
        # self.dark_btn.setStyleSheet("font-size: 20px; border: none; background: transparent; color: black;")
        
        self.github_btn = QPushButton(self)
        self.github_btn.setIcon(QIcon(("assets/icons/github.png")))
        self.github_btn.setIconSize(QSize(32, 32))
        self.github_btn.setStyleSheet("border: none; background: transparent;")
        self.light_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        
        ##############################################################
        ####################### TEXT ELEMENTS ########################
        ##############################################################
        
        # HEADER
        self.title_label = QLabel("3D Inverse-MDS for Embedding data", self)
        self.title_label.setStyleSheet(
            f"""
            font-size: 24px;
            font-family: '{self.DEFAULT_HEADING_FONT}';
            font-weight: 300;
            color: #333333;
            """
        )

        self.author_label = QLabel("By Noah Kogge", self)
        self.author_label.setStyleSheet(
            f"""
            font-size: 15px;
            font-family: '{self.DEFAULT_HEADING_FONT}';
            font-weight: 200;
            color: #333333;
            """
        )
        
        # ABOUT
        self.about_label = QLabel("About:", self)
        self.about_label.setStyleSheet(
            f"""
            font-size: 18px;
            font-family: '{self.DEFAULT_HEADING_FONT}';
            font-weight: 300;
            color: #333333;
            """
        )
        
        # LOGS
        self.log_label = QLabel("Logs:", self)
        self.log_label.setStyleSheet(
            f"""
            font-size: 18px;
            font-family: '{self.DEFAULT_HEADING_FONT}';
            font-weight: 300;
            color: #333333;
            """
        )
        
        # TUTORIALS
        self.tutorial_label = QLabel("Tutorials:", self)
        self.tutorial_label.setStyleSheet(
            f"""
            font-size: 18px;
            font-family: '{self.DEFAULT_HEADING_FONT}';
            font-weight: 300;
            color: #333333;
            """
        )
        # Toggle-Button links neben "Tutorials:" zum Ein-/Ausklappen
        self.tutorial_toggle_btn = QPushButton(self)
        # Initialzustand: Bereich ist offen → nach unten zeigendes Dreieck
        self.tutorial_toggle_btn.setIcon(QIcon("assets/svg/triangle-down.svg"))
        self.tutorial_toggle_btn.setIconSize(QSize(13, 13))
        self.tutorial_toggle_btn.setStyleSheet(
            "border: none; background: transparent;"
        )
        self.tutorial_toggle_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.tutorial_toggle_btn.clicked.connect(self.toggle_tutorials)
    
        self.tutorial_description1 = QLabel("Beginner", self)
        self.tutorial_description1.setStyleSheet(  
            f"""
            font-size: 14px;
            font-family: '{self.DEFAULT_HEADING_FONT}';
            font-weight: 200;
            color: Black;
            """
        )
        
        self.tutorial_description2 = QLabel("Intermediate", self)
        self.tutorial_description2.setStyleSheet(  
            f"""
            font-size: 14px;
            font-family: '{self.DEFAULT_HEADING_FONT}';
            font-weight: 200;
            color: Black;
            """
        )
        
        self.tutorial_description3 = QLabel("Professional", self)
        self.tutorial_description3.setStyleSheet(  
            f"""
            font-size: 14px;
            font-family: '{self.DEFAULT_HEADING_FONT}';
            font-weight: 200;
            color: Black;
            """
        )
        
        self.hint = QLabel("Hint!: Hover over the container to preview the video", self)
        self.hint.setStyleSheet(  
            f"""
            font-size: 10px;
            font-family: '{self.DEFAULT_HEADING_FONT}';
            font-weight: 100;
            color: Black;
            """
        )
        # Gruppe der Widgets, die gemeinsam ein-/ausgeblendet werden (Container + Überschriften + Hint)
        self._tutorial_group_widgets = [
            # wird später mit self.main_container gefüllt (nach der Container-Erstellung)
        ]
            
        ##############################################################
        ################### CONTAINER ELEMENTS #######################
        ##############################################################

        # MAIN CONTAINER
        main_container, main_container_layout = self.create_container(
            width=1000,
            height=350,
            orientation="h",
            stylesheet="""
            QWidget {
                background-color: white;
                border-radius: 8px;
            }
            """
        )
        self.main_container = main_container
        # Jetzt, da main_container existiert, die Gruppe vervollständigen
        if not hasattr(self, "_tutorial_group_widgets"):
            self._tutorial_group_widgets = []
        self._tutorial_group_widgets = [
            self.main_container,
            self.tutorial_description1,
            self.tutorial_description2,
            self.tutorial_description3,
            self.hint,
        ]
        main_container_layout.setContentsMargins(6, 0, 6, 6)
        main_container_layout.setSpacing(20)

        # -------------------- TUTORIAL 1 --------------------
        tutorial1_container, tutorial1_layout = self.create_container(
            width=0, height=0, orientation="v",
            stylesheet="""
            #tutorial1Card {
                background-color: white;
                border: 1px solid lightgrey;
                border-radius: 8px;
            }
            """,
        )
        tutorial1_container.setObjectName("tutorial1Card")
        tutorial1_container.setContentsMargins(5,5,5,5)

        # Top-Half: eigener Stil, feste Höhe
        top_half_container, top_half_layout = self.create_container(
            width=0, height=0, orientation="v",
            stylesheet="""
            #tutorial1Top {
                background-color: black;
                border: 1px solid lightgrey;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                border-bottom-left-radius: 8px;
                border-bottom-right-radius: 8px;
            }
            """,
        )
        top_half_container.setObjectName("tutorial1Top")
        top_half_layout.setContentsMargins(4, 4, 4, 4)
        top_half_container.setFixedHeight(190)
        top_half_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.video_player1, self.video_widget1, self.video_audio1 = self.load_video_sequence(
            ["videos/xyz.mov", "videos/zoom.mov"], top_half_container, volume=0.8, loop=True   
        )

        tutorial1_container._video_player = self.video_player1  # type: ignore[attr-defined]
        tutorial1_container.installEventFilter(self)
        if self.video_player1 is not None:
            self.video_player1.pause()
            self.video_player1.setPosition(0)

        tutorial1_layout.setContentsMargins(0, 0, 0, 0)
        tutorial1_layout.setSpacing(8)  # leichter Abstand zwischen Top-Half und Text
        tutorial1_layout.addWidget(top_half_container)
        tutorial1_layout.addStretch(1)

        # Caption (Text unten), ohne Border, transparent
        self.tutorial_description1_caption = QLabel(
            self.load_text_file("text/html_wrapper/beginner.html"), self
        )
        self.tutorial_description1_caption.setWordWrap(True)
        self.tutorial_description1_caption.setTextFormat(Qt.TextFormat.RichText)
        self.tutorial_description1_caption.setStyleSheet(
            f"""
            font-size: 12px;
            font-family: '{self.DEFAULT_HEADING_FONT}';
            font-weight: 200;
            color: #666666;
            border: none;
            background: transparent;
            padding: 10px 10px 10px 10px; 
            """
        )
        tutorial1_layout.addWidget(self.tutorial_description1_caption)
        self.apply_box_shadow(tutorial1_container)

        # -------------------- TUTORIAL 2 --------------------
        tutorial2_container, tutorial2_layout = self.create_container(
            width=0,
            height=0,
            orientation="v",
            stylesheet="""
            #tutorial2Card {
                background-color: white;
                border: 1px solid lightgrey;
                border-radius: 8px;
            }
            """
        )
        tutorial2_container.setObjectName("tutorial2Card")
        tutorial2_container.setContentsMargins(5,5,5,5)

        top_half_container_2, top_half_layout_2 = self.create_container(
            width=0,
            height=0,
            orientation="v",
            stylesheet="""
            #tutorial2Top {
                background-color: black;
                border: 1px solid lightgrey;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                border-bottom-left-radius: 8px;
                border-bottom-right-radius: 8px;
            }
            """
        )
        top_half_container_2.setObjectName("tutorial2Top")
        top_half_layout_2.setContentsMargins(4, 4, 4, 4)
        top_half_container_2.setFixedHeight(190)
        top_half_container_2.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        self.video_player2, self.video_widget2, self.video_audio2 = self.load_video_sequence(
            ["videos/Controlpannels.mov", "videos/snapbutton.mov"], top_half_container_2, volume=0.8, loop=True
        )

        tutorial2_container._video_player = self.video_player2  # type: ignore[attr-defined]
        tutorial2_container.installEventFilter(self)
        if self.video_player2 is not None:
            self.video_player2.pause()
            self.video_player2.setPosition(0)

        # Wichtig: bündig wie rechts → keine Margins/Spacing
        tutorial2_layout.setContentsMargins(0, 0, 0, 0)
        tutorial2_layout.setSpacing(0)
        tutorial2_layout.addWidget(top_half_container_2)
        tutorial2_layout.addStretch(1)
        tutorial2_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Caption wie links → Padding, kein Border
        self.tutorial_description2_caption = QLabel(
            self.load_text_file("text/html_wrapper/intermediate.html"), self
        )
        self.tutorial_description2_caption.setWordWrap(True)
        self.tutorial_description1_caption.setTextFormat(Qt.TextFormat.RichText)
        self.tutorial_description2_caption.setStyleSheet(
            f"""
            font-size: 12px;
            font-family: '{self.DEFAULT_HEADING_FONT}';
            font-weight: 200;
            color: #666666;
            border: none;
            background: transparent;
            padding: 10px 10px 10px 10px; 
            """
        )
        tutorial2_layout.addWidget(self.tutorial_description2_caption)
        self.apply_box_shadow(tutorial2_container)

        # -------------------- TUTORIAL 3 --------------------
        tutorial3_container, tutorial3_layout = self.create_container(
            width=0,
            height=0,
            orientation="v",
            stylesheet="""
            #tutorial3Card {
                background-color: white;
                border: 1px solid lightgrey;
                border-radius: 8px;
            }
            """
        )
        tutorial3_container.setObjectName("tutorial3Card")
        tutorial3_container.setContentsMargins(5,5,5,5)

        top_half_container_3, top_half_layout_3 = self.create_container(
            width=0,
            height=0,
            orientation="v",
            stylesheet="""
            #tutorial3Top {
                background-color: black;
                border: 1px solid lightgrey;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                border-bottom-left-radius: 8px;
                border-bottom-right-radius: 8px;
            }
            """
        )
        top_half_container_3.setObjectName("tutorial3Top")
        top_half_layout_3.setContentsMargins(4, 4, 4, 4)
        top_half_container_3.setFixedHeight(190)
        top_half_container_3.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.video_player3, self.video_widget3, self.video_audio3 = self.load_video_sequence(
            ["videos/imagetokens.mov", "videos/aligned.mov"], top_half_container_3, volume=0.8, loop=True
        )

        tutorial3_container._video_player = self.video_player3  # type: ignore[attr-defined]
        tutorial3_container.installEventFilter(self)
        if self.video_player3 is not None:
            self.video_player3.pause()
            self.video_player3.setPosition(0)

        # Bündig wie rechts
        tutorial3_layout.setContentsMargins(0, 0, 0, 0)
        tutorial3_layout.setSpacing(0)
        tutorial3_layout.addWidget(top_half_container_3)
        tutorial3_layout.addStretch(1)
        tutorial3_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Caption wie links
        self.tutorial_description3_caption = QLabel(
            self.load_text_file("text/html_wrapper/professional.html"), self
        )
        
        self.tutorial_description3_caption.setWordWrap(True)
        self.tutorial_description1_caption.setTextFormat(Qt.TextFormat.RichText)
        self.tutorial_description3_caption.setStyleSheet(
            f"""
            font-size: 12px;
            font-family: '{self.DEFAULT_HEADING_FONT}';
            font-weight: 200;
            color: #666666;
            border: none;
            background: transparent;
            padding: 10px 10px 10px 10px; 
            """
        )
        tutorial3_layout.addWidget(self.tutorial_description3_caption)
        self.apply_box_shadow(tutorial3_container)
        
        self.start_btn = QPushButton("Start Experiment", self)
        self.start_btn.setStyleSheet(
            """
            QPushButton {
                font-size: 12px; 
                font-weight: 200; 
                color: black; 
                font-familiy: "Sans Serif";
                background: white; border: none; border-radius: 6px; padding: 10px 16px;
            }
            QPushButton:pressed { background: lightgray; }
            """
        )
        self.start_btn.setFixedWidth(200)
        self.start_btn.clicked.connect(self.start_simulation)
        self.apply_box_shadow(self.start_btn)

        self._sim_monitor = QTimer(self)
        self._sim_monitor.setInterval(500)
        self._sim_monitor.timeout.connect(self._check_simulation_alive)
        self._sim_monitor.start()

        # Add the tutorials into the main_container
        main_container_layout.addWidget(tutorial1_container, stretch=1)
        main_container_layout.addWidget(tutorial2_container, stretch=1)
        main_container_layout.addWidget(tutorial3_container, stretch=1)

        layout.addWidget(main_container)
        
        self.adjust_position(self.title_label, y=60)
        self.adjust_position(self.author_label, y=90)
        
        self.adjust_position(self.about_label, x=110, y=130, defer=True)
        self.adjust_position(self.log_label, x=110, y=160, defer=True)
        self.adjust_position(self.tutorial_label, x=110, y=190, defer=True)
        # Toggle-Button leicht links vom "Tutorials:"-Label
        self.adjust_position(self.tutorial_toggle_btn, x=85, y=195, defer=True)
        self.adjust_position(self.hint, y= 565)
        
        self.adjust_position(self.tutorial_description1,x=125, y=415, defer=True)        
        self.adjust_position(self.tutorial_description2,x=460, y=415, defer=True)
        self.adjust_position(self.tutorial_description3,x=795, y=415, defer=True)

        self.adjust_position(self.light_btn, x=110, y=60, defer=True)
        self.adjust_position(self.github_btn, x=1060, y=60, defer=True)
        self.adjust_position(self.start_btn, y=600)
    
    ##############################################################
    ################### HELPER FUNCTIONS #########################
    ##############################################################
    
    def load_text_file(self, relative_path: str) -> str:
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(base_dir, relative_path)
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"[Fehler beim Laden: {e}]"
    
    def create_container(self, width=0, height=0, stylesheet="", orientation: str = "v"):
        container = QWidget()
        container_layout = QVBoxLayout() if orientation.lower().startswith("v") else QHBoxLayout()
        container.setLayout(container_layout)
        if width > 0 and height > 0:
            container.setFixedSize(width, height)
        else:
            if width > 0:
                container.setFixedWidth(width)
            else:
                sp = container.sizePolicy()
                sp.setHorizontalPolicy(QSizePolicy.Policy.Expanding)
                container.setSizePolicy(sp)
            if height > 0:
                container.setFixedHeight(height)
        container.setStyleSheet(stylesheet)
        container.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        container.setMouseTracking(True)
        return container, container_layout


    def center_on_screen(self):
        screen = QGuiApplication.primaryScreen()
        screen_geometry = screen.geometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)
        
    def adjust_position(self, label=None, container: QWidget | None = None, x: int | None = None, y: int | None = None, defer: bool = False):
        if label is None:
            return

        def _do_move():
            label.adjustSize()

            # Determine X
            if x is None:
                xpos = (self.width() - label.width()) // 2
            elif x == 0 and container is not None:
                xpos = container.geometry().left()
            else:
                xpos = x

            # Determine Y
            if y is None:
                ypos = label.y()
            elif y == 0 and container is not None:
                ypos = container.geometry().top() - label.height() - 5  # 10px über container
            else:
                ypos = y

            label.move(xpos, ypos)
            label.raise_()

        if defer:
            QTimer.singleShot(0, _do_move)
        else:
            _do_move()
    
    def apply_box_shadow(self, widget, blur_radius=15, x_offset=1, y_offset=2, color=QColor(0, 0, 0, 120)):
        # Create effect with the widget as parent so lifetime is managed by Qt
        shadow = QGraphicsDropShadowEffect(widget)
        shadow.setBlurRadius(blur_radius)
        shadow.setOffset(x_offset, y_offset)
        shadow.setColor(color)
        widget.setGraphicsEffect(shadow)

        # Hover handlers that only touch the effect if it is still a drop shadow
        def on_enter(event):
            eff = widget.graphicsEffect()
            if isinstance(eff, QGraphicsDropShadowEffect):
                eff.setOffset(x_offset, max(y_offset + 2, y_offset))
                eff.setBlurRadius(max(blur_radius + 4, blur_radius))
            QWidget.enterEvent(widget, event) if hasattr(QWidget, 'enterEvent') else None

        def on_leave(event):
            eff = widget.graphicsEffect()
            if isinstance(eff, QGraphicsDropShadowEffect):
                eff.setOffset(x_offset, y_offset)
                eff.setBlurRadius(blur_radius)
            QWidget.leaveEvent(widget, event) if hasattr(QWidget, 'leaveEvent') else None

        widget.enterEvent = on_enter
        widget.leaveEvent = on_leave

    def eventFilter(self, obj, ev):
        et = ev.type()
        if et == QEvent.Type.Enter:
            player = getattr(obj, "_video_player", None)
            if player is not None:
                try:
                    player.play()
                except Exception:
                    pass
        elif et == QEvent.Type.Leave:
            player = getattr(obj, "_video_player", None)
            if player is not None:
                try:
                    srcs = getattr(player, "_seq_sources", [])
                    if srcs:
                        player._seq_index = 0  # type: ignore[attr-defined]
                        player.setSource(srcs[0])
                    player.setPosition(0)
                    player.pause()
                except Exception:
                    pass
        return super().eventFilter(obj, ev)

    def _check_simulation_alive(self):
        try:
            if self._sim_proc is not None:
                if self._sim_proc.poll() is None:
                    # still running
                    self.start_btn.setEnabled(False)
                    self.start_btn.setText("Experiment running…")
                    return
                else:
                    # finished
                    self._sim_proc = None
            # no process: enable button
            self.start_btn.setEnabled(True)
            self.start_btn.setText("Start Experiment")
        except Exception:
            # Fail-open: don't block the UI
            self.start_btn.setEnabled(True)
            self.start_btn.setText("Start Experiment")


    def start_simulation(self):
        # Block if a simulation is already running
        if self._sim_proc is not None and self._sim_proc.poll() is None:
            self.setWindowTitle("Simulation already running")
            return
        # Resolve path to the simulation script (3-dimensional-mds.py) next to this Interface.py
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            sim_path = os.path.join(base_dir, "experiment.py")
            if not os.path.isfile(sim_path):
                # Fallback: try alternative common name
                alt = os.path.join(base_dir, "three_dimensional_mds.py")
                if os.path.isfile(alt):
                    sim_path = alt
                else:
                    # Give a friendly message in the UI title if not found
                    self.setWindowTitle("Simulation script not found")
                    return
            # Launch in a separate process to avoid multiple QApplication instances
            python_exe = sys.executable or "python3"
            self._sim_proc = subprocess.Popen([python_exe, sim_path], cwd=base_dir)
            self.start_btn.setEnabled(False)
            self.start_btn.setText("Experiment running…")
        except Exception as e:
            # Minimal feedback via window title for now
            self.setWindowTitle(f"Launch failed: {e}")


    def closeEvent(self, event):
        proc = self._sim_proc
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except Exception:
                        proc.kill()
            except Exception:
                pass
            finally:
                self._sim_proc = None
        for attr in ("video_player1", "video_player2", "video_player3"):
            player = getattr(self, attr, None)
            if player is not None:
                try:
                    player.stop()
                except Exception:
                    pass
        super().closeEvent(event)
        
    def load_video_sequence(self, inputpaths, container, volume: float = 0.8, loop: bool = False):
        try:
            # Resolve base dir and all paths
            base_dir = os.path.dirname(os.path.abspath(__file__))
            sources = []
            for p in inputpaths:
                vp = p if os.path.isabs(p) else os.path.join(base_dir, p)
                sources.append(QUrl.fromLocalFile(vp))

            # Ensure container has a layout
            cont_layout = container.layout()
            if cont_layout is None:
                cont_layout = QVBoxLayout()
                container.setLayout(cont_layout)

            # Video widget
            vw = QVideoWidget(container)
            vw.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            try:
                vw.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
            except Exception:
                pass

            # Audio + player
            audio = QAudioOutput(self)
            volume = max(0.0, min(1.0, float(volume)))
            try:
                audio.setVolume(volume)
            except Exception:
                pass

            player = QMediaPlayer(self)
            player.setAudioOutput(audio)
            player.setVideoOutput(vw)

            # Sequence state on the player object
            player._seq_sources = sources  # type: ignore[attr-defined]
            player._seq_index = 0          # type: ignore[attr-defined]
            player._seq_loop = bool(loop)  # type: ignore[attr-defined]

            # Start first
            if sources:
                player.setSource(sources[0])

            def _on_status_changed(status):
                try:
                    # When current finishes, advance to next
                    if status == player.MediaStatus.EndOfMedia:
                        idx = getattr(player, "_seq_index", 0) + 1
                        srcs = getattr(player, "_seq_sources", [])
                        if idx >= len(srcs):
                            if getattr(player, "_seq_loop", False) and srcs:
                                idx = 0
                            else:
                                return  # stop at the end
                        player._seq_index = idx  # type: ignore[attr-defined]
                        player.setSource(srcs[idx])
                        player.play()
                except Exception:
                    pass

            player.mediaStatusChanged.connect(_on_status_changed)

            cont_layout.addWidget(vw)

            # Tiny progress bar under the video indicating current clip progress
            pb = QProgressBar(container)
            pb.setTextVisible(False)
            pb.setFixedHeight(2)
            pb.setRange(0, 0)  # indeterminate until duration known
            pb.setStyleSheet(
                """
                QProgressBar { border: none; background: rgba(255,255, 255); border-radius: 3px; }
                QProgressBar::chunk { background-color: white; border-radius: 3px; }
                """
            )
            cont_layout.addWidget(pb)

            # Wire duration/position updates to progress bar
            def _on_duration_changed(d):
                try:
                    d = int(d)
                except Exception:
                    d = 0
                if d <= 0:
                    pb.setRange(0, 0)
                    pb.setValue(0)
                else:
                    pb.setRange(0, d)
                    pb.setValue(0)

            def _on_position_changed(p):
                try:
                    p = int(p)
                except Exception:
                    return
                if pb.maximum() > 0:
                    pb.setValue(min(p, pb.maximum()))

            player.durationChanged.connect(_on_duration_changed)
            player.positionChanged.connect(_on_position_changed)

            # When we switch to the next source in the sequence, duration will update; also reset explicitly
            def _on_media_changed(_):
                pb.setValue(0)
            try:
                player.sourceChanged.connect(_on_media_changed)
            except Exception:
                pass

            player.play()
            return player, vw, audio
        except Exception as e:
            self.setWindowTitle(f"Video sequence failed: {e}")
            return None, None, None
        
    def toggle_tutorials(self):
        try:
            # Sichtbarkeit des Haupt-Containers bestimmt aktuellen Zustand
            is_open = getattr(self, "main_container", None) is not None and self.main_container.isVisible()
            if is_open:
                # → zuklappen
                for w in getattr(self, "_tutorial_group_widgets", []):
                    if w is not None:
                        w.hide()
                if getattr(self, "tutorial_toggle_btn", None) is not None:
                    # zugeklappt: Dreieck nach oben (oder rechts, falls du so ein Icon nutzt)
                    self.tutorial_toggle_btn.setIcon(QIcon("assets/svg/triangle-up.svg"))
            else:
                # → aufklappen
                for w in getattr(self, "_tutorial_group_widgets", []):
                    if w is not None:
                        w.show()
                if getattr(self, "tutorial_toggle_btn", None) is not None:
                    # offen: Dreieck nach unten
                    self.tutorial_toggle_btn.setIcon(QIcon("assets/svg/triangle-down.svg"))
        except Exception:
            # Fallback: Wenn etwas schiefgeht, nur den Hauptcontainer togglen und Icon entsprechend setzen
            if getattr(self, "main_container", None) is not None:
                self.main_container.setVisible(not self.main_container.isVisible())
            if getattr(self, "tutorial_toggle_btn", None) is not None and getattr(self, "main_container", None) is not None:
                self.tutorial_toggle_btn.setIcon(
                    QIcon("assets/svg/triangle-down.svg") if self.main_container.isVisible() else QIcon("assets/svg/triangle-up.svg")
                )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Interface()
    win.show()
    sys.exit(app.exec())
