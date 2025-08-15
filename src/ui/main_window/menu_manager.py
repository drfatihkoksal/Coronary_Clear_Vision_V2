"""Menu management for the main window"""

from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QMenu
import logging

logger = logging.getLogger(__name__)


class MenuManager:
    """Handles all menu creation and management for the main window"""

    def __init__(self, main_window):
        """
        Initialize menu manager

        Args:
            main_window: Reference to the main window
        """
        self.main_window = main_window
        self.recent_files_menu = None

    def create_menus(self):
        """Create all application menus"""
        menubar = self.main_window.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")
        self._create_file_menu(file_menu)

        # View menu
        view_menu = menubar.addMenu("&View")
        self._create_view_menu(view_menu)

        # Analysis menu
        analysis_menu = menubar.addMenu("&Analysis")
        self._create_analysis_menu(analysis_menu)

        # Window menu
        window_menu = menubar.addMenu("&Window")
        self._create_window_menu(window_menu)

        # Help menu
        help_menu = menubar.addMenu("&Help")
        self._create_help_menu(help_menu)

        # Set menu bar style (VSCode dark theme)
        menubar.setStyleSheet(
            """
            QMenuBar {
                background-color: #2d2d30;
                color: #cccccc;
                border-bottom: 1px solid #3e3e42;
                padding: 2px;
                font-size: 13px;
            }
            QMenuBar::item:selected {
                background-color: #094771;
                color: white;
            }
            QMenuBar::item:pressed {
                background-color: #007acc;
                color: white;
            }
            QMenu {
                background-color: #252526;
                color: #cccccc;
                border: 1px solid #3e3e42;
                padding: 4px;
                font-size: 13px;
            }
            QMenu::item {
                padding: 6px 30px 6px 20px;
                border-radius: 2px;
            }
            QMenu::item:selected {
                background-color: #094771;
                color: white;
            }
            QMenu::separator {
                height: 1px;
                background: #3e3e42;
                margin: 4px 10px;
            }
            QMenu::indicator {
                width: 13px;
                height: 13px;
                left: 6px;
            }
            QMenu::indicator:checked {
                image: none;
                background-color: #007acc;
                border: 1px solid #cccccc;
                border-radius: 2px;
            }
        """
        )

    def _create_file_menu(self, file_menu):
        """Create file menu items"""
        # Open DICOM
        open_action = QAction("&Open DICOM File...", self.main_window)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.setStatusTip("Open a DICOM file")
        open_action.triggered.connect(self.main_window.open_dicom)
        file_menu.addAction(open_action)

        # Open DICOM Folder
        open_folder_action = QAction("Open DICOM &Folder...", self.main_window)
        open_folder_action.setShortcut(QKeySequence("Ctrl+Shift+O"))
        open_folder_action.setStatusTip("Open folder containing DICOM files")
        open_folder_action.triggered.connect(self.main_window.open_dicom_folder)
        file_menu.addAction(open_folder_action)

        # Open Default Folder
        open_default_action = QAction("Open &Default DICOM Folder", self.main_window)
        open_default_action.setShortcut(QKeySequence("Ctrl+D"))
        open_default_action.setStatusTip("Open default DICOM folder")
        open_default_action.triggered.connect(self.main_window.open_default_dicom_folder)
        file_menu.addAction(open_default_action)

        file_menu.addSeparator()

        # Recent files submenu
        self.recent_files_menu = QMenu("Recent Files", self.main_window)
        file_menu.addMenu(self.recent_files_menu)
        self.update_recent_menu()

        file_menu.addSeparator()

        # Save current image
        save_image_action = QAction("&Save Current Image...", self.main_window)
        save_image_action.setShortcut(QKeySequence("Ctrl+S"))
        save_image_action.setStatusTip("Save current frame as image")
        save_image_action.triggered.connect(self.main_window.save_current_image)
        file_menu.addAction(save_image_action)

        # Export video
        export_video_action = QAction("&Export Video...", self.main_window)
        export_video_action.setShortcut(QKeySequence("Ctrl+E"))
        export_video_action.setStatusTip("Export frames as video")
        export_video_action.triggered.connect(self.main_window.export_video)
        file_menu.addAction(export_video_action)

        file_menu.addSeparator()

        # Exit
        exit_action = QAction("E&xit", self.main_window)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.setStatusTip("Exit application")
        exit_action.triggered.connect(self.main_window.close)
        file_menu.addAction(exit_action)

    def _create_view_menu(self, view_menu):
        """Create view menu items"""
        # Zoom actions
        zoom_fit_action = QAction("Fit to &Window", self.main_window)
        zoom_fit_action.setShortcut(QKeySequence("Ctrl+0"))
        zoom_fit_action.triggered.connect(lambda: self.main_window.viewer_widget.fit_to_window())
        view_menu.addAction(zoom_fit_action)

        zoom_reset_action = QAction("&Reset Zoom", self.main_window)
        zoom_reset_action.setShortcut(QKeySequence("Ctrl+R"))
        zoom_reset_action.triggered.connect(lambda: self.main_window.viewer_widget.reset_zoom())
        view_menu.addAction(zoom_reset_action)

        view_menu.addSeparator()

        # Toggle overlays
        toggle_points_action = QAction("Show &Points", self.main_window)
        toggle_points_action.setCheckable(True)
        toggle_points_action.setChecked(True)
        toggle_points_action.triggered.connect(self.main_window.toggle_points_visibility)
        view_menu.addAction(toggle_points_action)

        toggle_seg_action = QAction("Show &Segmentation", self.main_window)
        toggle_seg_action.setCheckable(True)
        toggle_seg_action.setChecked(True)
        toggle_seg_action.triggered.connect(self.main_window.toggle_segmentation_visibility)
        view_menu.addAction(toggle_seg_action)

        toggle_qca_action = QAction("Show &QCA Results", self.main_window)
        toggle_qca_action.setCheckable(True)
        toggle_qca_action.setChecked(True)
        toggle_qca_action.triggered.connect(self.main_window.toggle_qca_visibility)
        view_menu.addAction(toggle_qca_action)

        view_menu.addSeparator()

        # Window/Level presets submenu
        self.main_window.wl_menu = view_menu.addMenu("Window/Level Presets")
        self.main_window.update_wl_menu()

        # Store references for external access
        self.main_window.toggle_points_action = toggle_points_action
        self.main_window.toggle_seg_action = toggle_seg_action
        self.main_window.toggle_qca_action = toggle_qca_action

    def _create_analysis_menu(self, analysis_menu):
        """Create analysis menu items"""
        # Calibration
        calibration_action = QAction("&Calibration", self.main_window)
        calibration_action.setShortcut(QKeySequence("C"))
        calibration_action.setStatusTip("Calibrate measurements")
        calibration_action.triggered.connect(self.main_window.start_calibration)
        analysis_menu.addAction(calibration_action)

        analysis_menu.addSeparator()

        # Tracking
        track_all_action = QAction("&Track All Frames", self.main_window)
        track_all_action.setShortcut(QKeySequence("Ctrl+T"))
        track_all_action.setStatusTip("Track points across all frames")
        track_all_action.triggered.connect(self.main_window.track_all_frames)
        analysis_menu.addAction(track_all_action)

        analysis_menu.addSeparator()

        # Segmentation
        segmentation_action = QAction("&Segmentation", self.main_window)
        segmentation_action.setShortcut(QKeySequence("S"))
        segmentation_action.setStatusTip("Perform vessel segmentation")
        segmentation_action.triggered.connect(self.main_window.start_segmentation)
        analysis_menu.addAction(segmentation_action)

        # QCA
        qca_action = QAction("&QCA Analysis", self.main_window)
        qca_action.setShortcut(QKeySequence("Q"))
        qca_action.setStatusTip("Perform QCA analysis")
        qca_action.triggered.connect(self.main_window.start_qca_analysis)
        analysis_menu.addAction(qca_action)

        analysis_menu.addSeparator()

        # Sequential processing
        sequential_action = QAction("Sequential &Processing", self.main_window)
        sequential_action.setShortcut(QKeySequence("Ctrl+P"))
        sequential_action.setStatusTip("Process multiple frames sequentially")
        sequential_action.triggered.connect(self.main_window.start_sequential_processing)
        analysis_menu.addAction(sequential_action)

        analysis_menu.addSeparator()

        # Toggle advanced diameter
        toggle_diameter_action = QAction("Show &Diameter Measurements", self.main_window)
        toggle_diameter_action.setCheckable(True)
        toggle_diameter_action.setChecked(False)
        toggle_diameter_action.triggered.connect(self.main_window.toggle_advanced_diameter)
        analysis_menu.addAction(toggle_diameter_action)

        # Store reference
        self.main_window.toggle_diameter_action = toggle_diameter_action

    def _create_window_menu(self, window_menu):
        """Create window menu items"""
        # Reset layout
        reset_layout_action = QAction("&Reset Layout", self.main_window)
        reset_layout_action.setStatusTip("Reset window layout to default")
        reset_layout_action.triggered.connect(self.main_window.reset_layout)
        window_menu.addAction(reset_layout_action)

    def _create_help_menu(self, help_menu):
        """Create help menu items"""
        # Keyboard shortcuts
        shortcuts_action = QAction("&Keyboard Shortcuts", self.main_window)
        shortcuts_action.setShortcut(QKeySequence("F1"))
        shortcuts_action.setStatusTip("Show keyboard shortcuts")
        shortcuts_action.triggered.connect(self.main_window.show_shortcuts)
        help_menu.addAction(shortcuts_action)

        help_menu.addSeparator()

        # About
        about_action = QAction("&About", self.main_window)
        about_action.setStatusTip("About this application")
        about_action.triggered.connect(self.main_window.show_about)
        help_menu.addAction(about_action)

    def update_recent_menu(self):
        """Update the recent files menu"""
        if not self.recent_files_menu:
            return

        self.recent_files_menu.clear()

        recent_files = self.main_window.settings.value("recent_files", [])
        if not recent_files:
            no_recent_action = QAction("(No recent files)", self.main_window)
            no_recent_action.setEnabled(False)
            self.recent_files_menu.addAction(no_recent_action)
            return

        for i, filepath in enumerate(recent_files[:10]):  # Show max 10 recent files
            action = QAction(f"{i+1}. {filepath}", self.main_window)
            action.setData(filepath)
            action.triggered.connect(
                lambda checked, path=filepath: self.main_window.load_dicom_file(path)
            )
            self.recent_files_menu.addAction(action)

        self.recent_files_menu.addSeparator()
        clear_action = QAction("Clear Recent Files", self.main_window)
        clear_action.triggered.connect(self._clear_recent_files)
        self.recent_files_menu.addAction(clear_action)

    def _clear_recent_files(self):
        """Clear recent files list"""
        self.main_window.settings.setValue("recent_files", [])
        self.update_recent_menu()
