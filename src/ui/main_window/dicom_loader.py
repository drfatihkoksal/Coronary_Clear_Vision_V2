"""DICOM file loading and management"""

from PyQt6.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QListWidget,
    QDialog,
    QVBoxLayout,
    QPushButton,
    QLabel,
)
import os
import logging
import traceback
from typing import List

from ...core.dicom_parser import DicomParser
from ...core.dicom_folder_loader import DicomFolderLoader
from ...domain.models.dicom_models import DicomSeries

logger = logging.getLogger(__name__)


class DicomLoader:
    """Handles DICOM file loading operations"""

    def __init__(self, main_window):
        """
        Initialize DICOM loader

        Args:
            main_window: Reference to the main window
        """
        self.main_window = main_window
        self.recent_files = []

    def open_dicom_file(self):
        """Open a single DICOM file via file dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window, "Open DICOM File", "", "DICOM Files (*.dcm *.dicom);;All Files (*.*)"
        )

        if file_path:
            self.load_dicom_file(file_path)

    def open_dicom_folder(self):
        """Open a folder containing DICOM files"""
        folder_path = QFileDialog.getExistingDirectory(
            self.main_window, "Select DICOM Folder", "", QFileDialog.Option.ShowDirsOnly
        )

        if folder_path:
            self._load_dicom_folder(folder_path)

    def open_default_folder(self):
        """Open the default DICOM folder (000...0A-F)"""
        base_path = "/Users/fatihkoksal/Projelerim/Coronary_Clear_Vision"
        folders = ["0000000A", "0000000B", "0000000C", "0000000D", "0000000E", "0000000F"]

        found_files = []
        for folder in folders:
            folder_path = os.path.join(base_path, folder)
            if os.path.exists(folder_path):
                dcm_files = [f for f in os.listdir(folder_path) if f.endswith(".dcm")]
                for dcm_file in dcm_files:
                    found_files.append(os.path.join(folder_path, dcm_file))

        if not found_files:
            QMessageBox.warning(
                self.main_window, "No DICOM Files", "No DICOM files found in default folders."
            )
            return

        # Load all series
        self._load_multiple_dicom_files(found_files)

    def load_dicom_file(self, file_path: str):
        """
        Load a specific DICOM file

        Args:
            file_path: Path to the DICOM file
        """
        try:
            # Check if it's a folder
            if os.path.isdir(file_path):
                self._load_dicom_folder(file_path)
                return

            # Single file
            parser = DicomParser()
            success = parser.load_dicom(file_path)

            if success:
                self._set_dicom_parser(parser)
                self._add_to_recent(file_path)
                self.main_window.save_settings()
                logger.info(f"Loaded DICOM file: {file_path}")
            else:
                QMessageBox.warning(
                    self.main_window, "Load Failed", f"Failed to load DICOM file:\n{file_path}"
                )

        except Exception as e:
            logger.error(f"Error loading DICOM: {e}\n{traceback.format_exc()}")
            QMessageBox.critical(self.main_window, "Error", f"Error loading DICOM file:\n{str(e)}")

    def _load_dicom_folder(self, folder_path: str):
        """Load all DICOM files from a folder"""
        try:
            loader = DicomFolderLoader()
            series_list = loader.load_folder(folder_path)

            if not series_list:
                QMessageBox.warning(
                    self.main_window,
                    "No DICOM Files",
                    "No valid DICOM files found in the selected folder.",
                )
                return

            # Store available projections
            self.main_window.available_projections = series_list
            self.main_window.current_projection_index = 0

            # If only one series, load it directly
            if len(series_list) == 1:
                self._load_series(series_list[0])
            else:
                # Show series selection dialog
                self._show_series_selection_dialog(series_list)

        except Exception as e:
            logger.error(f"Error loading folder: {e}")
            QMessageBox.critical(
                self.main_window, "Error", f"Error loading DICOM folder:\n{str(e)}"
            )

    def _load_multiple_dicom_files(self, file_paths: List[str]):
        """Load multiple DICOM files and organize by series"""
        try:
            DicomFolderLoader()
            series_list = []

            # Group files by series
            for file_path in file_paths:
                try:
                    parser = DicomParser()
                    if parser.load_dicom(file_path):
                        # Create a series entry
                        series = DicomSeries(
                            series_uid=parser.series_instance_uid or "Unknown",
                            series_number=parser.series_number or 0,
                            series_description=parser.series_description or "Unknown Series",
                            num_instances=1,
                            file_paths=[file_path],
                        )
                        series_list.append(series)
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")

            if series_list:
                self.main_window.available_projections = series_list
                self.main_window.current_projection_index = 0

                if len(series_list) == 1:
                    self._load_series(series_list[0])
                else:
                    self._show_series_selection_dialog(series_list)

        except Exception as e:
            logger.error(f"Error loading multiple files: {e}")

    def _load_series(self, series: DicomSeries):
        """Load a specific DICOM series"""
        try:
            # Reset any existing data
            self.main_window._reset_analysis_state()

            # Load first file in series
            if series.file_paths:
                parser = DicomParser()
                success = parser.load_dicom(series.file_paths[0])

                if success:
                    self._set_dicom_parser(parser)
                    logger.info(f"Loaded series: {series.series_description}")

        except Exception as e:
            logger.error(f"Error loading series: {e}")
            QMessageBox.critical(self.main_window, "Error", f"Error loading series:\n{str(e)}")

    def _show_series_selection_dialog(self, series_list: List[DicomSeries]):
        """Show dialog to select from multiple series"""
        dialog = QDialog(self.main_window)
        dialog.setWindowTitle("Select DICOM Series")
        dialog.setMinimumWidth(500)

        layout = QVBoxLayout()

        label = QLabel(f"Found {len(series_list)} DICOM series. Select one to open:")
        layout.addWidget(label)

        # List widget
        list_widget = QListWidget()
        for i, series in enumerate(series_list):
            item_text = f"{i+1}. {series.series_description} ({series.num_instances} files)"
            list_widget.addItem(item_text)
        list_widget.setCurrentRow(0)
        layout.addWidget(list_widget)

        # Buttons
        button_layout = QVBoxLayout()
        open_button = QPushButton("Open Series")
        cancel_button = QPushButton("Cancel")

        button_layout.addWidget(open_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        def on_open():
            selected_index = list_widget.currentRow()
            if selected_index >= 0:
                self._load_series(series_list[selected_index])
                dialog.accept()

        open_button.clicked.connect(on_open)
        cancel_button.clicked.connect(dialog.reject)
        list_widget.itemDoubleClicked.connect(lambda: on_open())

        dialog.setLayout(layout)
        dialog.exec()

    def _set_dicom_parser(self, parser: DicomParser):
        """Set the DICOM parser and update UI"""
        self.main_window.dicom_parser = parser
        self.main_window.viewer_widget.set_dicom_parser(parser)

        # Update UI elements
        self.main_window.frame_slider.setMaximum(parser.num_frames - 1)
        self.main_window.frame_slider.setValue(0)
        self.main_window.frame_slider.setEnabled(True)

        # Extract timestamps if available
        self.main_window.extract_frame_timestamps()

        # Update viewer
        self.main_window.viewer_widget.display_frame(0)
        self.main_window.viewer_widget.fit_to_window()

        # Update status
        file_name = os.path.basename(parser.file_path)
        self.main_window.update_status(f"Loaded: {file_name} ({parser.num_frames} frames)")

        # Enable navigation buttons
        self.main_window.next_button.setEnabled(True)
        self.main_window.prev_button.setEnabled(True)
        self.main_window.play_button.setEnabled(True)

        # Extract and display ECG if available
        self.main_window.extract_and_display_ekg()

    def _add_to_recent(self, file_path: str):
        """Add file to recent files list"""
        recent_files = self.main_window.settings.value("recent_files", [])

        # Remove if already exists
        if file_path in recent_files:
            recent_files.remove(file_path)

        # Add to beginning
        recent_files.insert(0, file_path)

        # Keep only last 10
        recent_files = recent_files[:10]

        # Save
        self.main_window.settings.setValue("recent_files", recent_files)

        # Update menu if exists
        if hasattr(self.main_window, "menu_manager"):
            self.main_window.menu_manager.update_recent_menu()

    def auto_load_last_file(self, file_path: str, load_type: str, series_index: int = 0):
        """Auto-load last opened file on startup"""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Last file no longer exists: {file_path}")
                return

            if load_type == "file":
                self.load_dicom_file(file_path)
            elif load_type == "folder":
                self._load_dicom_folder(file_path)
            elif load_type == "series" and hasattr(self.main_window, "available_projections"):
                if series_index < len(self.main_window.available_projections):
                    self._load_series(self.main_window.available_projections[series_index])

        except Exception as e:
            logger.error(f"Failed to auto-load last file: {e}")
