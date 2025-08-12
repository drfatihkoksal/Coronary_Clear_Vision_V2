#!/usr/bin/env python3
"""
Siemens DICOM Viewer - Main Entry Point
MVP Phase 1: Basic DICOM Viewing
"""

import sys
import os
import logging
import subprocess
import traceback
from datetime import datetime
from pathlib import Path

# Disable problematic Qt plugins on macOS to prevent crashes
if sys.platform == 'darwin':
    os.environ['QT_PLUGIN_PATH'] = ''
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''
    # Disable location services to prevent crash
    os.environ['QT_ENABLE_GEOSERVICES_PLUGIN_CORELOCATION'] = '0'
    os.environ['QT_LOGGING_RULES'] = 'qt.permissions.corelocation.warning=false'
    # Additional macOS crash prevention
    os.environ['PYTHONFAULTHANDLER'] = '1'
    os.environ['QT_MAC_WANTS_LAYER'] = '1'

# Comprehensive import error handling
try:
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from PyQt6.QtGui import QFont
    from PyQt6.QtCore import Qt
except ImportError as e:
    print(f"CRITICAL: Failed to import PyQt6 components: {e}")
    print("Please ensure PyQt6 is installed: pip install PyQt6")
    sys.exit(1)

try:
    from src.ui.main_window_original import MainWindow
except ImportError as e:
    print(f"CRITICAL: Failed to import MainWindow: {e}")
    print("Please ensure all project files are present and accessible")
    sys.exit(1)

try:
    from src.core.model_manager import ModelManager
except ImportError as e:
    print(f"WARNING: Failed to import ModelManager: {e}")
    print("Some features may not be available")
    ModelManager = None

# Configure comprehensive logging
def setup_logging():
    """Setup comprehensive logging with rotation and error handling"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"coronary_analysis_{timestamp}.log"
    
    # Also maintain a latest.log symlink/copy
    latest_log = log_dir / "coronary_analysis_latest.log"
    
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.FileHandler(latest_log, mode='w', encoding='utf-8')
            ]
        )
        logging.info(f"Logging initialized. Log file: {log_file}")
        return True
    except Exception as e:
        print(f"WARNING: Failed to setup logging: {e}")
        # Fallback to basic logging
        logging.basicConfig(level=logging.INFO)
        return False

# Initialize logging
setup_logging()

# Wayland compatibility check
def check_wayland_compatibility():
    """Check and fix Wayland compatibility issues"""
    # Skip if already configured
    if os.environ.get('QT_QPA_PLATFORM'):
        return

    # Check if running under Wayland
    if os.environ.get('WAYLAND_DISPLAY'):
        # Try to import xcb to check if dependencies are available
        try:
            # Check if xcb platform is available by checking libraries
            # Try X11 if possible
            result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True, check=False)
            if 'libxcb-cursor' in result.stdout:
                os.environ['QT_QPA_PLATFORM'] = 'xcb'
                os.environ['GDK_BACKEND'] = 'x11'
            else:
                os.environ['QT_QPA_PLATFORM'] = 'wayland'
                os.environ['QT_WAYLAND_DISABLE_WINDOWDECORATION'] = '1'
        except Exception:
            # Fallback to Wayland
            os.environ['QT_QPA_PLATFORM'] = 'wayland'
            os.environ['QT_WAYLAND_DISABLE_WINDOWDECORATION'] = '1'

def handle_exception(exc_type, exc_value, exc_traceback):
    """Global exception handler for uncaught exceptions"""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    # Write detailed crash report
    write_crash_report(exc_type, exc_value, exc_traceback)
    
    # Try to show GUI error dialog
    try:
        app = QApplication.instance()
        if app:
            QMessageBox.critical(
                None, 
                "Critical Error",
                f"An unexpected error occurred:\n\n{exc_type.__name__}: {exc_value}\n\n"
                f"A crash report has been saved to crash_report.txt"
            )
    except Exception:
        pass

def write_crash_report(exc_type, exc_value, exc_traceback):
    """Write detailed crash report to file"""
    crash_dir = Path("crash_reports")
    crash_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    crash_file = crash_dir / f"crash_{timestamp}.txt"
    
    try:
        with open(crash_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("CORONARY CLEAR VISION CRASH REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Python Version: {sys.version}\n")
            f.write(f"Platform: {sys.platform}\n")
            f.write(f"Working Directory: {os.getcwd()}\n\n")
            
            f.write("EXCEPTION DETAILS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Type: {exc_type.__name__}\n")
            f.write(f"Value: {exc_value}\n\n")
            
            f.write("TRACEBACK:\n")
            f.write("-" * 50 + "\n")
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
            
            f.write("\n" + "=" * 70 + "\n")
        
        # Also copy to latest crash file
        latest_crash = crash_dir / "crash_latest.txt"
        import shutil
        shutil.copy2(crash_file, latest_crash)
        
        logging.info(f"Crash report written to {crash_file}")
    except Exception as e:
        logging.error(f"Failed to write crash report: {e}")

def main():
    """Main application entry point with comprehensive error handling"""
    
    # Install global exception handler
    sys.excepthook = handle_exception
    
    # Check Wayland compatibility before creating QApplication
    try:
        check_wayland_compatibility()
    except Exception as e:
        logging.warning(f"Wayland compatibility check failed: {e}")

    app = None
    window = None
    
    try:
        # Create Qt application
        app = QApplication(sys.argv)
        app.setOrganizationName("CoronaryClearVision")
        app.setApplicationName("DICOM Viewer")
        
        # Handle Qt warnings and critical messages
        app.aboutToQuit.connect(lambda: logging.info("Application shutting down"))

        # Set application-wide font (1.25x larger for all dialogs)
        try:
            app_font = QFont()
            app_font.setPointSize(13)  # 1.25x larger than default
            app.setFont(app_font)
        except Exception as e:
            logging.warning(f"Failed to set application font: {e}")

        # Set style sheet for all dialogs and widgets
        app.setStyleSheet("""
        QFileDialog {
            font-size: 13px;
        }
        QFileDialog QPushButton {
            min-height: 30px;
            padding: 5px 10px;
        }
        QFileDialog QComboBox {
            min-height: 28px;
        }
        QFileDialog QLineEdit {
            min-height: 28px;
            font-size: 13px;
        }
        QFileDialog QListView {
            font-size: 12px;
        }
        QFileDialog QTreeView {
            font-size: 12px;
        }
        QFileDialog QLabel {
            font-size: 12px;
        }
        
        QMessageBox {
            font-size: 13px;
        }
        QMessageBox QPushButton {
            min-height: 30px;
            min-width: 80px;
            font-size: 12px;
        }
        
        QInputDialog {
            font-size: 13px;
        }
        QInputDialog QLineEdit {
            font-size: 13px;
            min-height: 28px;
        }
        QInputDialog QPushButton {
            min-height: 28px;
            font-size: 12px;
        }
        
        QColorDialog {
            font-size: 12px;
        }
        
        QFontDialog {
            font-size: 12px;
        }
        
        /* Tooltip font size */
        QToolTip {
            font-size: 11px;
        }
        """)

        # Preload AngioPy model with error handling
        try:
            if ModelManager is not None:
                logging.info("Initializing model manager...")
                model_manager = ModelManager.instance()
                model_manager.preload_model_async(auto_download=True)
                logging.info("Model manager initialized successfully")
            else:
                logging.warning("ModelManager not available - some features disabled")
        except Exception as e:
            logging.error(f"Failed to initialize model manager: {e}")
            # Continue without models - basic functionality should still work
            
            # Notify user
            try:
                QMessageBox.warning(
                    None,
                    "Model Loading Warning", 
                    f"Failed to load AI models:\n{str(e)}\n\n"
                    "The application will continue with limited functionality."
                )
            except Exception:
                pass

        # Create and show main window with error recovery
        try:
            logging.info("Creating main window...")
            window = MainWindow()
            
            # Set up crash recovery
            window.destroyed.connect(lambda: logging.info("Main window destroyed"))
            
            # Show window
            window.show()
            logging.info("Main window displayed successfully")
            
        except Exception as e:
            logging.critical(f"Failed to create main window: {e}")
            
            # Try to show error dialog
            try:
                QMessageBox.critical(
                    None,
                    "Startup Error",
                    f"Failed to create main window:\n\n{str(e)}\n\n"
                    "The application cannot continue."
                )
            except Exception:
                pass
            
            # Write crash report and exit
            write_crash_report(type(e), e, sys.exc_info()[2])
            sys.exit(1)

        # Run application event loop
        logging.info("Starting application event loop...")
        exit_code = app.exec()
        logging.info(f"Application exited with code: {exit_code}")
        sys.exit(exit_code)

    except MemoryError as e:
        logging.critical(f"Out of memory: {e}")
        try:
            if app:
                QMessageBox.critical(
                    None,
                    "Memory Error",
                    "The application has run out of memory.\n"
                    "Please close other applications and try again."
                )
        except Exception:
            pass
        write_crash_report(type(e), e, sys.exc_info()[2])
        sys.exit(1)
        
    except KeyboardInterrupt:
        logging.info("Application interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        logging.exception("Fatal error in main loop")
        
        # Try to show error dialog
        try:
            if app:
                QMessageBox.critical(
                    None,
                    "Fatal Error",
                    f"Application crashed:\n\n{str(e)}\n\n"
                    f"Please check crash_reports/ folder for details."
                )
        except Exception:
            pass

        # Write crash report
        write_crash_report(type(e), e, sys.exc_info()[2])
        sys.exit(1)
        
    finally:
        # Cleanup
        try:
            if window:
                window.close()
            if app:
                app.quit()
            logging.info("Application cleanup completed")
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    main()
