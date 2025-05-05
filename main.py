import sys
import logging # Import logging
import argparse # Import argparse
import os # Import os
from PySide6.QtWidgets import QApplication

# Import CONFIG_DIR and ensure_config_exists
from llama_runner.config_loader import CONFIG_DIR, ensure_config_exists

# Import the MainWindow class from your UI file
# Import this *after* configuring logging
from llama_runner.main_window import MainWindow

def main():
    """
    Initializes and runs the PySide6 application.
    Configures logging to console and a file.
    """
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Llama Runner GUI application.")
    parser.add_argument(
        "--log-level",
        default="INFO", # Default to INFO
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the minimum logging level for console output (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    args = parser.parse_args()

    # Ensure config directory exists for log files
    ensure_config_exists()

    # Map string level to logging constant for console handler
    log_level_console = getattr(logging, args.log_level.upper(), logging.INFO)

    # Get the root logger
    root_logger = logging.getLogger()

    # Set the root logger level to DEBUG. This ensures that messages of all levels
    # are processed by the logger and passed to handlers. Handlers will then filter
    # based on their own levels.
    root_logger.setLevel(logging.DEBUG)

    # Remove any default handlers added by basicConfig if it was called implicitly
    # before this point (e.g., by an imported module's top-level logging call).
    # This ensures we have full control over handlers.
    if root_logger.hasHandlers():
        # Iterate over a copy of the list to allow removal
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    # Create formatter
    # Added %(name)s to the format to show which logger emitted the message
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    # Set console level based on argument
    console_handler.setLevel(log_level_console)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Create file handler for app.log
    log_file_path = os.path.join(CONFIG_DIR, "app.log")
    try:
        file_handler = logging.FileHandler(log_file_path)
        # Set file level to DEBUG to capture all messages (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logging.info(f"File logging enabled. Log file path: {log_file_path}")
    except Exception as e:
        logging.error(f"Failed to create file handler for {log_file_path}: {e}")


    logging.info(f"Console logging level set to {args.log_level.upper()}")


    # Create the Qt application instance
    app = QApplication(sys.argv)

    # Create the main window instance
    window = MainWindow()

    # Show the main window
    window.show()

    # Start the Qt event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
