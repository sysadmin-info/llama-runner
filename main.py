import sys
import logging # Import logging
import argparse # Import argparse
from PySide6.QtWidgets import QApplication

# Import the MainWindow class from your UI file
# Import this *after* configuring basicConfig
from llama_runner.main_window import MainWindow

def main():
    """
    Initializes and runs the PySide6 application.
    """
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Llama Runner GUI application.")
    parser.add_argument(
        "--log-level",
        default="INFO", # Default to INFO
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    args = parser.parse_args()

    # Map string level to logging constant
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)

    # Configure logging *before* importing other modules that might use it
    # basicConfig does nothing if the root logger already has handlers,
    # so configuring it here first ensures our level is used.
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Logging level set to {args.log_level.upper()}")


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
