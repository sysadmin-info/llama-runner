import sys
import logging # Import logging
import argparse # Import argparse
import os # Import os
from datetime import datetime # Import datetime
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon # Import QIcon

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
    parser.add_argument(
        "--log-prompts",
        action="store_true", # Store as boolean flag
        help="Enable logging of prompts (requests and responses) to a dedicated file."
    )
    args = parser.parse_args()

    # Ensure config directory exists for log files
    ensure_config_exists()

    # Map string level to logging constant for console handler

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
    if args.log_level.upper() == "DEBUG":
        console_handler.setLevel(logging.DEBUG)
    elif args.log_level.upper() == "INFO":
        console_handler.setLevel(logging.INFO)
    elif args.log_level.upper() == "WARNING":
        console_handler.setLevel(logging.WARNING)
    elif args.log_level.upper() == "ERROR":
        console_handler.setLevel(logging.ERROR)
    elif args.log_level.upper() == "CRITICAL":
        console_handler.setLevel(logging.CRITICAL)
    else:
        # Default to INFO if somehow an invalid level got through argparse
        console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Create file handler for app.log
    app_log_file_path = os.path.join(CONFIG_DIR, "app.log")
    try:
        app_file_handler = logging.FileHandler(app_log_file_path)
        # Set file level to DEBUG to capture all messages (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        app_file_handler.setLevel(logging.DEBUG)
        app_file_handler.setFormatter(formatter)
        root_logger.addHandler(app_file_handler)
        logging.info(f"App file logging enabled. Log file path: {app_log_file_path}")
    except Exception as e:
        logging.error(f"Failed to create app file handler for {app_log_file_path}: {e}")

    # Create a dedicated logger for prompts
    prompts_logger = logging.getLogger("prompts")
    prompts_logger.setLevel(logging.DEBUG) # Set level to DEBUG to capture all prompt messages

    # If --log-prompts is enabled, add a file handler for prompts
    if args.log_prompts:
        prompt_log_filename = f"prompts-{datetime.now().strftime('%Y%m%d')}.log"
        prompt_log_file_path = os.path.join(CONFIG_DIR, prompt_log_filename)
        try:
            prompt_file_handler = logging.FileHandler(prompt_log_file_path)
            # Use the same formatter, or a different one if preferred
            prompt_file_handler.setFormatter(formatter)
            # Set level for prompt file handler (e.g., INFO or DEBUG)
            prompt_file_handler.setLevel(logging.INFO) # Log INFO and above for prompts
            prompts_logger.addHandler(prompt_file_handler)
            logging.info(f"Prompt logging enabled. Log file path: {prompt_log_file_path}")
        except Exception as e:
            logging.error(f"Failed to create prompt file handler for {prompt_log_file_path}: {e}")

    # Store the prompt logging state to be passed to proxy threads
    prompt_logging_enabled = args.log_prompts

    logging.info(f"Console logging level set to {args.log_level.upper()}")
    logging.info(f"Prompt logging is {'enabled' if prompt_logging_enabled else 'disabled'}")


    # Create the Qt application instance
    app = QApplication(sys.argv)

    # Set the application icon
    app.setWindowIcon(QIcon('app_icon.png'))

    # Create the main window instance
    window = MainWindow()

    # Show the main window
    window.show()

    # Start the Qt event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
