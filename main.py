import sys
from PySide6.QtWidgets import QApplication

# Import the MainWindow class from your UI file
from llama_runner.main_window import MainWindow

def main():
    """
    Initializes and runs the PySide6 application.
    """
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
