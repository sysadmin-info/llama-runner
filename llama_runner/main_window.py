import sys
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel,
                               QPushButton, QLineEdit, QTabWidget)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Llama Runner")

        self.layout = QVBoxLayout()

        self.tabs = QTabWidget()

        # Llama Runner Tab
        self.llama_tab = QWidget()
        self.llama_layout = QVBoxLayout()
        self.llama_layout.addWidget(QLabel("Llama Runner Status:"))
        self.llama_status_label = QLabel("Not Running")
        self.llama_layout.addWidget(self.llama_status_label)
        self.llama_start_button = QPushButton("Start Llama Runner")
        self.llama_stop_button = QPushButton("Stop Llama Runner")
        self.llama_layout.addWidget(self.llama_start_button)
        self.llama_layout.addWidget(self.llama_stop_button)
        self.llama_tab.setLayout(self.llama_layout)
        self.tabs.addTab(self.llama_tab, "Llama Runner")

        # LiteLLM Proxy Tab
        self.litellm_tab = QWidget()
        self.litellm_layout = QVBoxLayout()
        self.litellm_layout.addWidget(QLabel("LiteLLM Proxy Status:"))
        self.litellm_status_label = QLabel("Not Running")
        self.litellm_layout.addWidget(self.litellm_status_label)
        self.litellm_start_button = QPushButton("Start LiteLLM Proxy")
        self.litellm_stop_button = QPushButton("Stop LiteLLM Proxy")
        self.litellm_layout.addWidget(self.litellm_start_button)
        self.litellm_layout.addWidget(self.litellm_stop_button)
        self.litellm_tab.setLayout(self.litellm_layout)
        self.tabs.addTab(self.litellm_tab, "LiteLLM Proxy")

        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
````

Now that the basic UI structure is created, I will need to modify the `main.py` file to integrate the PySide6 UI with the LlamaCppRunner and LiteLLM proxy.

Do you want me to proceed with modifying the `main.py` file?
