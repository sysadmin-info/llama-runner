import logging
from typing import Optional, Dict, Any

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QSizePolicy
from PySide6.QtCore import Qt

def human_readable_size(size_in_bytes: Optional[int]) -> str:
    """Formats a size in bytes into a human-readable string (e.g., KB, MB, GB)."""
    if size_in_bytes is None:
        return "N/A"
    if size_in_bytes < 1024:
        return f"{size_in_bytes} B"
    elif size_in_bytes < 1024**2:
        return f"{size_in_bytes / 1024:.2f} KB"
    elif size_in_bytes < 1024**3:
        return f"{size_in_bytes / (1024**2):.2f} MB"
    elif size_in_bytes < 1024**4:
        return f"{size_in_bytes / (1024**3):.2f} GB"
    else:
        return f"{size_in_bytes / (1024**4):.2f} TB"

class ModelStatusWidget(QWidget):
    """
    Widget to display status and controls for a single model.
    """
    def __init__(self, model_name: str, metadata: Optional[Dict[str, Any]] = None, parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.metadata = metadata
        self.layout = QVBoxLayout()

        # Metadata section
        self.metadata_layout = QVBoxLayout()
        self.metadata_label = QLabel("<b>Metadata:</b>")
        self.metadata_layout.addWidget(self.metadata_label)

        self.arch_label = QLabel("Architecture: N/A")
        self.metadata_layout.addWidget(self.arch_label)

        self.quant_label = QLabel("Quantization: N/A")
        self.metadata_layout.addWidget(self.quant_label)

        self.size_label = QLabel("Size: N/A")
        self.metadata_layout.addWidget(self.size_label)

        self.layout.addLayout(self.metadata_layout) # Add metadata layout to main widget layout

        self.model_label = QLabel(f"<b>{self.model_name}</b>")
        self.model_label.setAlignment(Qt.AlignCenter)
        self.model_label.setStyleSheet("font-size: 16pt;") # Larger font for model name
        self.layout.addWidget(self.model_label)

        self.status_label = QLabel("Status: Not Running")
        self.layout.addWidget(self.status_label)

        self.port_label = QLabel("Port: N/A")
        self.layout.addWidget(self.port_label)

        self.start_button = QPushButton(f"Start {self.model_name}")
        self.layout.addWidget(self.start_button)

        self.stop_button = QPushButton(f"Stop {self.model_name}")
        self.stop_button.setEnabled(False) # Initially disabled
        self.layout.addWidget(self.stop_button)

        self.layout.addStretch() # Push everything to the top
        self.setLayout(self.layout)
        # Apply styling to the ModelStatusWidget and its children
        self.setStyleSheet("""
            ModelStatusWidget {
                background-color: #ffffff;
                border: 1px solid #dddddd;
                border-radius: 8px;
                padding: 15px; /* Add padding to the widget */
                margin: 10px; /* Add margin around the widget */
            }
            QLabel {
                font-size: 10pt; /* Default label font size */
                margin-bottom: 5px;
            }
            QLabel:first-child { /* Style for the "Metadata:" label */
                 font-weight: bold;
                 margin-bottom: 10px;
            }
            QLabel[text^="Architecture:"],
            QLabel[text^="Quantization:"],
            QLabel[text^="Size:"] {
                 font-size: 9pt; /* Smaller font for metadata details */
                 color: #555555;
                 margin-left: 10px; /* Indent metadata details */
            }
            QLabel[text^="Status:"] {
                 font-weight: bold;
                 margin-top: 10px;
            }
            QLabel[text^="Port:"] {
                 font-weight: bold;
            }
        """)

        # Update metadata display if metadata is provided
        if self.metadata:
            self.update_metadata(self.metadata)

    def update_metadata(self, metadata: Dict[str, Any]):
        """Updates the displayed metadata."""
        self.metadata = metadata
        self.arch_label.setText(f"Architecture: {metadata.get('arch', 'N/A')}")
        self.quant_label.setText(f"Quantization: {metadata.get('quantization', 'N/A')}")
        # Format size nicely, assuming size is in bytes (LM Studio format)
        size_bytes = metadata.get('size', None)
        if size_bytes is not None:
            # Attempt to convert to integer as per user suggestion
            try:
                size_bytes_int = int(size_bytes)
                self.size_label.setText(f"Size: {human_readable_size(size_bytes_int)}")
            except (ValueError, TypeError):
                # Fallback if conversion fails (should ideally not happen if gguf_metadata is fixed)
                logging.warning(f"Could not convert size metadata '{size_bytes}' to integer. Displaying raw value.")
                self.size_label.setText(f"Size: {size_bytes}") # Display raw value if conversion fails
        else:
            self.size_label.setText("Size: N/A")

    def update_status(self, status: str):
        self.status_label.setText(f"Status: {status}")

    def update_port(self, port: int | str):
        self.port_label.setText(f"Port: {port}")

    def set_buttons_enabled(self, start_enabled: bool, stop_enabled: bool):
        self.start_button.setEnabled(start_enabled)
        self.stop_button.setEnabled(stop_enabled)