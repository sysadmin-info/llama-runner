from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QTextEdit, QDialogButtonBox, QSizePolicy
)

class ErrorOutputDialog(QDialog):
    """
    Custom dialog to display error message and process output.
    """
    def __init__(self, title, message, output_lines, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(400)
        self.setMinimumHeight(200)

        layout = QVBoxLayout()

        message_label = QLabel(message)
        layout.addWidget(message_label)

        if output_lines:
            output_text_edit = QTextEdit()
            output_text_edit.setReadOnly(True)
            output_text_edit.setPlainText("\n".join(output_lines))
            output_text_edit.setMinimumHeight(100)
            output_text_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            layout.addWidget(QLabel("Last Output Lines:"))
            layout.addWidget(output_text_edit)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)

        self.setLayout(layout)