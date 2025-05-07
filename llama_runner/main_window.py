import asyncio
import os
import logging
import traceback
from typing import Optional, Dict, Any # Import Callable

# Removed: from litellm.proxy.proxy_server import app

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QDialog, QTextEdit, QDialogButtonBox, QListWidget,
                               QStackedWidget, QSizePolicy)
from PySide6.QtCore import QThread, Signal, Slot, Qt, QTimer # Import QTimer

from llama_runner.config_loader import load_config
from llama_runner.llama_cpp_runner import LlamaCppRunner
# Updated import: Import FastAPIProxyThread
from llama_runner.lmstudio_proxy_thread import FastAPIProxyThread # Import the FastAPI proxy thread
from llama_runner.ollama_proxy_thread import OllamaProxyThread # Import the Ollama proxy thread
from llama_runner import gguf_metadata # Import the new metadata module

# Configure logging
# Note: basicConfig is now handled in main.py for configurable levels
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
class LlamaRunnerThread(QThread):
    """
    QThread to run the LlamaCppRunner in a separate thread to avoid blocking the UI.
    """
    started = Signal()
    stopped = Signal()
    error = Signal(str, list) # Signal includes error message and output buffer
    port_ready = Signal(str, int) # Signal includes model_name and port

    def __init__(self, model_name: str, model_path: str, llama_cpp_runtime: str = None, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.model_path = model_path
        self.llama_cpp_runtime = llama_cpp_runtime
        self.kwargs = kwargs
        self.runner = None
        self.is_running = False
        self._error_emitted = False # Flag to track if error signal was emitted

    def run(self):
        """
        Runs the LlamaCppRunner in the thread.
        """
        self.is_running = True
        self._error_emitted = False # Reset flag
        try:
            # Use a new event loop for this thread
            # Note: This thread runs its own asyncio loop. Bridging is needed
            # to communicate with the main thread's Qt loop.
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.run_async())
        except Exception as e:
            # This catch is mostly for unexpected errors in the asyncio loop itself
            logging.error(f"Unexpected error in LlamaRunnerThread run: {e}\n{traceback.format_exc()}")
            # If an error occurred here and not in run_async, emit a generic error
            if not self._error_emitted:
                 # Pass an empty list for output buffer in this case
                 self.error.emit(f"Unexpected thread error: {e}", [])
                 self._error_emitted = True
        finally:
            self.is_running = False
            # The stopped signal is emitted in run_async's finally block
            if hasattr(self, 'loop') and self.loop.is_running():
                 self.loop.stop()
            if hasattr(self, 'loop') and not self.loop.is_closed():
                 self.loop.close()


    async def run_async(self):
        """
        Asynchronous part of the runner.
        """
        try:
            self.runner = LlamaCppRunner(
                model_name=self.model_name,
                model_path=self.model_path,
                llama_cpp_runtime=self.llama_cpp_runtime,
                **self.kwargs
            )
            # The start method now raises exceptions on failure
            await self.runner.start()

            # Check if startup message was found and port was set
            if self.runner.port is None:
                 raise RuntimeError("Llama.cpp server failed to start or extract port.")

            self.started.emit()
            # Emit model_name along with the port number
            self.port_ready.emit(self.model_name, self.runner.get_port())

            # Keep the runner alive until the thread is stopped or process exits
            while self.is_running and self.runner.is_running():
                await asyncio.sleep(1)

            # If loop exited because self.is_running became False (clean stop requested)
            if not self.is_running and self.runner.is_running():
                 await self.runner.stop() # Stop the process

            # If loop exited because process stopped unexpectedly (self.is_running is still True)
            # The finally block will handle checking return code and emitting error/stopped

        except Exception as e:
            logging.error(f"Error running LlamaCppRunner: {e}\n{traceback.format_exc()}")
            # Emit the error message and the output buffer
            output_buffer = self.runner.get_output_buffer() if self.runner else []
            self.error.emit(str(e), output_buffer)
            self._error_emitted = True # Set flag

        finally:
            # Ensure the runner process is stopped if it's still running
            if self.runner and self.runner.is_running():
                logging.warning(f"Llama.cpp process for {self.model_name} was still running in finally block, stopping.")
                try:
                    await self.runner.stop()
                except Exception as stop_e:
                    logging.error(f"Error stopping LlamaCppRunner in finally: {stop_e}\n{traceback.format_exc()}")
                    # Don't overwrite the main error_occurred flag

            # Check the final return code after ensuring the process is stopped
            # Only emit error if one hasn't been emitted by the except block
            if not self._error_emitted and self.runner and self.runner.process and self.runner.process.returncode != 0:
                error_message = f"Llama.cpp server for {self.model_name} exited with code {self.runner.process.returncode}"
                output_buffer = self.runner.get_output_buffer() if self.runner else []
                logging.error(error_message)
                self.error.emit(error_message, output_buffer)
                self._error_emitted = True # Set flag

            self.is_running = False # Ensure this is false
            self.stopped.emit() # Always emit stopped when the thread is truly finished


    def stop(self):
        """
        Signals the LlamaCppRunner thread to stop.
        The actual stopping happens in run_async.
        """
        self.is_running = False
        # If the asyncio loop is running, schedule the stop coroutine
        if hasattr(self, 'loop') and self.loop.is_running():
             asyncio.run_coroutine_threadsafe(self._request_stop_runner(), self.loop)

    async def _request_stop_runner(self):
        """Helper coroutine to request runner stop from within the thread's loop."""
        if self.runner:
            await self.runner.stop()


# FastAPIProxyThread is now in its own file: llama_runner.lite_llm_proxy_thread


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
            QPushButton {
                background-color: #007bff; /* Primary button color */
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 15px; /* Padding for buttons */
                font-size: 10pt;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #0056b3; /* Darker color on hover */
            }
            QPushButton:pressed {
                background-color: #004085; /* Even darker color when pressed */
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
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


class MainWindow(QWidget):
    # Signal emitted when a runner's port is ready
    # This signal is emitted by MainWindow, connected to the proxy thread
    runner_port_ready_for_proxy = Signal(str, int)
    # Signal emitted when a runner stops
    # This signal is emitted by MainWindow, connected to the proxy thread
    runner_stopped_for_proxy = Signal(str)

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Llama Runner")
        self.resize(800, 600)

        self.config = load_config()
        self.llama_runtimes = self.config.get("llama-runtimes", {})
        self.default_runtime = "llama-server"
        self.models = self.config.get("models", {})
        # Removed: self.litellm_proxy_config = self.config.get("litellm-proxy", {}) # LiteLLM config removed
        # Get concurrent runners limit from config, default to 1 if not specified or invalid
        self.concurrent_runners_limit = self.config.get("concurrentRunners", 1)
        if not isinstance(self.concurrent_runners_limit, int) or self.concurrent_runners_limit < 1:
             logging.warning(f"Invalid 'concurrentRunners' value in config: {self.concurrent_runners_limit}. Defaulting to 1.")
             self.concurrent_runners_limit = 1

        # Initialize metadata cache and load metadata for all models
        gguf_metadata.ensure_cache_dir_exists()
        self.model_metadata_cache = {}
        for model_name, model_config in self.models.items():
             model_path = model_config.get("model_path")
             if model_path:
                 # Load/extract metadata on startup. State is 'not-loaded' initially.
                 metadata = gguf_metadata.get_model_lmstudio_format(model_name, model_path, is_running=False)
                 if metadata:
                     self.model_metadata_cache[model_name] = metadata
             else:
                 logging.warning(f"Model '{model_name}' has no 'model_path' in config. Skipping metadata caching.")


        self.llama_runner_threads: Dict[str, LlamaRunnerThread] = {}  # Dictionary to store threads for each model
        # Updated type hint for proxy thread
        self.fastapi_proxy_thread: Optional[FastAPIProxyThread] = None
        self.ollama_proxy_thread: Optional[OllamaProxyThread] = None # Add Ollama proxy thread instance

        # Dictionary to hold asyncio Futures for runners that are starting
        # Key: model_name, Value: asyncio.Future
        self._runner_startup_futures: Dict[str, asyncio.Future] = {}


        # Use QHBoxLayout for side-by-side layout
        self.layout = QHBoxLayout() # Main layout is now QHBoxLayout

        # Left side: Model List
        self.model_list_widget = QListWidget()
        self.model_list_widget.setMinimumWidth(150) # Give the list some space
        # Style the model list
        self.model_list_widget.setStyleSheet("""
            QListWidget {
                border: 0;
                border-radius: 5px;
                padding: 5px;
                background-color: #f0f0f0;
            }
            QListWidget::item {
                padding: 8px; /* Add padding to list items */
                font-size: 12pt; /* Increase font size */
                margin-bottom: 4px; /* Add space between items */
            }
            QListWidget::item:selected {
                background-color: #a0c0f0; /* Highlight color */
                color: #333333;
            }
        """)
        self.layout.addWidget(self.model_list_widget)

        # Right side: Model Status Stack
        self.model_status_stack = QStackedWidget()
        self.layout.addWidget(self.model_status_stack)

        self.model_status_widgets: Dict[str, ModelStatusWidget] = {} # Store status widgets by model name

        # Populate model list and create status widgets
        for model_name in self.models.keys():
            self.model_list_widget.addItem(model_name)

            # Get metadata for the model
            model_metadata = self.model_metadata_cache.get(model_name)
            status_widget = ModelStatusWidget(model_name, metadata=model_metadata)
            self.model_status_stack.addWidget(status_widget)
            self.model_status_widgets[model_name] = status_widget

            # Connect signals from the status widget's buttons
            # Use lambda to pass the model_name to the slots
            status_widget.start_button.clicked.connect(lambda checked, name=model_name: self.request_runner_start(name))
            status_widget.stop_button.clicked.connect(lambda checked, name=model_name: self.stop_llama_runner(name))


        # Connect model list selection change
        self.model_list_widget.currentItemChanged.connect(self.on_model_selection_changed)

        # Add a placeholder widget if no model is selected initially
        self.no_model_selected_widget = QWidget()
        no_model_layout = QVBoxLayout(self.no_model_selected_widget)
        no_model_label = QLabel("Select a model from the list.")
        no_model_label.setAlignment(Qt.AlignCenter)
        no_model_layout.addWidget(no_model_label)
        no_model_layout.addStretch()
        self.model_status_stack.addWidget(self.no_model_selected_widget)
        self.model_status_stack.setCurrentWidget(self.no_model_selected_widget) # Show placeholder initially


        # The main layout now contains the model list and status stack directly
        self.setLayout(self.layout)

        # Connect buttons to actions (removed proxy buttons)
        # self.litellm_start_button.clicked.connect(self.start_litellm_proxy)
        # self.litellm_stop_button.clicked.connect(self.stop_litellm_proxy)

        # Connect signals from LlamaRunnerThreads (will be connected when threads are created)
        # We need to connect these signals dynamically when a thread is started.

        # --- Start the FastAPI Proxy automatically ---
        self.start_fastapi_proxy() # Updated method call
        # --- End automatic proxy start ---

        # --- Start the Ollama Proxy automatically ---
        self.start_ollama_proxy() # Add method call to start Ollama proxy
        # --- End automatic proxy start ---


    def closeEvent(self, event):
        """
        Handles the window close event. Stops all running threads.
        """
        print("MainWindow closing. Stopping all runners and proxy...")
        self.stop_all_llama_runners()
        self.stop_fastapi_proxy() # Updated method call
        self.stop_ollama_proxy() # Add method call to stop Ollama proxy

        # Give threads a moment to stop gracefully
        # A more robust shutdown might involve waiting for threads to finish
        # using thread.wait() or similar, but doing so in the closeEvent
        # can potentially freeze the UI if threads don't exit quickly.
        # Signaling and letting them clean up in their run methods is generally preferred.
        # For this application's scale, a small sleep might be acceptable,
        # but relying on the threads' internal stop logic is better.
        # time.sleep(0.5) # Optional: brief pause

        # Accept the close event to allow the window to close
        event.accept()

    @Slot(str)
    def on_model_selection_changed(self, current_item, previous_item):
        """
        Slot to handle selection changes in the model list.
        Switches the stacked widget to show the selected model's status.
        """
        if current_item:
            model_name = current_item.text()
            if model_name in self.model_status_widgets:
                self.model_status_stack.setCurrentWidget(self.model_status_widgets[model_name])
            else:
                 # Should not happen if list is populated from models dict
                 logging.error(f"Status widget not found for selected model: {model_name}")
                 self.model_status_stack.setCurrentWidget(self.no_model_selected_widget)
        else:
            # No item selected
            self.model_status_stack.setCurrentWidget(self.no_model_selected_widget)

    def is_llama_runner_running(self, model_name: str) -> bool:
        """
        Checks if the Llama runner thread for a given model is currently running
        and if the underlying process is also running.
        This method is a callback passed to the proxy thread.
        """
        thread = self.llama_runner_threads.get(model_name)
        if thread and thread.isRunning():
            # Check if the runner instance exists and its process is running
            if thread.runner and thread.runner.is_running():
                 return True
            else:
                 # Thread is running, but runner process is not. This is an inconsistent state.
                 # Log a warning and treat as not running.
                 logging.warning(f"Llama runner thread for {model_name} is running, but process is not.")
                 return False
        return False

    def get_runner_port(self, model_name: str) -> Optional[int]:
        """
        Gets the port of a running Llama runner.
        This method is a callback passed to the proxy thread.
        Returns the port number or None if the runner is not running or port is not available.
        """
        thread = self.llama_runner_threads.get(model_name)
        if thread and thread.isRunning() and thread.runner and thread.runner.is_running():
             return thread.runner.get_port()
        return None

    def request_runner_start(self, model_name: str) -> asyncio.Future:
        """
        Initiates the process of starting a Llama runner for the given model.
        Handles concurrency limits. This method is a callback passed to the proxy thread.
        It returns an asyncio.Future that will be resolved when the runner is ready
        or an exception occurs.
        """
        logging.info(f"Received request to start runner for model: {model_name}")

        # Check if a Future already exists for this model (meaning startup is in progress)
        if model_name in self._runner_startup_futures and not self._runner_startup_futures[model_name].done():
             logging.info(f"Runner for {model_name} is already starting. Returning existing Future.")
             return self._runner_startup_futures[model_name]

        # Check if the runner is already running
        if self.is_llama_runner_running(model_name):
             port = self.get_runner_port(model_name)
             if port is not None:
                 logging.info(f"Runner for {model_name} is already running on port {port}. Returning completed Future.")
                 future = asyncio.Future()
                 future.set_result(port)
                 # Store completed future briefly to prevent duplicate starts if requests arrive rapidly
                 self._runner_startup_futures[model_name] = future
                 # Use a timer to remove the completed future after a short delay
                 QTimer.singleShot(1000, lambda: self._cleanup_completed_future(model_name))
                 return future
             else:
                 # Should not happen if is_llama_runner_running is True, but handle defensively
                 logging.error(f"Runner for {model_name} is reported as running but port is None.")
                 future = asyncio.Future()
                 future.set_exception(RuntimeError(f"Runner for {model_name} is running but port is unavailable."))
                 self._runner_startup_futures[model_name] = future
                 QTimer.singleShot(1000, lambda: self._cleanup_completed_future(model_name))
                 return future


        # Check concurrent runner limit
        running_runners = {name: thread for name, thread in self.llama_runner_threads.items() if thread.isRunning()}
        num_running = len(running_runners)

        if num_running >= self.concurrent_runners_limit:
            if self.concurrent_runners_limit == 1:
                logging.info(f"Concurrent runner limit ({self.concurrent_runners_limit}) reached. Stopping existing runner before starting {model_name}.")
                # Stop the first running runner found (or all if limit is 1)
                # Signal stop for all currently running runners
                for name_to_stop in list(running_runners.keys()):
                     self.stop_llama_runner(name_to_stop)

                # The proxy thread will need to implement the waiting logic for the old runner(s)
                # to stop and the new one to start. This method just initiates the stop/start process.
                # We still create a Future for the *new* runner we are about to start.

            else:
                logging.warning(f"Concurrent runner limit ({self.concurrent_runners_limit}) reached. Cannot start {model_name}.")
                future = asyncio.Future()
                future.set_exception(RuntimeError(f"Concurrent runner limit ({self.concurrent_runners_limit}) reached. Cannot start '{model_name}'."))
                # Store completed future briefly to prevent duplicate starts
                self._runner_startup_futures[model_name] = future
                QTimer.singleShot(1000, lambda: self._cleanup_completed_future(model_name))
                return future

        # If we reached here, we are clear to start the requested runner (either limit allows, or old ones were signaled to stop)

        # Create a new Future for this startup request
        future = asyncio.Future()
        self._runner_startup_futures[model_name] = future

        model_config = self.models[model_name]
        model_path = model_config.get("model_path")
        llama_cpp_runtime_key = model_config.get("llama_cpp_runtime", "default")
        _raw_llama_cpp_runtime_config = self.llama_runtimes.get(llama_cpp_runtime_key, self.default_runtime)
        
        # Extract the actual runtime command string
        if isinstance(_raw_llama_cpp_runtime_config, dict):
            llama_cpp_runtime_command = _raw_llama_cpp_runtime_config.get("runtime")
            if not llama_cpp_runtime_command: # Should not happen if config_loader is correct
                logging.error(f"Runtime configuration for '{llama_cpp_runtime_key}' is a dict but missing 'runtime' key.")
                future.set_exception(RuntimeError(f"Invalid runtime config for '{llama_cpp_runtime_key}'."))
                return future
        elif isinstance(_raw_llama_cpp_runtime_config, str):
            llama_cpp_runtime_command = _raw_llama_cpp_runtime_config
        else:
            # This case should ideally not be reached if config_loader normalizes correctly
            logging.error(f"Unexpected type for runtime configuration '{llama_cpp_runtime_key}': {type(_raw_llama_cpp_runtime_config)}")
            future.set_exception(RuntimeError(f"Invalid runtime configuration type for '{llama_cpp_runtime_key}'."))
            return future

        if not model_path:
             logging.error(f"Configuration Error: Model '{model_name}' has no 'model_path' specified in config.json.")
             future.set_exception(RuntimeError(f"Configuration Error: Model '{model_name}' has no 'model_path'."))
             return future

        # Check if the model file exists
        if not os.path.exists(model_path):
             logging.error(f"File Not Found: Model file not found: {model_path}")
             future.set_exception(FileNotFoundError(f"Model file not found: {model_path}"))
             return future

        # Check if the runtime executable exists if it's not the default "llama-server" (which is expected in PATH)
        if llama_cpp_runtime_key != "default" and not os.path.exists(llama_cpp_runtime_command):
             logging.error(f"Runtime Not Found: Llama.cpp runtime not found: {llama_cpp_runtime_command}")
             future.set_exception(FileNotFoundError(f"Llama.cpp runtime not found: {llama_cpp_runtime_command}"))
             return future


        print(f"Starting Llama Runner for {model_name}...")
        status_widget = self.model_status_widgets.get(model_name)
        if status_widget:
            status_widget.update_status("Starting...")
            status_widget.update_port("N/A")
            status_widget.set_buttons_enabled(False, False) # Disable both while starting

        thread = LlamaRunnerThread(
            model_name=model_name,
            model_path=model_path,
            llama_cpp_runtime=llama_cpp_runtime_command,
            **model_config.get("parameters", {})
        )
        # Connect signals
        thread.started.connect(lambda: self.on_llama_runner_started(model_name))
        thread.stopped.connect(lambda: self.on_llama_runner_stopped(model_name))
        thread.error.connect(lambda msg, output: self.on_llama_runner_error(model_name, msg, output))
        # Connect the port_ready signal to a slot that emits the MainWindow's signal
        thread.port_ready.connect(self.on_llama_runner_port_ready_and_emit)

        self.llama_runner_threads[model_name] = thread
        thread.start()

        # Return the Future that will be resolved when the runner is ready
        return future

    def _cleanup_completed_future(self, model_name: str):
        """Helper to remove a completed future from the dictionary after a delay."""
        if model_name in self._runner_startup_futures and self._runner_startup_futures[model_name].done():
             logging.debug(f"Cleaning up completed future for {model_name}")
             del self._runner_startup_futures[model_name]


    def stop_llama_runner(self, model_name: str):
        """
        Stops the LlamaCppRunner thread for a specific model.
        """
        if model_name in self.llama_runner_threads and self.llama_runner_threads[model_name].isRunning():
            print(f"Stopping Llama Runner for {model_name}...")
            status_widget = self.model_status_widgets.get(model_name)
            if status_widget:
                status_widget.update_status("Stopping...")
                status_widget.set_buttons_enabled(False, False) # Disable both while stopping

            self.llama_runner_threads[model_name].stop()
            # Don't wait() here in the UI thread, it will freeze the UI.
            # The thread will emit 'stopped' when it's done.
        else:
            print(f"Llama Runner for {model_name} is not running.")
            # If it's not running but still in the dict (e.g. crashed before cleanup),
            # clean up the UI state and dict entry.
            if model_name in self.llama_runner_threads:
                 print(f"Cleaning up state for non-running thread {model_name}")
                 # Simulate the stopped signal handling
                 self.on_llama_runner_stopped(model_name)


    def stop_all_llama_runners(self):
        """
        Stops all LlamaCppRunner threads.
        """
        print("Stopping all Llama Runners...")
        # Iterate over a copy of the keys because stop_llama_runner modifies the dict
        for model_name in list(self.llama_runner_threads.keys()):
            self.stop_llama_runner(model_name)
            # Add a small delay between stopping multiple runners if needed,
            # but the wait logic in start_llama_runner is more critical.
            # time.sleep(0.1) # Optional: brief pause


    def start_fastapi_proxy(self): # Updated method name
        """
        Starts the FastAPI proxy in a separate thread.
        This is now called automatically in __init__.
        """
        # Updated variable name
        if self.fastapi_proxy_thread is not None and self.fastapi_proxy_thread.isRunning():
            print("FastAPI Proxy is already running.") # Updated print
            return

        print("Starting FastAPI Proxy...") # Updated print
        # Updated status label variable name
        # Proxy buttons are removed, status label is enough

        # Removed: proxy_api_key = self.litellm_proxy_config.get("api_key") # LiteLLM config removed

        # Pass models config and the necessary callback methods to the proxy thread
        # Updated class name
        self.fastapi_proxy_thread = FastAPIProxyThread(
            all_models_config=self.models, # Pass the main models configuration
            runtimes_config=self.llama_runtimes, # Pass the runtimes configuration
            is_model_running_callback=self.is_llama_runner_running, # Pass the callback method
            get_runner_port_callback=self.get_runner_port, # Pass the callback method
            request_runner_start_callback=self.request_runner_start, # Pass the callback method
            # api_key is no longer passed if not used by our FastAPI app
            # api_key=proxy_api_key,
        )
        # Connect signals from the proxy thread if needed (e.g., started, stopped, error)
        # self.fastapi_proxy_thread.started.connect(self.on_fastapi_proxy_started)
        # self.fastapi_proxy_thread.stopped.connect(self.on_fastapi_proxy_stopped)
        # self.fastapi_proxy_thread.error.connect(self.on_fastapi_proxy_error)

        # Connect MainWindow signals to proxy thread slots for bridging
        # Updated variable name
        self.runner_port_ready_for_proxy.connect(self.fastapi_proxy_thread.on_runner_port_ready)
        self.runner_stopped_for_proxy.connect(self.fastapi_proxy_thread.on_runner_stopped)


        # Updated variable name
        self.fastapi_proxy_thread.start()


    def start_ollama_proxy(self):
        """
        Starts the Ollama proxy in a separate thread.
        This is now called automatically in __init__.
        """
        if self.ollama_proxy_thread is not None and self.ollama_proxy_thread.isRunning():
            print("Ollama Proxy is already running.")
            return

        print("Starting Ollama Proxy...")

        # Pass the necessary callback methods to the proxy thread
        self.ollama_proxy_thread = OllamaProxyThread(
            all_models_config=self.models, # Pass the main models configuration
            runtimes_config=self.llama_runtimes, # Pass the runtimes configuration
            is_model_running_callback=self.is_llama_runner_running, # Pass the callback method
            get_runner_port_callback=self.get_runner_port, # Pass the callback method
            request_runner_start_callback=self.request_runner_start, # Pass the callback method
        )
        # Connect signals from the proxy thread if needed (e.g., started, stopped, error)
        # self.ollama_proxy_thread.started.connect(self.on_ollama_proxy_started)
        # self.ollama_proxy_thread.stopped.connect(self.on_ollama_proxy_stopped)
        # self.ollama_proxy_thread.error.connect(self.on_ollama_proxy_error)

        # Connect MainWindow signals to proxy thread slots for bridging
        self.runner_port_ready_for_proxy.connect(self.ollama_proxy_thread.on_runner_port_ready)
        self.runner_stopped_for_proxy.connect(self.ollama_proxy_thread.on_runner_stopped)

        self.ollama_proxy_thread.start()


    def stop_fastapi_proxy(self): # Updated method name
        """
        Stops the FastAPI proxy thread.
        This is now called automatically on closeEvent.
        """
        # Updated variable name
        if self.fastapi_proxy_thread and self.fastapi_proxy_thread.isRunning():
            print("Stopping FastAPI Proxy...") # Updated print
            # Updated status label variable name
            # Proxy buttons are removed

            # Updated variable name
            self.fastapi_proxy_thread.stop()
            # Don't wait() here in the UI thread
            # The thread will eventually finish and the status will update (manually for now)
            # A signal from the proxy thread would be better here.
            # For now, manually update status after signaling stop
            # Updated status label variable name

        else:
            print("FastAPI Proxy is not running.") # Updated print
            # Updated status label variable name


    def stop_ollama_proxy(self):
        """
        Stops the Ollama proxy thread.
        This is now called automatically on closeEvent.
        """
        if self.ollama_proxy_thread and self.ollama_proxy_thread.isRunning():
            print("Stopping Ollama Proxy...")

            self.ollama_proxy_thread.stop()
            # Don't wait() here in the UI thread
            # The thread will eventually finish and the status will update (manually for now)
            # A signal from the proxy thread would be better here.
            # For now, manually update status after signaling stop

        else:
            print("Ollama Proxy is not running.")


    @Slot(str)
    def on_llama_runner_started(self, model_name: str):
        """
        Slot to handle the LlamaCppRunner started signal.
        Updates UI status. Port update happens in on_llama_runner_port_ready.
        """
        status_widget = self.model_status_widgets.get(model_name)
        if status_widget:
             status_widget.update_status("Starting...")
             status_widget.set_buttons_enabled(False, False) # Keep disabled until port is ready


    @Slot(str)
    def on_llama_runner_stopped(self, model_name: str):
        """
        Slot to handle the LlamaCppRunner stopped signal.
        Cleans up the thread reference and updates UI.
        Emits a signal for the proxy thread.
        """
        print(f"Llama Runner for {model_name} stopped.")
        if model_name in self.llama_runner_threads:
            # Clean up the thread object
            thread = self.llama_runner_threads.pop(model_name)
            thread.deleteLater() # Schedule for deletion

            # Update UI for this model
            status_widget = self.model_status_widgets.get(model_name)
            if status_widget:
                status_widget.update_status("Not Running")
                status_widget.update_port("N/A")
                status_widget.set_buttons_enabled(True, False) # Enable start, disable stop

            # Emit signal for the proxy thread
            self.runner_stopped_for_proxy.emit(model_name)

            # If there's a pending Future for this runner, set an exception
            if model_name in self._runner_startup_futures and not self._runner_startup_futures[model_name].done():
                 logging.debug(f"Runner {model_name} stopped unexpectedly while startup Future was pending.")
                 self._runner_startup_futures[model_name].set_exception(RuntimeError(f"Runner for {model_name} stopped unexpectedly during startup."))
                 # The future will be cleaned up by _cleanup_completed_future after a delay

        else:
            print(f"Stopped signal received for unknown or already cleaned up model: {model_name}")


    @Slot(str, str, list) # Slot receives model_name, message, and output_buffer
    def on_llama_runner_error(self, model_name: str, message: str, output_buffer: list):
        """
        Slot to handle the LlamaCppRunner error signal.
        Displays an error message and updates UI status.
        """
        print(f"Llama Runner for {model_name} error: {message}")

        # Create and show the custom error dialog
        dialog_message = f"Llama.cpp server for {model_name} encountered an error:\n{message}"
        error_dialog = ErrorOutputDialog(
            title=f"Llama Runner Error: {model_name}",
            message=dialog_message,
            output_lines=output_buffer,
            parent=self # Set parent to MainWindow
        )
        error_dialog.exec() # Show the dialog modally

        # Update UI status for this model
        status_widget = self.model_status_widgets.get(model_name)
        if status_widget:
            status_widget.update_status("Error")
            # The stopped signal will follow and update the status to "Not Running" and re-enable the start button

        # If there's a pending Future for this runner, set an exception
        if model_name in self._runner_startup_futures and not self._runner_startup_futures[model_name].done():
             logging.debug(f"Runner {model_name} errored while startup Future was pending.")
             self._runner_startup_futures[model_name].set_exception(RuntimeError(f"Runner for {model_name} errored during startup: {message}"))
             # The future will be cleaned up by _cleanup_completed_future after a delay


    @Slot(str, int) # Slot receives model_name and port
    def on_llama_runner_port_ready_and_emit(self, model_name: str, port: int):
        """
        Slot to handle the LlamaCppRunner port ready signal.
        Updates the UI with the assigned port and status.
        Also resolves the corresponding Future and emits a signal for the proxy thread.
        """
        print(f"Llama Runner for {model_name} ready on port {port}.")
        status_widget = self.model_status_widgets.get(model_name)
        if status_widget:
            status_widget.update_port(port)
            status_widget.update_status("Running")
            status_widget.set_buttons_enabled(False, True) # Disable start, enable stop

        # Resolve the corresponding Future
        if model_name in self._runner_startup_futures and not self._runner_startup_futures[model_name].done():
             logging.debug(f"Resolving runner_startup_future for {model_name} with port {port}")
             self._runner_startup_futures[model_name].set_result(port)
             # The future will be cleaned up by _cleanup_completed_future after a delay
        elif model_name in self._runner_startup_futures and self._runner_startup_futures[model_name].done():
             logging.warning(f"Runner_port_ready signal received for {model_name}, but Future was already done.")
        else:
             logging.warning(f"Runner_port_ready signal received for {model_name}, but no pending Future found.")


        # Emit signal for the proxy thread
        self.runner_port_ready_for_proxy.emit(model_name, port)