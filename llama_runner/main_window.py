import sys
import asyncio
import subprocess
import os
import tempfile
import yaml
import logging
import traceback
import time # Import time for sleep

from litellm.proxy.proxy_server import app
import uvicorn

from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QLineEdit, QTabWidget, QMessageBox,
                               QDialog, QTextEdit, QDialogButtonBox, QListWidget,
                               QStackedWidget, QSizePolicy, QSpacerItem)
from PySide6.QtCore import QThread, QObject, Signal, Slot, Qt

from llama_runner.config_loader import load_config
from llama_runner.llama_cpp_runner import LlamaCppRunner

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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


class LlamaRunnerThread(QThread):
    """
    QThread to run the LlamaCppRunner in a separate thread to avoid blocking the UI.
    """
    started = Signal()
    stopped = Signal()
    error = Signal(str, list) # Signal includes error message and output buffer
    port_ready = Signal(int)

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
            self.port_ready.emit(self.runner.get_port()) # Emit the port number

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


class LiteLLMProxyThread(QThread):
    """
    QThread to run the LiteLLM proxy in a separate thread.
    """
    def __init__(self, model_name: str, llama_cpp_port: int, api_key: str = None):
        super().__init__()
        self.model_name = model_name
        self.llama_cpp_port = llama_cpp_port
        self.api_key = api_key # Store the optional API key for proxy authentication
        self.process = None # LiteLLM proxy is run embedded, not as a separate process
        self.is_running = False
        self._uvicorn_server = None # Hold reference to the uvicorn server

    def run(self):
        """
        Runs the LiteLLM proxy in the thread.
        """
        self.is_running = True
        try:
            # Use a new event loop for this thread
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.run_async())
        except Exception as e:
            logging.error(f"Unexpected error in LiteLLMProxyThread run: {e}\n{traceback.format_exc()}")
        finally:
            self.is_running = False
            if hasattr(self, 'loop') and self.loop.is_running():
                 self.loop.stop()
            if hasattr(self, 'loop') and not self.loop.is_closed():
                 self.loop.close()


    async def run_async(self):
        """
        Asynchronous part of the proxy runner.
        """
        print("Starting LiteLLM Proxy...")
        try:
            # 1. Define your proxy config as a Python dict
            proxy_config = {
                "model_list": [
                    {
                        # This is the alias clients will use to refer to the model
                        "model_name": self.model_name,
                        "litellm_params": {
                            # Use 'openai' provider and point api_base to the llama.cpp server
                            # The model name here should be openai/<model_id>.
                            # The model_id can be anything, using the actual model name is clear.
                            "model": f"openai/{self.model_name}",
                            "api_base": f"http://127.0.0.1:{self.llama_cpp_port}",
                            # Add a dummy API key for the openai provider, as LiteLLM seems to require it
                            "api_key": "sk-dummy"
                        }
                    }
                ],
                "general_settings": {
                    # Include master_key only if an API key was provided in config.json
                }
            }

            if self.api_key:
                 proxy_config["general_settings"]["master_key"] = self.api_key
                 print("LiteLLM Proxy configured with API Key authentication.")
            else:
                 print("LiteLLM Proxy configured without API Key authentication.")


            # 2. Dump to a temp YAML file
            # Use a persistent temp file that we can clean up later
            with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode='w') as f:
                yaml.dump(proxy_config, f)
                tmp_path = f.name
            logging.info(f"LiteLLM proxy config written to {tmp_path}")

            # 3. Point LiteLLM at that file
            os.environ["CONFIG_FILE_PATH"] = tmp_path

            # 4. Start the proxy embedded via Uvicorn
            # We need to run uvicorn's serve() method in the asyncio loop
            uvicorn_config = uvicorn.Config(app, host="0.0.0.0", port=4000, reload=False) # reload=False in embedded mode
            self._uvicorn_server = uvicorn.Server(uvicorn_config)

            # This call is blocking until the server stops
            await self._uvicorn_server.serve()

            # Clean up the temporary config file after the server stops
            try:
                os.unlink(tmp_path)
                logging.info(f"Cleaned up temporary LiteLLM config file: {tmp_path}")
            except OSError as e:
                logging.error(f"Error cleaning up temporary LiteLLM config file {tmp_path}: {e}")


        except Exception as e:
            print(f"Error starting LiteLLM Proxy: {e}")
            logging.error(f"Error starting LiteLLM Proxy: {e}\n{traceback.format_exc()}")
        finally:
            # The uvicorn server.serve() call blocks until it's stopped,
            # so cleanup happens after it returns.
            print("LiteLLM Proxy stopped.")


    def stop(self):
        """
        Signals the LiteLLM proxy thread to stop.
        """
        self.is_running = False
        if self._uvicorn_server:
            # This will cause the server.serve() await call to return
            self._uvicorn_server.should_exit = True
            # Note: Stopping uvicorn gracefully can be tricky in an embedded context.
            # This might not immediately stop the serve() call.
            # A more robust stop might involve sending a signal or using a shutdown event.
            # For now, setting should_exit is the standard uvicorn way.


class ModelStatusWidget(QWidget):
    """
    Widget to display status and controls for a single model.
    """
    def __init__(self, model_name: str, parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.layout = QVBoxLayout()

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

    def update_status(self, status: str):
        self.status_label.setText(f"Status: {status}")

    def update_port(self, port: int | str):
        self.port_label.setText(f"Port: {port}")

    def set_buttons_enabled(self, start_enabled: bool, stop_enabled: bool):
        self.start_button.setEnabled(start_enabled)
        self.stop_button.setEnabled(stop_enabled)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Llama Runner")
        self.resize(800, 600)

        self.config = load_config()
        self.llama_runtimes = self.config.get("llama-runtimes", {})
        self.default_runtime = "llama-server"
        self.models = self.config.get("models", {})
        self.litellm_proxy_config = self.config.get("litellm-proxy", {})
        # Get concurrent runners limit from config, default to 1 if not specified or invalid
        self.concurrent_runners_limit = self.config.get("concurrentRunners", 1)
        if not isinstance(self.concurrent_runners_limit, int) or self.concurrent_runners_limit < 1:
             logging.warning(f"Invalid 'concurrentRunners' value in config: {self.concurrent_runners_limit}. Defaulting to 1.")
             self.concurrent_runners_limit = 1


        self.llama_runner_threads = {}  # Dictionary to store threads for each model
        self.litellm_proxy_thread = None

        self.layout = QVBoxLayout()

        self.tabs = QTabWidget()

        # Llama Runner Tab
        self.llama_tab = QWidget()
        self.llama_layout = QHBoxLayout() # Use QHBoxLayout for side-by-side layout

        # Left side: Model List
        self.model_list_widget = QListWidget()
        self.model_list_widget.setMinimumWidth(150) # Give the list some space
        self.llama_layout.addWidget(self.model_list_widget)

        # Right side: Model Status Stack
        self.model_status_stack = QStackedWidget()
        self.llama_layout.addWidget(self.model_status_stack)

        self.model_status_widgets = {} # Store status widgets by model name

        # Populate model list and create status widgets
        for model_name in self.models.keys():
            self.model_list_widget.addItem(model_name)

            status_widget = ModelStatusWidget(model_name)
            self.model_status_stack.addWidget(status_widget)
            self.model_status_widgets[model_name] = status_widget

            # Connect signals from the status widget's buttons
            # Use lambda to pass the model_name to the slots
            status_widget.start_button.clicked.connect(lambda checked, name=model_name: self.start_llama_runner(name))
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
        self.litellm_stop_button.setEnabled(False) # Initially disabled
        self.litellm_layout.addWidget(self.litellm_start_button)
        self.litellm_layout.addWidget(self.litellm_stop_button)
        self.litellm_layout.addStretch()
        self.litellm_tab.setLayout(self.litellm_layout)
        self.tabs.addTab(self.litellm_tab, "LiteLLM Proxy")

        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

        # Connect buttons to actions
        self.litellm_start_button.clicked.connect(self.start_litellm_proxy)
        self.litellm_stop_button.clicked.connect(self.stop_litellm_proxy)

        # Connect signals from LlamaRunnerThreads (will be connected when threads are created)
        # We need to connect these signals dynamically when a thread is started.


    def closeEvent(self, event):
        """
        Handles the window close event. Stops all running threads.
        """
        print("MainWindow closing. Stopping all runners and proxy...")
        self.stop_all_llama_runners()
        self.stop_litellm_proxy()

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


    def start_llama_runner(self, model_name: str):
        """
        Starts the LlamaCppRunner for a specific model in a separate thread.
        Handles concurrent runner limit.
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found in config.")
            return

        # Check current running runners
        running_runners = {name: thread for name, thread in self.llama_runner_threads.items() if thread.isRunning()}
        num_running = len(running_runners)

        # If the requested model is already running, do nothing
        if model_name in running_runners:
             print(f"Llama Runner for {model_name} is already running.")
             return

        # Check concurrent runner limit
        if num_running >= self.concurrent_runners_limit:
            if self.concurrent_runners_limit == 1:
                print(f"Concurrent runner limit ({self.concurrent_runners_limit}) reached. Stopping existing runner before starting {model_name}.")
                # Stop the first running runner found (or all if limit is 1)
                for name_to_stop in list(running_runners.keys()):
                     self.stop_llama_runner(name_to_stop)
                # Wait briefly for stop signals to process
                QApplication.processEvents()
                # Re-check if any are still running after attempting stop
                running_after_stop = {name: thread for name, thread in self.llama_runner_threads.items() if thread.isRunning()}
                if running_after_stop:
                     print("Warning: Could not stop existing runner(s). Cannot start new runner.")
                     QMessageBox.warning(self, "Concurrent Runner Limit",
                                         f"Concurrent runner limit ({self.concurrent_runners_limit}) reached and could not stop existing runner(s).")
                     return
            else:
                print(f"Concurrent runner limit ({self.concurrent_runners_limit}) reached. Cannot start {model_name}.")
                QMessageBox.warning(self, "Concurrent Runner Limit",
                                    f"Concurrent runner limit ({self.concurrent_runners_limit}) reached. Cannot start '{model_name}'.")
                return


        model_config = self.models[model_name]
        model_path = model_config.get("model_path")
        llama_cpp_runtime_key = model_config.get("llama_cpp_runtime", "default")
        llama_cpp_runtime = self.llama_runtimes.get(llama_cpp_runtime_key, self.default_runtime)

        if not model_path:
             QMessageBox.critical(self, "Configuration Error", f"Model '{model_name}' has no 'model_path' specified in config.json.")
             return

        # Check if the model file exists
        if not os.path.exists(model_path):
             QMessageBox.critical(self, "File Not Found", f"Model file not found: {model_path}")
             return

        # Check if the runtime executable exists if it's not the default "llama-server" (which is expected in PATH)
        if llama_cpp_runtime_key != "default" and not os.path.exists(llama_cpp_runtime):
             QMessageBox.critical(self, "Runtime Not Found", f"Llama.cpp runtime not found: {llama_cpp_runtime}")
             return


        print(f"Starting Llama Runner for {model_name}...")
        status_widget = self.model_status_widgets.get(model_name)
        if status_widget:
            status_widget.update_status("Starting...")
            status_widget.update_port("N/A")
            status_widget.set_buttons_enabled(False, False) # Disable both while starting

        thread = LlamaRunnerThread(
            model_name=model_name,
            model_path=model_path,
            llama_cpp_runtime=llama_cpp_runtime,
            **model_config.get("parameters", {})
        )
        # Connect signals
        thread.started.connect(lambda: self.on_llama_runner_started(model_name))
        thread.stopped.connect(lambda: self.on_llama_runner_stopped(model_name))
        thread.error.connect(lambda msg, output: self.on_llama_runner_error(model_name, msg, output))
        thread.port_ready.connect(lambda port: self.on_llama_runner_port_ready(model_name, port))

        self.llama_runner_threads[model_name] = thread
        thread.start()


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

    def start_litellm_proxy(self):
        """
        Starts the LiteLLM proxy in a separate thread.
        """
        if self.litellm_proxy_thread is not None and self.litellm_proxy_thread.isRunning():
            print("LiteLLM Proxy is already running.")
            return

        # Find the first running Llama Runner to connect to
        running_model_name = None
        running_port = None
        # Iterate over a copy of items to avoid issues if a thread stops during iteration
        for model_name, thread in list(self.llama_runner_threads.items()):
            # Check if the thread is running AND the runner instance exists AND has a port
            # Also check if the underlying process is still alive
            if thread.isRunning() and thread.runner is not None and thread.runner.is_running() and thread.runner.get_port() is not None:
                running_model_name = model_name
                running_port = thread.runner.get_port()
                break

        if not running_model_name or running_port is None:
            QMessageBox.warning(self, "LiteLLM Proxy", "No Llama Runner is currently running or port not available. Start one first.")
            print("No Llama Runner is running or port not available. Cannot start LiteLLM Proxy.")
            return

        print(f"Starting LiteLLM Proxy for model '{running_model_name}' on port {running_port}...")
        self.litellm_status_label.setText("Running...")
        self.litellm_start_button.setEnabled(False)
        self.litellm_stop_button.setEnabled(True)

        # Get the optional API key from the config
        proxy_api_key = self.litellm_proxy_config.get("api_key")

        self.litellm_proxy_thread = LiteLLMProxyThread(
            model_name=running_model_name,
            llama_cpp_port=running_port,
            api_key=proxy_api_key # Pass the API key to the thread
        )
        # LiteLLM proxy thread doesn't currently have error/stopped signals, could add later
        self.litellm_proxy_thread.start()


    def stop_litellm_proxy(self):
        """
        Stops the LiteLLM proxy thread.
        """
        if self.litellm_proxy_thread and self.litellm_proxy_thread.isRunning():
            print("Stopping LiteLLM Proxy...")
            self.litellm_status_label.setText("Stopping...")
            self.litellm_start_button.setEnabled(False)
            self.litellm_stop_button.setEnabled(False)

            self.litellm_proxy_thread.stop()
            # Don't wait() here in the UI thread
            # The thread will eventually finish and the status will update (manually for now)
            # A signal from the proxy thread would be better here.
            # For now, manually update status after signaling stop
            self.litellm_status_label.setText("Not Running")
            self.litellm_start_button.setEnabled(True)
            self.litellm_stop_button.setEnabled(False)

        else:
            print("LiteLLM Proxy is not running.")
            self.litellm_status_label.setText("Not Running")
            self.litellm_start_button.setEnabled(True)
            self.litellm_stop_button.setEnabled(False)


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


    @Slot(str, int) # Slot receives model_name and port
    def on_llama_runner_port_ready(self, model_name: str, port: int):
        """
        Slot to handle the LlamaCppRunner port ready signal.
        Updates the UI with the assigned port and status.
        """
        print(f"Llama Runner for {model_name} ready on port {port}.")
        status_widget = self.model_status_widgets.get(model_name)
        if status_widget:
            status_widget.update_port(port)
            status_widget.update_status("Running")
            status_widget.set_buttons_enabled(False, True) # Disable start, enable stop


# The main function is now in main.py
# async def main():
#    pass # REMOVE

# if __name__ == "__main__":
#    pass # REMOVE
