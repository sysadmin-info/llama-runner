import sys
import asyncio
import subprocess
import os
import tempfile
import yaml
import logging
import traceback # Import traceback for detailed error logging

from litellm.proxy.proxy_server import app
import uvicorn

from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel,
                               QPushButton, QLineEdit, QTabWidget, QMessageBox,
                               QDialog, QTextEdit, QDialogButtonBox) # Import necessary widgets for custom dialog
from PySide6.QtCore import QThread, QObject, Signal, Slot

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
        self.setMinimumWidth(400) # Give it a reasonable minimum width
        self.setMinimumHeight(200) # Give it a reasonable minimum height

        layout = QVBoxLayout()

        message_label = QLabel(message)
        layout.addWidget(message_label)

        if output_lines:
            output_text_edit = QTextEdit()
            output_text_edit.setReadOnly(True)
            output_text_edit.setPlainText("\n".join(output_lines))
            output_text_edit.setMinimumHeight(100) # Ensure text box has some height
            output_text_edit.setSizePolicy(output_text_edit.sizePolicy().horizontalPolicy(), output_text_edit.sizePolicy().verticalPolicy()) # Allow vertical expansion

            layout.addWidget(QLabel("Last Output Lines:")) # Label for the text box
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
    # Modified error signal to include output buffer
    error = Signal(str, list)
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
            # The runner.start() method now handles the case where the process exits immediately
            # This check is for cases where the process starts but doesn't print the expected line
            if self.runner.port is None:
                 # The error message is now generic, the UI will show the buffer
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
            # Check self.runner and self.runner.process exists before accessing returncode
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
    def __init__(self, model_name: str, llama_cpp_port: int):
        super().__init__()
        self.model_name = model_name
        self.llama_cpp_port = llama_cpp_port
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
                        "model_name": "gpt-3.5-turbo", # This is the alias LiteLLM will use
                        "litellm_params": {
                            # Use 'openai' provider and point api_base to the llama.cpp server
                            "model": "openai/gpt-3.5-turbo", # Use openai provider, model name can be anything for openai-compatible
                            "api_base": f"http://127.0.0.1:{self.llama_cpp_port}"
                            # Removed "custom_llm_provider": "llama_cpp"
                        }
                    }
                ],
                "general_settings": {
                    "master_key": "sk-xxx" # Placeholder
                }
            }

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


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Llama Runner")
        self.resize(800, 600)  # Set initial window size

        self.config = load_config()
        self.llama_runtimes = self.config.get("llama-runtimes", {})
        self.default_runtime = "llama-server"  # Default to llama-server from PATH
        self.models = self.config.get("models", {})

        self.llama_runner_threads = {}  # Dictionary to store threads for each model
        self.litellm_proxy_thread = None

        self.layout = QVBoxLayout()

        self.tabs = QTabWidget()

        # Llama Runner Tab
        self.llama_tab = QWidget()
        self.llama_layout = QVBoxLayout()
        self.llama_layout.addWidget(QLabel("Llama Runners:"))

        self.model_buttons = {}
        self.model_status_labels = {}
        self.model_port_labels = {}

        # Create a layout for model buttons and status
        self.model_widgets_layout = QVBoxLayout()

        for model_name, model_config in self.models.items():
            model_button_layout = QVBoxLayout()
            model_label = QLabel(f"<b>{model_name}</b>:") # Make model name bold
            model_button_layout.addWidget(model_label)

            status_label = QLabel("Status: Not Running") # Add "Status: " prefix
            model_button_layout.addWidget(status_label)
            self.model_status_labels[model_name] = status_label

            port_label = QLabel("Port: N/A")
            model_button_layout.addWidget(port_label)
            self.model_port_labels[model_name] = port_label

            start_button = QPushButton(f"Start {model_name}")
            # Use a lambda to pass the model_name to the slot
            start_button.clicked.connect(lambda checked, name=model_name: self.start_llama_runner(name))
            model_button_layout.addWidget(start_button)
            self.model_buttons[model_name] = start_button

            # Add a spacer or separator between model sections if needed
            # self.model_widgets_layout.addLayout(model_button_layout)
            # self.model_widgets_layout.addStretch() # Add stretch between models

            # Add the model's layout to the main model widgets layout
            self.model_widgets_layout.addLayout(model_button_layout)


        self.stop_all_button = QPushButton("Stop All Llama Runners")
        self.stop_all_button.clicked.connect(self.stop_all_llama_runners)

        # Add the model widgets layout and the stop all button to the main llama layout
        self.llama_layout.addLayout(self.model_widgets_layout)
        self.llama_layout.addStretch()  # Add stretch to push buttons to the top
        self.llama_layout.addWidget(self.stop_all_button)

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
        self.litellm_layout.addStretch() # Add stretch to push buttons to the top
        self.litellm_tab.setLayout(self.litellm_layout)
        self.tabs.addTab(self.litellm_tab, "LiteLLM Proxy")

        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

        # Connect buttons to actions
        self.litellm_start_button.clicked.connect(self.start_litellm_proxy)
        self.litellm_stop_button.clicked.connect(self.stop_litellm_proxy)

    def start_llama_runner(self, model_name: str):
        """
        Starts the LlamaCppRunner for a specific model in a separate thread.
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found in config.")
            return

        # Stop any currently running instance of this model first
        if model_name in self.llama_runner_threads and self.llama_runner_threads[model_name].isRunning():
             print(f"Stopping existing Llama Runner for {model_name} before starting a new one.")
             self.stop_llama_runner(model_name)
             # Wait briefly for the stop to process signals if necessary
             QApplication.processEvents() # Process events to allow stop signals to be handled

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
        # Note: This check might be redundant now that LlamaCppRunner.start handles FileNotFoundError
        # but keeping it here provides a quicker UI feedback for common errors.
        if llama_cpp_runtime_key != "default" and not os.path.exists(llama_cpp_runtime):
             QMessageBox.critical(self, "Runtime Not Found", f"Llama.cpp runtime not found: {llama_cpp_runtime}")
             return


        print(f"Starting Llama Runner for {model_name}...")
        self.model_status_labels[model_name].setText("Status: Starting...")
        self.model_buttons[model_name].setEnabled(False)
        self.model_port_labels[model_name].setText("Port: N/A") # Clear old port

        thread = LlamaRunnerThread(
            model_name=model_name,
            model_path=model_path,
            llama_cpp_runtime=llama_cpp_runtime,
            **model_config.get("parameters", {})
        )
        # Connect signals using partial or lambda to pass model_name
        thread.started.connect(lambda: self.on_llama_runner_started(model_name))
        thread.stopped.connect(lambda: self.on_llama_runner_stopped(model_name))
        # Connect the error signal with the new signature
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
            self.model_status_labels[model_name].setText("Status: Stopping...")
            # Button should already be disabled, but ensure it is
            self.model_buttons[model_name].setEnabled(False)
            # Port label might still show the port until stopped signal
            # self.model_port_labels[model_name].setText("Port: N/A") # Clear port immediately? Or wait for stopped?

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


        self.litellm_proxy_thread = LiteLLMProxyThread(model_name=running_model_name, llama_cpp_port=running_port)
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
        """
        # Status is updated by port_ready, but we can ensure button is disabled
        if model_name in self.model_buttons:
             self.model_buttons[model_name].setEnabled(False)
             self.model_status_labels[model_name].setText("Status: Starting...") # Update status to Starting if not already


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
            if model_name in self.model_status_labels:
                self.model_status_labels[model_name].setText("Status: Not Running")
            if model_name in self.model_buttons:
                self.model_buttons[model_name].setEnabled(True)
            if model_name in self.model_port_labels:
                self.model_port_labels[model_name].setText("Port: N/A")
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
        if model_name in self.model_status_labels:
            self.model_status_labels[model_name].setText("Status: Error")
        # The stopped signal will follow and update the status to "Not Running" and re-enable the button


    @Slot(str, int) # Slot receives model_name and port
    def on_llama_runner_port_ready(self, model_name: str, port: int):
        """
        Slot to handle the LlamaCppRunner port ready signal.
        Updates the UI with the assigned port and status.
        """
        print(f"Llama Runner for {model_name} ready on port {port}.")
        if model_name in self.model_port_labels:
            self.model_port_labels[model_name].setText(f"Port: {port}")
        if model_name in self.model_status_labels:
            self.model_status_labels[model_name].setText("Status: Running")
        # Button should already be disabled


# The main function is now in main.py
# async def main():
#    pass # REMOVE

# if __name__ == "__main__":
#    pass # REMOVE
