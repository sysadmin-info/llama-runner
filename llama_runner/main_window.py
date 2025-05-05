import sys
import asyncio
import subprocess
import os
import tempfile
import yaml
import logging

from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel,
                               QPushButton, QLineEdit, QTabWidget)
from PySide6.QtCore import QThread, QObject, pyqtSignal, Slot

from llama_runner.config_loader import load_config
from llama_runner.llama_cpp_runner import LlamaCppRunner

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LlamaRunnerThread(QThread):
    """
    QThread to run the LlamaCppRunner in a separate thread to avoid blocking the UI.
    """
    def __init__(self, model_name: str, model_path: str, llama_cpp_runtime: str = None, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.model_path = model_path
        self.llama_cpp_runtime = llama_cpp_runtime
        self.kwargs = kwargs
        self.runner = None
        self.is_running = False

    def run(self):
        """
        Runs the LlamaCppRunner in the thread.
        """
        self.is_running = True
        asyncio.run(self.run_async())
        self.is_running = False

    async def run_async(self):
        """
        Asynchronous part of the runner.
        """
        self.runner = LlamaCppRunner(
            model_name=self.model_name,
            model_path=self.model_path,
            llama_cpp_runtime=self.llama_cpp_runtime,
            **self.kwargs
        )
        await self.runner.start()
        # Keep the runner alive until the thread is stopped
        while self.is_running:
            await asyncio.sleep(1)
        await self.runner.stop()

    def stop(self):
        """
        Stops the LlamaCppRunner.
        """
        self.is_running = False

class LiteLLMProxyThread(QThread):
    """
    QThread to run the LiteLLM proxy in a separate thread.
    """
    def __init__(self):
        super().__init__()
        self.process = None
        self.is_running = False

    def run(self):
        """
        Runs the LiteLLM proxy in the thread.
        """
        self.is_running = True
        asyncio.run(self.run_async())
        self.is_running = False

    async def run_async(self):
        """
        Asynchronous part of the proxy runner.
        """
        # TODO: Implement the LiteLLM proxy startup logic here
        # This is a placeholder implementation
        print("Starting LiteLLM Proxy...")
        try:
            # Example: Run a simple command
            self.process = subprocess.Popen(
                ["echo", "LiteLLM Proxy is running"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            while self.is_running:
                if self.process.poll() is not None:
                    break
                await asyncio.sleep(1)
        except Exception as e:
            print(f"Error starting LiteLLM Proxy: {e}")
        finally:
            if self.process:
                self.process.terminate()
                self.process.wait()
            print("LiteLLM Proxy stopped.")

    def stop(self):
        """
        Stops the LiteLLM proxy.
        """
        self.is_running = False
        if self.process:
            self.process.terminate()
            self.process.wait()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Llama Runner")

        self.config = load_config()
        self.llama_runtimes = self.config.get("llama-runtimes", {})
        self.default_runtime = "llama-server"  # Default to llama-server from PATH
        self.model_name = "test-model"  # hardcoded for now
        self.model_config = self.config.get("models", {}).get(self.model_name, {})
        self.model_path = self.model_config.get("model_path")
        self.llama_cpp_runtime = self.llama_runtimes.get(self.model_config.get("llama_cpp_runtime", "default"), self.default_runtime)

        self.llama_runner_thread = None
        self.litellm_proxy_thread = None

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

        # Connect buttons to actions
        self.llama_start_button.clicked.connect(self.start_llama_runner)
        self.llama_stop_button.clicked.connect(self.stop_llama_runner)
        self.litellm_start_button.clicked.connect(self.start_litellm_proxy)
        self.litellm_stop_button.clicked.connect(self.stop_litellm_proxy)

    def start_llama_runner(self):
        """
        Starts the LlamaCppRunner in a separate thread.
        """
        if self.llama_runner_thread is None or not self.llama_runner_thread.isRunning():
            self.llama_runner_thread = LlamaRunnerThread(
                model_name=self.model_name,
                model_path=self.model_path,
                llama_cpp_runtime=self.llama_cpp_runtime,
                **self.model_config.get("parameters", {})
            )
            self.llama_runner_thread.start()
            self.llama_status_label.setText("Running...")
        else:
            print("Llama Runner is already running.")

    def stop_llama_runner(self):
        """
        Stops the LlamaCppRunner thread.
        """
        if self.llama_runner_thread and self.llama_runner_thread.isRunning():
            self.llama_runner_thread.stop()
            self.llama_runner_thread.wait()  # Wait for the thread to finish
            self.llama_status_label.setText("Not Running")
        else:
            print("Llama Runner is not running.")

    def start_litellm_proxy(self):
        """
        Starts the LiteLLM proxy in a separate thread.
        """
        if self.litellm_proxy_thread is None or not self.litellm_proxy_thread.isRunning():
            self.litellm_proxy_thread = LiteLLMProxyThread()
            self.litellm_proxy_thread.start()
            self.litellm_status_label.setText("Running...")
        else:
            print("LiteLLM Proxy is already running.")

    def stop_litellm_proxy(self):
        """
        Stops the LiteLLM proxy thread.
        """
        if self.litellm_proxy_thread and self.litellm_proxy_thread.isRunning():
            self.litellm_proxy_thread.stop()
            self.litellm_proxy_thread.wait()  # Wait for the thread to finish
            self.litellm_status_label.setText("Not Running")
        else:
            print("LiteLLM Proxy is not running.")

# Here's the first *SEARCH/REPLACE* block to modify the `main.py` file:
#
# main.py
# <source>python
# <<<<<<< SEARCH
# import asyncio
# import os
# import tempfile
# import yaml
# import logging
# from litellm.proxy.proxy_server import app
# import uvicorn
#
# from llama_runner.config_loader import load_config
# from llama_runner.llama_cpp_runner import LlamaCppRunner, CONFIG_DIR
# from llama_runner.main_window import MainWindow  # Import the MainWindow class
#
#
# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#
# # The main function now primarily sets up the UI.
# # Async operations will be managed within the MainWindow class.
# def main():
#     # Load configuration (might be needed by the UI later)
#     # config = load_config()
#     # llama_runtimes = config.get("llama-runtimes", {})
#     # default_runtime = "llama-server"  # Default to llama-server from PATH
#     # model_name = "test-model"  # hardcoded for now
#     # model_config = config.get("models", {}).get(model_name, {})
#     # model_path = model_config.get("model_path")
#     # llama_cpp_runtime = llama_runtimes.get(model_config.get("llama_cpp_runtime", "default"), default_runtime)
#
#     # Create the LlamaCppRunner instance (will be managed by the UI later)
#     # runner = LlamaCppRunner(
#     #     model_name=model_name,
#     #     model_path=model_path,
#     #     llama_cpp_runtime=llama_cpp_runtime,
#     #     **model_config.get("parameters", {})
#     # )
#
#     # await runner.start() # Will be managed by the UI later
#
#     # 1. Define your proxy config as a Python dict (will be managed by the UI later)
#     # proxy_config = {
#     #     "model_list": [
#     #         {
#     #             "model_name": "gpt-3.5-turbo",
#     #             "litellm_params": {
#     #                 "model": model_name,
#     #                 "api_key": "os.environ/OPENAI_API_KEY",
#     #                 "custom_llm_provider": "llama_cpp",
#     #                 "api_base": f"http://127.0.0.1:{runner.get_port()}" # Needs runner instance
#     #             }
#     #         }
#     #     ],
#     #     "general_settings": {
#     #         "master_key": "sk-xxx"
#     #     }
#     # }
#
#     # 2. Dump to a temp YAML file (will be managed by the UI later)
#     # with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as f:
#     #     yaml.dump(proxy_config, f)
#     #     tmp_path = f.name
#
#     # 3. Point LiteLLM at that file (will be managed by the UI later)
#     # os.environ["CONFIG_FILE_PATH"] = tmp_path
#
#     # 4. Start the proxy embedded via Uvicorn (will be managed by the UI later)
#     # uvicorn_config = uvicorn.Config(app, host="0.0.0.0", port=4000, reload=True)
#     # server = uvicorn.Server(uvicorn_config)
#     # await server.serve() # This needs to run in an async context, likely managed by the UI
#
#     # await runner.stop() # Will be managed by the UI later
#
#
#     # Create the Qt application
#     qt_app = QApplication(sys.argv)
#
#     # Create the main window
#     # The MainWindow will need access to config and potentially create runners/servers
#     window = MainWindow()
#     window.show()
#
#     # Start the event loop
#     sys.exit(qt_app.exec())
#
#
# if __name__ == "__main__":
#     # The main function now runs the Qt event loop synchronously.
#     # Async tasks will be handled within the Qt application's lifecycle.
#     main()
# </source>
</code>

main.py
<code>python
<<<<<<< SEARCH
import sys
from PySide6.QtWidgets import QApplication
from llama_runner.main_window import MainWindow

def main():
    # Create the Qt application
    app = QApplication(sys.argv)

    # Create the main window
    window = MainWindow()
    window.show()

    # Start the event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
</python>

Next, I will modify the `main.py` file to:

1.  Import the `MainWindow` class from `llama_runner/main_window.py`.
2.  Create an instance of the `QApplication` and `MainWindow`.
3.  Show the main window.
4.  Move the LlamaCppRunner and LiteLLM proxy logic to be managed by the UI.

Here's the first *SEARCH/REPLACE* block to modify the `main.py` file:

main.py
<source>python
<<<<<<< SEARCH
import asyncio
import os
import tempfile
import yaml
import logging
from litellm.proxy.proxy_server import app
import uvicorn

from llama_runner.config_loader import load_config
from llama_runner.llama_cpp_runner import LlamaCppRunner, CONFIG_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def main():
    config = load_config()
    llama_runtimes = config.get("llama-runtimes", {})
    default_runtime = "llama-server"  # Default to llama-server from PATH
    model_name = "test-model" # hardcoded for now
    model_config = config.get("models", {}).get(model_name, {})
    model_path = model_config.get("model_path")
    llama_cpp_runtime = llama_runtimes.get(model_config.get("llama_cpp_runtime", "default"), default_runtime)

    runner = LlamaCppRunner(
        model_name=model_name,
        model_path=model_path,
        llama_cpp_runtime=llama_cpp_runtime,
        **model_config.get("parameters", {})
    )

    await runner.start()

    # 1. Define your proxy config as a Python dict
    proxy_config = {
        "model_list": [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": model_name,
                    "api_key": "os.environ/OPENAI_API_KEY",
                    "custom_llm_provider": "llama_cpp",
                    "api_base": f"http://127.0.0.1:{runner.get_port()}"
                }
            }
        ],
        "general_settings": {
            "master_key": "sk-xxx"
        }
    }

    # 2. Dump to a temp YAML file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as f:
        yaml.dump(proxy_config, f)
        tmp_path = f.name

    # 3. Point LiteLLM at that file
    os.environ["CONFIG_FILE_PATH"] = tmp_path

    # 4. Start the proxy embedded via Uvicorn
    uvicorn_config = uvicorn.Config(app, host="0.0.0.0", port=4000, reload=True)
    server = uvicorn.Server(uvicorn_config)
    await server.serve()

    await runner.stop()

if __name__ == "__main__":
    asyncio.run(main())
</source>
