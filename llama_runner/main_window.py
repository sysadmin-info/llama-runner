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
</code>

main.py
<code>python
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
