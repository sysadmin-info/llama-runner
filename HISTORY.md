# Project History

1.  **Project Definition:** Defined the project goals in `OUTLINE.md`. The aim is to create a llama.cpp runner application with a configurable proxy interface to allow it to emulate other popular runners (mostly LM Studio and Ollama).
    The runner should dynamically destroy and create new llama.cpp instances based on the model selected by the end user and the configuration (although the options should allow for running more than one llama.cpp instance in parallel).
    To facilitate monitoring and control, we will implement a simple API using PySide6.
2.  **Configuration Loading:**
    *   Created `llama_runner/config_loader.py` to handle loading configuration from `~/.llama-runner/config.json`.
    *   Implemented functions `ensure_config_exists` and `load_config` with basic error handling and logging to `~/.llama-runner/error.log`.
    *   Created a sample `config.json` demonstrating the structure for `llama-runtimes` and `models`.
3.  **Llama.cpp Runner Implementation:**
    *   Created `llama_runner/llama_cpp_runner.py` containing the `LlamaCppRunner` class.
    *   Implemented asynchronous methods (`start`, `stop`, `wait_for_server_startup`) to manage the `llama-server` subprocess.
    *   The runner uses configuration loaded via `config_loader`, supports custom runtimes, passes parameters, detects server startup via stdout, and extracts the dynamically assigned port.
    *   Added process termination logic with a timeout before killing the process.
4.  **PySide6 UI Structure:**
    *   Created `llama_runner/main_window.py` with the main `MainWindow` class inheriting from `QWidget`.
    *   Set up a basic tabbed interface (`QTabWidget`) for "Llama Runner" and "LiteLLM Proxy".
    *   Added status labels (`QLabel`) and start/stop buttons (`QPushButton`) to each tab.
    *   Implemented `LlamaRunnerThread` and `LiteLLMProxyThread` (initially as a placeholder) inheriting from `QThread` to run backend tasks asynchronously without blocking the UI.
    *   Connected button signals (`clicked`) to slots (`start_llama_runner`, `stop_llama_runner`, etc.) in `MainWindow`.
5.  **UI Integration:**
    *   Modified `main.py` to initialize the PySide6 `QApplication` and the `MainWindow`.
    *   Refactored `main.py` to focus solely on launching the UI, removing direct runner/proxy startup logic from it. The UI components (`MainWindow` and its threads) are now responsible for managing these processes.
6.  **History Tracking:** Created this `HISTORY.md` file to document the development steps.
7.  **main.py Modification:**
    *   Ensured `main.py` initializes and runs the PySide6 application, creating the `QApplication` and `MainWindow`.
    *   The application starts the Qt event loop to display the UI.
8.  **UI Refactor and Concurrent Runners:**
    *   Refactored `llama_runner/main_window.py` to use a `QListWidget` for model selection and a `QStackedWidget` for model status display.
    *   Implemented logic in `start_llama_runner` to respect the `concurrentRunners` limit from `config.json`.
    *   Added logic to stop existing runners when the limit is 1 and a new runner is requested.
9.  **Wait for Runner Stop:**
    *   Modified `llama_runner/main_window.py` to add a 15-second wait period after signaling an existing runner to stop before attempting to start a new one, specifically when `concurrentRunners` is 1.
    *   Added error handling if the old runner fails to stop within the timeout.
10. **LMStudio Backend Support (Metadata & API):**
    *   Created `llama_runner/gguf_metadata.py` to extract and cache GGUF metadata using the `gguf` library.
    *   Moved `LiteLLMProxyThread` to `llama_runner/lite_llm_proxy_thread.py`.
    *   Modified `LiteLLMProxyThread` to use port 1234 and add FastAPI routes (`/api/v0/models`, `/api/v0/models/{model_id}`) to serve GGUF metadata in LM Studio format.
    *   Updated `llama_runner/main_window.py` to initialize the metadata cache and pass necessary data/callbacks to `LiteLLMProxyThread`.
    *   Added `gguf` and `httpx` to `requirements.txt`.
11. **Fix GGUF Metadata Extraction:**
    *   Corrected metadata extraction in `llama_runner/gguf_metadata.py` to use `reader.fields.items()` and access values via `field.parts[field.data[0]]`.
    *   Added `Optional` import to `llama_runner/main_window.py`.
    *   Fixed a syntax error in `llama_runner/lite_llm_proxy_thread.py`.
12. **Improve GGUF Metadata Robustness:**
    *   Added a helper function `get_scalar_metadata` in `llama_runner/gguf_metadata.py` to handle potential list/tuple/numpy array wrapping of scalar metadata values.
    *   Improved error logging with tracebacks in `gguf_metadata.py`.
13. **Refine GGUF Metadata Extraction:**
    *   Simplified `max_context_length` extraction in `llama_runner/gguf_metadata.py` to only use the `${general.architecture}.context_length` field.
14. **Configurable Logging:**
    *   Modified `main.py` to implement configurable logging levels via a `--log-level` command-line argument.
    *   Configured logging to output to both console (at the specified level) and a file (`app.log` at DEBUG level) in the config directory.
15. **Fix GGUF Quantization and Cache:**
    *   Corrected quantization extraction in `llama_runner/gguf_metadata.py` to use the `LlamaFileType` enum based on the `general.file_type` integer value.
    *   Changed the metadata cache key from file hash to file size for performance.
    *   Added debug logging for the raw `general.file_type` value.
16. **Fix LlamaFileType Import:**
    *   Corrected the import path for `LlamaFileType` in `llama_runner/gguf_metadata.py` from `gguf.gguf_reader` to `gguf.constants`.
17. **Clean Quantization String:**
    *   Added logic in `llama_runner/gguf_metadata.py` to remove the "MOSTLY\_" prefix from the quantization string.
18. **Auto-start Proxy and On-Demand Runners:**
    *   Modified `llama_runner/main_window.py` to automatically start the `LiteLLMProxyThread` in `__init__` and remove manual start/stop buttons.
    *   Added callback methods (`is_llama_runner_running`, `get_runner_port`, `request_runner_start`) in `main_window.py` to manage runners. `request_runner_start` now returns an `asyncio.Future`.
    *   Modified `llama_runner/lite_llm_proxy_thread.py` to accept these callbacks.
    *   Implemented dynamic routing in `lite_llm_proxy_thread.py` using `httpx` to intercept `/v1/*` requests, wait for the runner to start via the `asyncio.Future`, and forward the request.
    *   Added `qasync` to `requirements.txt` for bridging Qt and asyncio event loops.
19. **Proxy /api/v0/* Endpoints:**
    *   Modified `llama_runner/lite_llm_proxy_thread.py` to change the `/api/v0/*` handlers (excluding `/models`) from HTTP redirects to transparent proxies that call the `/v1/*` dynamic routing handler internally.
    *   Imported `json` in `lite_llm_proxy_thread.py`.
20. **Remove LiteLLM Dependency:**
    *   Removed `litellm[proxy]` from `requirements.txt`.
    *   Modified `llama_runner/lite_llm_proxy_thread.py` to create its own FastAPI app instance and remove LiteLLM config generation.
    *   Updated `OUTLINE.md` and `HISTORY.md`.
