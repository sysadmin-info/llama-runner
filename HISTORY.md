# Project History

1.  **Project Definition:** Defined the project goals in `OUTLINE.md`. The aim is to create a `llama.cpp` runner application with a configurable LiteLLM proxy interface and a PySide6 GUI for control and monitoring.
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
