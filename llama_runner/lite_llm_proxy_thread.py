import sys
import asyncio
import subprocess
import os
import tempfile
import yaml
import logging
import traceback
import time
from typing import Dict, Any, Callable, List, Optional

from litellm.proxy.proxy_server import app # Import the global FastAPI app instance
import uvicorn
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from PySide6.QtCore import QThread, QObject, Signal, Slot

from llama_runner import gguf_metadata # Import the new metadata module

# Configure logging (already done in main_window, but good practice here too)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LiteLLMProxyThread(QThread):
    """
    QThread to run the LiteLLM proxy in a separate thread.
    """
    # Define signals if needed (e.g., started, stopped, error)
    # started = Signal()
    # stopped = Signal()
    # error = Signal(str)

    def __init__(self, models_config: Dict[str, Dict[str, Any]], is_model_running_callback: Callable[[str], bool], api_key: str = None):
        super().__init__()
        self.models_config = models_config
        self.is_model_running_callback = is_model_running_callback
        self.api_key = api_key # Store the optional API key for proxy authentication
        self.is_running = False
        self._uvicorn_server = None # Hold reference to the uvicorn server
        self._temp_config_path = None # To store the path of the temporary config file

        # Add LM Studio compatible routes to the global FastAPI app instance
        # We need to do this before uvicorn starts serving the app.
        # These routes need access to the thread's state (models_config, is_model_running_callback)
        # This is a bit tricky with a global app instance.
        # A simple way is to make the necessary data accessible globally or via a singleton,
        # but passing it to the thread is cleaner.
        # Let's define the handlers here and add them to the global 'app'.
        # This assumes only one LiteLLMProxyThread instance is ever running,
        # which is true for this application's design.

        # Check if routes already exist to avoid adding them multiple times
        # This is a simple check and might not be robust against all changes
        existing_routes = [route.path for route in app.routes]
        if "/api/v0/models" not in existing_routes:
             logging.info("Adding LM Studio compatible API routes.")
             # Add the routes using the global 'app' instance
             app.add_api_route("/api/v0/models", self._get_lmstudio_models, methods=["GET"])
             app.add_api_route("/api/v0/models/{model_id}", self._get_lmstudio_model, methods=["GET"])
             # Add redirects (LiteLLM might handle these by default, but explicit is safer)
             # Note: Adding redirects might require custom middleware or more advanced FastAPI techniques
             # For now, let's rely on LiteLLM's default behavior for /v0 -> /v1 if it exists,
             # and focus on the /api/v0/models endpoints which definitely need custom handling.
             # If LiteLLM doesn't redirect /v0 automatically, we'd need to add more routes/middleware.
             # Example redirect (requires FastAPI's RedirectResponse):
             # from fastapi.responses import RedirectResponse
             # app.add_api_route("/api/v0/chat/completions", lambda: RedirectResponse(url="/v1/chat/completions"), methods=["POST"])
             # ... and so on for other endpoints. Let's skip explicit redirects for now.
        else:
             logging.info("LM Studio compatible API routes already exist.")


    # Define the route handlers as methods within the thread class
    # These methods will be called by FastAPI when the routes are hit.
    # They need access to self.models_config and self.is_model_running_callback.
    # FastAPI route handlers typically expect (request: Request) or specific path/query params.
    # We can't directly pass 'self' to the route handler function signature when adding it to 'app'.
    # A common pattern is to store the necessary state in a way accessible to the handler,
    # or use dependency injection if the app structure was set up for it.
    # Given the global 'app' import, the simplest (though not cleanest) way is to
    # make the thread instance or its state globally accessible *during the request handling*.
    # This is problematic.

    # Let's rethink how the handlers access state.
    # Option 1: Pass state via `app.state` or similar (FastAPI feature).
    # Option 2: Use a global variable (bad practice).
    # Option 3: Modify the handler function closure (complex).
    # Option 4: Have the LiteLLMProxyThread *create* the FastAPI app instance instead of importing a global one.
    # This would be cleaner but might conflict with how LiteLLM's proxy_server.app is intended to be used.

    # Let's try Option 1: Use app.state. We'll set the state just before starting uvicorn.

    async def _get_lmstudio_models(self, request: Request):
        """Handler for GET /api/v0/models"""
        # Access state from the request's app instance
        models_config = request.app.state.models_config
        is_model_running_callback = request.app.state.is_model_running_callback

        if not gguf_metadata.GGUF_AVAILABLE:
             return JSONResponse(content={"error": "GGUF library not available for metadata extraction."}, status_code=500)

        try:
            all_models_data = gguf_metadata.get_all_models_lmstudio_format(
                models_config, is_model_running_callback
            )
            return JSONResponse(content={
                "object": "list",
                "data": all_models_data
            })
        except Exception as e:
            logging.error(f"Error handling /api/v0/models: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail="Internal Server Error retrieving models metadata")


    async def _get_lmstudio_model(self, model_id: str, request: Request):
        """Handler for GET /api/v0/models/{model_id}"""
        # Access state from the request's app instance
        models_config = request.app.state.models_config
        is_model_running_callback = request.app.state.is_model_running_callback

        if not gguf_metadata.GGUF_AVAILABLE:
             raise HTTPException(status_code=500, detail="GGUF library not available for metadata extraction.")

        try:
            # Find the model in the config by its LM Studio ID (which is the model_name from config)
            model_data = gguf_metadata.get_single_model_lmstudio_format(
                model_id, models_config, is_model_running_callback
            )

            if model_data:
                return JSONResponse(content=model_data)
            else:
                raise HTTPException(status_code=404, detail=f"Model with id '{model_id}' not found")

        except Exception as e:
            logging.error(f"Error handling /api/v0/models/{model_id}: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail="Internal Server Error retrieving model metadata")


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
            # Clean up the temporary config file
            if self._temp_config_path and os.path.exists(self._temp_config_path):
                 try:
                     os.unlink(self._temp_config_path)
                     logging.info(f"Cleaned up temporary LiteLLM config file: {self._temp_config_path}")
                 except OSError as e:
                     logging.error(f"Error cleaning up temporary LiteLLM config file {self._temp_config_path}: {e}")
            self._temp_config_path = None # Clear the path

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
            # We need to find the currently running model to configure LiteLLM's model_list
            # The LiteLLM proxy, when running, typically serves *one* backend model at a time
            # unless configured for multiple backends. For LM Studio emulation, it usually
            # points to the *currently active* model.
            # We need to find which model is running. The MainWindow passes the running state callback.
            # However, the LiteLLM config is generated *before* we know which model the user
            # will start.
            # Let's assume for now that the LiteLLM proxy will be configured to point to
            # the *first* model started via the UI. This is a limitation of the current design.
            # A better approach might be to dynamically update the LiteLLM config or restart
            # the proxy when the active model changes, but that's more complex.

            # For simplicity, let's configure LiteLLM to point to the *first* model in the config
            # that is currently running. If none are running, the proxy might not be fully functional
            # for chat/completion, but the /v0/models endpoint should still work.

            running_model_name = None
            running_model_config = None
            running_port = None

            # Find the currently running model and its port
            for model_name, model_config in self.models_config.items():
                 if self.is_model_running_callback(model_name):
                     # We need the port of the running model. The callback only returns True/False.
                     # The MainWindow holds the threads and their ports.
                     # This state dependency is awkward.
                     # Let's modify the MainWindow to pass the *port* of the running model
                     # when starting the proxy, or pass a callback that returns the port.
                     # For now, let's assume the MainWindow will ensure a model is running
                     # and pass its port when starting the proxy.
                     # This means the LiteLLMProxyThread needs a `running_model_name` and `running_port`
                     # passed to its __init__ or start method.

                     # Let's update the __init__ signature to accept running_model_name and running_port
                     # and update MainWindow accordingly.

                     # --- REVISING LiteLLMProxyThread __init__ and run_async ---
                     # The LiteLLM proxy needs to know which model is active to configure its `model_list`.
                     # The `/api/v0/models` endpoint needs the full list of *configured* models.
                     # These are two different pieces of information.
                     # The `/api/v0/models` endpoint logic is handled by the methods added to `app`.
                     # The `model_list` in the LiteLLM config should point to the *currently active* runner.
                     # This implies the LiteLLM proxy might need to be restarted or reconfigured
                     # when the active runner changes.

                     # Let's simplify: The LiteLLM proxy will *always* be configured to point to
                     # the runner that was active *when the proxy was started*.
                     # If the active runner changes, the proxy needs to be stopped and restarted.
                     # This makes the UI flow: Start Model -> Start Proxy (points to that model).
                     # If user starts a different model, they must Stop Proxy -> Stop Old Model -> Start New Model -> Start Proxy.
                     # This is not ideal for the "switch runners" requirement.

                     # Let's go back to the idea that the proxy can serve *any* configured model,
                     # but the `model_list` in the config needs to reflect the available models.
                     # LiteLLM can be configured with multiple backends in `model_list`.
                     # We can list *all* configured models in the LiteLLM `model_list`,
                     # pointing each one to its expected `api_base` (which is the llama.cpp port).
                     # The challenge is the port is dynamic.

                     # Let's try this: The LiteLLM config will list ALL models from config.json.
                     # For each model, the `api_base` will be constructed using a placeholder port.
                     # The actual port mapping needs to happen dynamically, perhaps via a custom LiteLLM router
                     # or by having the `/v1/chat/completions` endpoint handler in LiteLLM
                     # dynamically determine the correct running runner's port.
                     # This seems overly complex.

                     # Let's return to the simplest approach that meets the requirements:
                     # The LiteLLM proxy, when started, will connect to the *first* running Llama runner it finds.
                     # The `/api/v0/models` endpoint will list *all* configured models,
                     # indicating their state (loaded/not-loaded) based on the `is_model_running_callback`.

                     # --- Back to the original plan for run_async ---
                     # Find the currently running model and its port to configure LiteLLM's model_list
                     # This requires the port to be available. The MainWindow knows the ports.
                     # The MainWindow must pass the *port* of the model the proxy should connect to.
                     # Let's add `llama_cpp_port` to the `LiteLLMProxyThread.__init__`.

                     # --- REVISED LiteLLMProxyThread __init__ and run_async again ---
                     # __init__ will take `models_config`, `is_model_running_callback`, `api_key`,
                     # and `active_model_name: str`, `active_model_port: int`.
                     # The `active_model_name` and `active_model_port` are for the LiteLLM `model_list`.
                     # The `models_config` and `is_model_running_callback` are for the /v0/models endpoint.

                     # --- Final Plan for LiteLLMProxyThread ---
                     # __init__ will take `models_config`, `is_model_running_callback`, `api_key`,
                     # `active_model_name: str`, `active_model_port: int`.
                     # run_async will configure LiteLLM's `model_list` using `active_model_name` and `active_model_port`.
                     # The /v0/models handlers will use `models_config` and `is_model_running_callback`.
                     # We need to set `app.state` *before* `uvicorn.Server` is created.

                     pass # Logic moved below

            # 1. Define your proxy config as a Python dict
            # This config is for LiteLLM's core functionality (which backend to use for /v1 calls)
            proxy_config = {
                "model_list": [], # Will add the active model here
                "general_settings": {
                    "master_key": self.api_key if self.api_key else None
                }
            }

            # Add the active model to the LiteLLM model_list
            # This requires the active model name and port to be passed to the thread.
            # Let's assume they are passed to __init__ as self.active_model_name and self.active_model_port
            if hasattr(self, 'active_model_name') and hasattr(self, 'active_model_port') and self.active_model_name and self.active_model_port:
                 proxy_config["model_list"].append({
                     "model_name": self.active_model_name,
                     "litellm_params": {
                         "model": f"openai/{self.active_model_name}", # Use openai provider pointing to llama.cpp
                         "api_base": f"http://127.0.0.1:{self.active_model_port}",
                         "api_key": "sk-dummy" # Dummy key required by LiteLLM for openai provider
                     }
                 })
                 print(f"LiteLLM Proxy configured for active model '{self.active_model_name}' on port {self.active_model_port}.")
            else:
                 print("LiteLLM Proxy started without an active Llama runner configured in model_list.")
                 # The proxy will start, but /v1 endpoints won't work until a model is configured/restarted.
                 # The /v0/models endpoint should still work.


            # 2. Dump to a temp YAML file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode='w') as f:
                yaml.dump(proxy_config, f)
                self._temp_config_path = f.name # Store path for cleanup
            logging.info(f"LiteLLM proxy config written to {self._temp_config_path}")

            # 3. Point LiteLLM at that file
            os.environ["CONFIG_FILE_PATH"] = self._temp_config_path

            # 4. Start the proxy embedded via Uvicorn
            # Set state on the app instance BEFORE creating the server
            app.state.models_config = self.models_config
            app.state.is_model_running_callback = self.is_model_running_callback

            # Use port 1234 as required
            uvicorn_config = uvicorn.Config(app, host="0.0.0.0", port=1234, reload=False)
            self._uvicorn_server = uvicorn.Server(uvicorn_config)

            print("LiteLLM Proxy listening on http://0.0.0.0:1234")

            # This call is blocking until the server stops
            await self._uvicorn_server.serve()

        except Exception as e:
            print(f"Error starting LiteLLM Proxy: {e}")
            logging.error(f"Error starting LiteLLM Proxy: {e}\n{traceback.format_exc()}")
            # Emit error signal if needed
            # self.error.emit(str(e))
        finally:
            # Cleanup happens in run()'s finally block now
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


# Update the __init__ signature to accept active model info
LiteLLMProxyThread.__init__ = lambda self, models_config, is_model_running_callback, api_key=None, active_model_name=None, active_model_port=None: (
    QThread.__init__(self),
    setattr(self, 'models_config', models_config),
    setattr(self, 'is_model_running_callback', is_model_running_callback),
    setattr(self, 'api_key', api_key),
    setattr(self, 'active_model_name', active_model_name),
    setattr(self, 'active_model_port', active_model_port),
    setattr(self, 'is_running', False),
    setattr(self, '_uvicorn_server', None),
    setattr(self, '_temp_config_path', None),
    # Add routes here or in run_async? Adding here means they are added when the thread object is created.
    # Adding in run_async means they are added when the thread starts.
    # If the thread is stopped and restarted, adding in run_async would add them again.
    # Adding them once when the module is loaded or the first instance is created is better.
    # Let's add them outside the class definition, guarded by a check.
    # This requires the handlers to be defined outside the class or access state differently.
    # Reverting to adding routes in __init__ but guarding against duplicates.
    # The handlers _get_lmstudio_models and _get_lmstudio_model need to be defined outside the class
    # or access state via app.state as planned. Let's define them outside.
    # This means they can't directly use `self`.

    # --- REVISING LiteLLMProxyThread and Handlers ---
    # Handlers must be defined outside the class to be added to the global 'app'.
    # They will access state via `request.app.state`.
    # The state (`models_config`, `is_model_running_callback`) must be set on `app.state`
    # before uvicorn starts, within the thread's `run_async`.

    # Let's redefine the handlers and the __init__ and run_async methods.
    # The handlers will be standalone functions.
    # The __init__ will store the state.
    # The run_async will set app.state and start uvicorn.

    pass # Redefining below

) # End of lambda for __init__


# Define standalone handlers that access state via app.state
async def _get_lmstudio_models_handler(request: Request):
    """Handler for GET /api/v0/models"""
    # Access state from the request's app instance
    models_config = request.app.state.models_config
    is_model_running_callback = request.app.state.is_model_running_callback

    if not gguf_metadata.GGUF_AVAILABLE:
         return JSONResponse(content={"error": "GGUF library not available for metadata extraction."}, status_code=500)

    try:
        all_models_data = gguf_metadata.get_all_models_lmstudio_format(
            models_config, is_model_running_callback
        )
        return JSONResponse(content={
            "object": "list",
            "data": all_models_data
        })
    except Exception as e:
        logging.error(f"Error handling /api/v0/models: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal Server Error retrieving models metadata")


async def _get_lmstudio_model_handler(model_id: str, request: Request):
    """Handler for GET /api/v0/models/{model_id}"""
    # Access state from the request's app instance
    models_config = request.app.state.models_config
    is_model_running_callback = request.app.state.is_model_running_callback

    if not gguf_metadata.GGUF_AVAILABLE:
         raise HTTPException(status_code=500, detail="GGUF library not available for metadata extraction.")

    try:
        # Find the model in the config by its LM Studio ID (which is the model_name from config)
        model_data = gguf_metadata.get_single_model_lmstudio_format(
            model_id, models_config, is_model_running_callback
        )

        if model_data:
            return JSONResponse(content=model_data)
        else:
            raise HTTPException(status_code=404, detail=f"Model with id '{model_id}' not found")

    except Exception as e:
        logging.error(f"Error handling /api/v0/models/{model_id}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal Server Error retrieving model metadata")


# Add LM Studio compatible routes to the global FastAPI app instance
# Check if routes already exist to avoid adding them multiple times
# This is a simple check and might not be robust against all changes
existing_routes = [route.path for route in app.routes]
if "/api/v0/models" not in existing_routes:
     logging.info("Adding LM Studio compatible API routes.")
     app.add_api_route("/api/v0/models", _get_lmstudio_models_handler, methods=["GET"])
     app.add_api_route("/api/v0/models/{model_id}", _get_lmstudio_model_handler, methods=["GET"])
     # Add redirects if necessary - relying on LiteLLM defaults for now
else:
     logging.info("LM Studio compatible API routes already exist.")


class LiteLLMProxyThread(QThread):
    """
    QThread to run the LiteLLM proxy in a separate thread.
    """
    # Define signals if needed (e.g., started, stopped, error)
    # started = Signal()
    # stopped = Signal()
    # error = Signal(str)

    def __init__(self, models_config: Dict[str, Dict[str, Any]], is_model_running_callback: Callable[[str], bool], api_key: str = None, active_model_name: str = None, active_model_port: int = None):
        super().__init__()
        self.models_config = models_config
        self.is_model_running_callback = is_model_running_callback
        self.api_key = api_key
        self.active_model_name = active_model_name # Model name for LiteLLM's model_list
        self.active_model_port = active_model_port # Port for LiteLLM's model_list
        self.is_running = False
        self._uvicorn_server = None
        self._temp_config_path = None

    def run(self):
        """
        Runs the LiteLLM proxy in the thread.
        """
        self.is_running = True
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.run_async())
        except Exception as e:
            logging.error(f"Unexpected error in LiteLLMProxyThread run: {e}\n{traceback.format_exc()}")
        finally:
            self.is_running = False
            # Clean up the temporary config file
            if self._temp_config_path and os.path.exists(self._temp_config_path):
                 try:
                     os.unlink(self._temp_config_path)
                     logging.info(f"Cleaned up temporary LiteLLM config file: {self._temp_config_path}")
                 except OSError as e:
                     logging.error(f"Error cleaning up temporary LiteLLM config file {self._temp_config_path}: {e}")
            self._temp_config_path = None # Clear the path

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
                "model_list": [], # Will add the active model here
                "general_settings": {
                    "master_key": self.api_key if self.api_key else None
                }
            }

            # Add the active model to the LiteLLM model_list if provided
            if self.active_model_name and self.active_model_port:
                 proxy_config["model_list"].append({
                     "model_name": self.active_model_name,
                     "litellm_params": {
                         "model": f"openai/{self.active_model_name}", # Use openai provider pointing to llama.cpp
                         "api_base": f"http://127.0.0.1:{self.active_model_port}",
                         "api_key": "sk-dummy" # Dummy key required by LiteLLM for openai provider
                     }
                 })
                 print(f"LiteLLM Proxy configured for active model '{self.active_model_name}' on port {self.active_model_port}.")
            else:
                 print("LiteLLM Proxy started without an active Llama runner configured in model_list.")
                 # The proxy will start, but /v1 endpoints won't work until a model is configured/restarted.
                 # The /v0/models endpoint should still work.


            # 2. Dump to a temp YAML file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode='w') as f:
                yaml.dump(proxy_config, f)
                self._temp_config_path = f.name # Store path for cleanup
            logging.info(f"LiteLLM proxy config written to {self._temp_config_path}")

            # 3. Point LiteLLM at that file
            os.environ["CONFIG_FILE_PATH"] = self._temp_config_path

            # 4. Start the proxy embedded via Uvicorn
            # Set state on the global app instance BEFORE creating the server
            # This state will be accessible by the standalone handler functions
            app.state.models_config = self.models_config
            app.state.is_model_running_callback = self.is_model_running_callback

            # Use port 1234 as required
            uvicorn_config = uvicorn.Config(app, host="0.0.0.0", port=1234, reload=False)
            self._uvicorn_server = uvicorn.Server(uvicorn_config)

            print("LiteLLM Proxy listening on http://0.0.0.0:1234")

            # This call is blocking until the server stops
            await self._uvicorn_server.serve()

        except Exception as e:
            print(f"Error starting LiteLLM Proxy: {e}")
            logging.error(f"Error starting LiteLLM Proxy: {e}\n{traceback.format_exc()}")
            # Emit error signal if needed
            # self.error.emit(str(e))
        finally:
            # Cleanup happens in run()'s finally block now
            print("LiteLLM Proxy stopped.")


    def stop(self):
        """
        Signals the LiteLLM proxy thread to stop.
        """
        self.is_running = False
        if self._uvicorn_server:
            self._uvicorn_server.should_exit = True
            # Note: Stopping uvicorn gracefully can be tricky in an embedded context.
            # This might not immediately stop the serve() call.
            # A more robust stop might involve sending a signal or using a shutdown event.
            # For now, setting should_exit is the standard uvicorn way.

