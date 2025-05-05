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
