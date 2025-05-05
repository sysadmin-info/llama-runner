import sys
import asyncio
import subprocess
import os
import tempfile # Keep tempfile for potential future use, but config file logic removed
import yaml # Keep yaml for potential future use, but config file logic removed
import logging
import traceback
import time
import json
from typing import Dict, Any, Callable, List, Optional

# Removed: from litellm.proxy.proxy_server import app
from fastapi import FastAPI, HTTPException, Request, Response, status # Import FastAPI and necessary components
from fastapi.responses import JSONResponse, StreamingResponse # RedirectResponse is not needed if we proxy
import httpx # Import httpx for forwarding requests

from PySide6.QtCore import QThread, QObject, Signal, Slot, QTimer # Import QTimer for potential use

from llama_runner import gguf_metadata # Import the new metadata module

# Configure logging (already done in main.py for configurable levels)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Create our own FastAPI app instance ---
app = FastAPI()
# --- End create app instance ---


# Define standalone handlers that access state via app.state
@app.get("/api/v0/models")
async def _get_lmstudio_models_handler(request: Request):
    """Handler for GET /api/v0/models"""
    # Access state from the request's app instance
    models_config = request.app.state.models_config
    is_model_running_callback = request.app.state.is_model_running_callback

    if not gguf_metadata.GGUF_AVAILABLE:
         return JSONResponse(content={"error": "GGUF library not available for metadata extraction."}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

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
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal Server Error retrieving models metadata")

@app.get("/api/v0/models/{model_id}")
async def _get_lmstudio_model_handler(model_id: str, request: Request):
    """Handler for GET /api/v0/models/{model_id}"""
    # Access state from the request's app instance
    models_config = request.app.state.models_config
    is_model_running_callback = request.app.state.is_model_running_callback

    if not gguf_metadata.GGUF_AVAILABLE:
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="GGUF library not available for metadata extraction.")

    try:
        # Find the model in the config by its LM Studio ID (which is the model_name from config)
        model_data = gguf_metadata.get_single_model_lmstudio_format(
            model_id, models_config, is_model_running_callback
        )

        if model_data:
            return JSONResponse(content=model_data)
        else:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model with id '{model_id}' not found")

    except Exception as e:
        logging.error(f"Error handling /api/v0/models/{model_id}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal Server Error retrieving model metadata")


# --- Handler for dynamic routing of /v1/* requests ---
async def _dynamic_route_v1_request(request: Request):
    """
    Intercepts /v1/* requests, ensures the target runner is running,
    and forwards the request to the runner's port.
    """
    # Access state and callbacks from the request's app instance
    models_config = request.app.state.models_config
    is_model_running_callback = request.app.state.is_model_running_callback
    get_runner_port_callback = request.app.state.get_runner_port_callback
    request_runner_start_callback = request.app.state.request_runner_start_callback
    # Access the proxy thread instance to get the futures dictionary
    proxy_thread: FastAPIProxyThread = request.app.state.proxy_thread_instance # We'll set this in run_async

    # Extract the model name from the request body (common for chat/completions)
    # This is a simplification; a more robust solution would inspect the path and body structure
    # based on the specific endpoint (/v1/chat/completions, /v1/completions, etc.)
    # For now, assume the model name is in the request JSON body under the 'model' key.
    # Need to read the body without consuming it for the downstream request.
    try:
        # Read body bytes first, then attempt to parse as JSON
        body_bytes = await request.body()
        body = {}
        try:
            if body_bytes:
                body = json.loads(body_bytes)
        except json.JSONDecodeError:
            logging.warning(f"Could not decode request body as JSON for {request.url.path}")
            # Continue without body if JSON parsing fails, maybe model is in path or headers?
            # For now, assume model is required in body for these endpoints.
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON request body.")


        model_name = body.get("model")
        if not model_name:
             # If model is not in body, try path parameters if applicable (less common for v1)
             # Or raise an error if model is required but missing
             logging.warning(f"Model name not found in request body for {request.url.path}")
             raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Model name not specified in request body.")

    except HTTPException:
         # Re-raise HTTPException if already set
         raise
    except Exception as e:
        logging.error(f"Error reading request body or extracting model name: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid request: {e}")

    logging.debug(f"Intercepted request for model: {model_name} at path: {request.url.path}")

    if model_name not in models_config:
        logging.warning(f"Request for unknown model: {model_name}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model '{model_name}' not found in configuration.")

    # Check if the runner is already running
    port = get_runner_port_callback(model_name)

    if port is None:
        # Runner is not running, request startup and wait
        logging.info(f"Runner for {model_name} not running. Requesting startup.")
        try:
            # Request startup via the callback, which returns an asyncio.Future
            # Store the future locally in the proxy thread instance
            if model_name not in proxy_thread._runner_ready_futures or proxy_thread._runner_ready_futures[model_name].done():
                 logging.debug(f"Creating new startup future for {model_name}")
                 proxy_thread._runner_ready_futures[model_name] = request_runner_start_callback(model_name)
            else:
                 logging.debug(f"Using existing startup future for {model_name}")

            # Wait for the runner to become ready (Future to resolve)
            # Use a timeout to prevent infinite waiting
            startup_timeout = 60 # seconds
            port = await asyncio.wait_for(
                proxy_thread._runner_ready_futures[model_name],
                timeout=startup_timeout
            )
            logging.info(f"Runner for {model_name} is ready on port {port} after startup.")

        except asyncio.TimeoutError:
            logging.error(f"Timeout waiting for runner {model_name} to start after {startup_timeout} seconds.")
            # Clean up the future if it timed out
            if model_name in proxy_thread._runner_ready_futures and not proxy_thread._runner_ready_futures[model_name].done():
                 proxy_thread._runner_ready_futures[model_name].cancel() # Cancel the future
                 del proxy_thread._runner_ready_futures[model_name]
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Timeout starting runner for model '{model_name}'.")
        except Exception as e:
            logging.error(f"Error during runner startup for {model_name}: {e}\n{traceback.format_exc()}")
            # Clean up the future if it failed
            if model_name in proxy_thread._runner_ready_futures and not proxy_thread._runner_ready_futures[model_name].done():
                 proxy_thread._runner_ready_futures[model_name].set_exception(e) # Set exception on the future
                 del proxy_thread._runner_ready_futures[model_name]
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error starting runner for model '{model_name}': {e}")

    else:
        logging.debug(f"Runner for {model_name} is already running on port {port}.")
        # If it was running, ensure its future is marked as done with the port
        # This handles cases where the proxy restarts but the runner is still alive
        if model_name not in proxy_thread._runner_ready_futures or not proxy_thread._runner_ready_futures[model_name].done():
             logging.debug(f"Creating completed future for already running runner {model_name}")
             future = asyncio.Future()
             future.set_result(port)
             proxy_thread._runner_ready_futures[model_name] = future
             # No need for timer cleanup here, as it's already running.
             # The future will be removed if the runner stops later.


    # Runner is ready and port is known. Forward the request.
    # Construct the target URL using the known port and the original request path
    target_url = f"http://127.0.0.1:{port}{request.url.path}"
    logging.debug(f"Forwarding request for {model_name} to {target_url}")

    # Use httpx to forward the request
    async with httpx.AsyncClient() as client:
        try:
            # Reconstruct headers, removing host and potentially others that shouldn't be forwarded
            headers = dict(request.headers)
            headers.pop('host', None) # Remove host header
            # Remove content-length header as httpx will set it correctly based on the forwarded content
            headers.pop('content-length', None)


            # Forward the request, including method, URL path, headers, and body
            # Use the body_bytes read earlier
            proxy_response = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body_bytes, # Pass the raw body bytes
                timeout=600.0, # Use a generous timeout for model responses
                # stream=True # Enable streaming response handling - httpx handles this automatically with async iteration
            )

            # Check if the response is streaming (e.g., Server-Sent Events)
            # This requires inspecting headers like 'content-type'
            content_type = proxy_response.headers.get('content-type', '').lower()
            if 'text/event-stream' in content_type:
                 logging.debug(f"Forwarding streaming response from {target_url}")
                 # Return a StreamingResponse that reads from the httpx response stream
                 return StreamingResponse(
                     content=proxy_response.aiter_bytes(), # Use aiter_bytes for async iteration
                     status_code=proxy_response.status_code,
                     headers=proxy_response.headers
                 )
            else:
                 logging.debug(f"Forwarding non-streaming response from {target_url}")
                 # For non-streaming responses, read the body and return a standard Response
                 response_body = await proxy_response.aread() # Read the entire body asynchronously
                 return Response(
                     content=response_body,
                     status_code=proxy_response.status_code,
                     headers=proxy_response.headers
                 )


        except httpx.RequestError as e:
            logging.error(f"Error forwarding request to runner {model_name} on port {port}: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Error communicating with runner for model '{model_name}': {e}")
        except Exception as e:
            logging.error(f"Unexpected error during request forwarding for {model_name}: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal error processing request for model '{model_name}': {e}")

# --- End handler for /v1/* requests ---


# --- Handlers for /api/v0/* proxying ---
# These handlers will call the _dynamic_route_v1_request handler internally
@app.post("/api/v0/chat/completions")
async def _proxy_v0_chat_completions(request: Request):
    """Proxies /api/v0/chat/completions to /v1/chat/completions."""
    logging.debug("Proxying /api/v0/chat/completions to /v1/chat/completions")
    # Modify the request URL path to /v1/chat/completions before passing to the dynamic router
    # Create a new Request object with the modified URL path
    # Note: Creating a new Request object and copying state is complex.
    # A simpler approach is to modify the path on the original request object if possible,
    # or pass the target path explicitly to the dynamic router.
    # Let's modify the request.url.path temporarily for the call. This might have side effects.
    # A cleaner way is to pass the target path to _dynamic_route_v1_request.
    # Let's refactor _dynamic_route_v1_request to accept a target_path.

    # --- Refactoring _dynamic_route_v1_request to accept target_path ---
    # This requires changing the signature of _dynamic_route_v1_request
    # and updating where it's called.

    # Let's stick to the simpler approach for now: modify the request path temporarily.
    # This is less ideal but avoids a larger refactor.
    original_path = request.url.path
    request.scope['path'] = "/v1/chat/completions"
    try:
        response = await _dynamic_route_v1_request(request)
        return response
    finally:
        # Restore the original path
        request.scope['path'] = original_path

@app.post("/api/v0/completions")
async def _proxy_v0_completions(request: Request):
    """Proxies /api/v0/completions to /v1/completions."""
    logging.debug("Proxying /api/v0/completions to /v1/completions")
    original_path = request.url.path
    request.scope['path'] = "/v1/completions"
    try:
        response = await _dynamic_route_v1_request(request)
        return response
    finally:
        request.scope['path'] = original_path

@app.post("/api/v0/embeddings")
async def _proxy_v0_embeddings(request: Request):
    """Proxies /api/v0/embeddings to /v1/embeddings."""
    logging.debug("Proxying /api/v0/embeddings to /v1/embeddings")
    original_path = request.url.path
    request.scope['path'] = "/v1/embeddings"
    try:
        response = await _dynamic_route_v1_request(request)
        return response
    finally:
        request.scope['path'] = original_path

# --- End handlers for /api/v0/* proxying ---


# --- Add routes for /v1/* endpoints to be intercepted ---
# Use a path parameter to capture the rest of the path after /v1
# This allows intercepting /v1/chat/completions, /v1/completions, etc.
# The handler will then forward to the correct path on the runner.
# Note: This overrides LiteLLM's default handling for /v1 endpoints.
# If LiteLLM's internal routing is needed *after* runner startup, a different approach is required.
# This approach assumes we are completely bypassing LiteLLM's model routing for /v1.
# We need to add these routes *before* LiteLLM's default /v1 routes are potentially added
# if LiteLLM's app instance includes them by default.
# Inspecting `app.routes` might be necessary to ensure our routes take precedence.
# For simplicity, let's assume adding them here works.

# Check if specific /v1 routes exist before adding our dynamic handler
# This check is complex because LiteLLM might add routes like /v1/chat/completions directly.
# A simpler approach is to add our catch-all /v1/{path:path} route and ensure it's processed first.
# FastAPI processes routes in the order they are added.

# Let's add specific routes for the common endpoints first, then a catch-all if needed.
# This is safer than a broad catch-all if LiteLLM adds other /v1 routes we don't want to intercept.

# Add specific routes for common /v1 endpoints
# The handler will extract the model name from the request body
# Check if our dynamic handlers are already added to avoid duplicates on reload/restart
# This check is fragile, a better approach might be needed if routes are added dynamically elsewhere
dynamic_v1_paths = ["/v1/chat/completions", "/v1/completions", "/v1/embeddings"]
# Check if the *handler function itself* is already associated with the path
# This is a more robust check than just checking the path string
current_v1_handlers = {route.path: route.endpoint for route in app.routes if route.path in dynamic_v1_paths}

# Add routes using the @app.post decorator
@app.post("/v1/chat/completions")
async def _v1_chat_completions_handler(request: Request):
    return await _dynamic_route_v1_request(request)

@app.post("/v1/completions")
async def _v1_completions_handler(request: Request):
    return await _dynamic_route_v1_request(request)

@app.post("/v1/embeddings")
async def _v1_embeddings_handler(request: Request):
    return await _dynamic_route_v1_request(request)

logging.info("Added dynamic routing handlers for /v1/chat/completions, /v1/completions, /v1/embeddings.")


# If needed, add a catch-all for other /v1 paths, but be cautious
# @app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
# async def _v1_catch_all_handler(request: Request):
#     return await _dynamic_route_v1_request(request)
# logging.info("Added catch-all dynamic routing handler for /v1/*.")


# --- End add routes ---


class FastAPIProxyThread(QThread): # Renamed class
    """
    QThread to run the FastAPI proxy in a separate thread. # Updated description
    Handles LM Studio API emulation and runner management requests.
    """
    # Define signals if needed (e.g., started, stopped, error)
    # started = Signal()
    # stopped = Signal()
    # error = Signal(str)

    def __init__(self,
                 models_config: Dict[str, Dict[str, Any]],
                 is_model_running_callback: Callable[[str], bool],
                 get_runner_port_callback: Callable[[str], Optional[int]],
                 request_runner_start_callback: Callable[[str], asyncio.Future], # Callback now returns Future
                 api_key: str = None):
        super().__init__()
        self.models_config = models_config
        self.is_model_running_callback = is_model_running_callback
        self.get_runner_port_callback = get_runner_port_callback
        self.request_runner_start_callback = request_runner_start_callback # Store the callback
        self.api_key = api_key
        self.is_running = False
        self._uvicorn_server = None
        # self._temp_config_path = None # No longer needed

        # Dictionary to hold asyncio Futures for runners that are starting
        # This dictionary is managed by MainWindow, but the proxy thread needs
        # a reference to it or a way to access/manage the Futures.
        # Let's pass the reference to the dictionary from MainWindow.
        # Or, the callbacks from MainWindow can manage the Futures directly.
        # The current design has MainWindow manage the Futures and return them
        # from request_runner_start_callback. The proxy thread will store
        # a *local* copy of the Futures it's currently waiting on.
        self._runner_ready_futures: Dict[str, asyncio.Future] = {}

        # Connect to signals from MainWindow (MainWindow connects these signals to our slots)
        # Signals: runner_port_ready_for_proxy, runner_stopped_for_proxy
        # Slots: on_runner_port_ready, on_runner_stopped


    @Slot(str, int)
    def on_runner_port_ready(self, model_name: str, port: int):
        """Slot to handle runner_port_ready_for_proxy signal from MainWindow."""
        logging.debug(f"Proxy thread received runner_port_ready for {model_name} on port {port}")
        # When a runner is ready, resolve its corresponding Future in our local dictionary
        if model_name in self._runner_ready_futures and not self._runner_ready_futures[model_name].done():
            logging.debug(f"Resolving local runner_ready_future for {model_name} with port {port}")
            self._runner_ready_futures[model_name].set_result(port)
        elif model_name in self._runner_ready_futures and self._runner_ready_futures[model_name].done():
             logging.warning(f"Received runner_port_ready for {model_name}, but local Future was already done.")
        else:
             logging.warning(f"Received runner_port_ready for {model_name}, but no pending local Future found.")


    @Slot(str)
    def on_runner_stopped(self, model_name: str):
        """Slot to handle runner_stopped_for_proxy signal from MainWindow."""
        logging.debug(f"Proxy thread received runner_stopped for {model_name}")
        # If a runner stops, cancel or set exception on its Future in our local dictionary
        if model_name in self._runner_ready_futures:
             if not self._runner_ready_futures[model_name].done():
                 logging.debug(f"Setting exception on local runner_ready_future for {model_name} due to stop.")
                 self._runner_ready_futures[model_name].set_exception(RuntimeError(f"Runner for {model_name} stopped unexpectedly."))
             # Remove the future from the local dictionary
             del self._runner_ready_futures[model_name]
             logging.debug(f"Cleaned up local runner_ready_future for {model_name}")
        else:
             logging.debug(f"Received runner_stopped for {model_name}, but no local Future found.")


    def run(self):
        """
        Runs the FastAPI proxy in the thread using Uvicorn.
        """
        self.is_running = True
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            # --- Set the proxy thread instance on the app state ---
            # This allows the standalone handlers to access the thread's local state (like _runner_ready_futures)
            app.state.proxy_thread_instance = self
            # --- End set instance ---

            self.loop.run_until_complete(self.run_async())
        except Exception as e:
            logging.error(f"Unexpected error in FastAPIProxyThread run: {e}\n{traceback.format_exc()}")
        finally:
            self.is_running = False
            # Clean up the temporary config file (no longer generated)
            # if self._temp_config_path and os.path.exists(self._temp_config_path):
            #      try:
            #          os.unlink(self._temp_config_path)
            #          logging.info(f"Cleaned up temporary LiteLLM config file: {self._temp_config_path}")
            #      except OSError as e:
            #          logging.error(f"Error cleaning up temporary LiteLLM config file {self._temp_config_path}: {e}")
            # self._temp_config_path = None # Clear the path

            # --- Clean up the proxy thread instance from app state ---
            if hasattr(app.state, 'proxy_thread_instance'):
                 del app.state.proxy_thread_instance
            # --- End cleanup ---

            if hasattr(self, 'loop') and self.loop.is_running():
                 self.loop.stop()
            if hasattr(self, 'loop') and not self.loop.is_closed():
                 self.loop.close()


    async def run_async(self):
        """
        Asynchronous part of the proxy runner.
        Starts the Uvicorn server for the FastAPI app.
        """
        print("Starting FastAPI Proxy...")
        try:
            # LiteLLM config generation removed.
            # The FastAPI app 'app' is already created at the module level.
            # Routes are added at the module level.

            # 4. Start the proxy embedded via Uvicorn
            # Set state on the global app instance BEFORE creating the server
            # This state will be accessible by the standalone handler functions
            app.state.models_config = self.models_config
            app.state.is_model_running_callback = self.is_model_running_callback
            app.state.get_runner_port_callback = self.get_runner_port_callback # Pass the new callback
            app.state.request_runner_start_callback = self.request_runner_start_callback # Pass the new callback
            # app.state.runner_ready_futures = self._runner_ready_futures # Pass the futures dict

            # Use port 1234 as required
            uvicorn_config = uvicorn.Config(app, host="0.0.0.0", port=1234, reload=False)
            self._uvicorn_server = uvicorn.Server(uvicorn_config)

            print("FastAPI Proxy listening on http://0.0.0.0:1234")
            logging.info("FastAPI Proxy listening on http://0.0.0.0:1234")


            # This call is blocking until the server stops
            await self._uvicorn_server.serve()

        except Exception as e:
            print(f"Error starting FastAPI Proxy: {e}")
            logging.error(f"Error starting FastAPI Proxy: {e}\n{traceback.format_exc()}")
            # Emit error signal if needed
            # self.error.emit(str(e))
        finally:
            # Cleanup happens in run()'s finally block now
            print("FastAPI Proxy stopped.")
            logging.info("FastAPI Proxy stopped.")


    def stop(self):
        """
        Signals the FastAPI proxy thread to stop.
        """
        self.is_running = False
        if self._uvicorn_server:
            self._uvicorn_server.should_exit = True
            # Note: Stopping uvicorn gracefully can be tricky in an embedded context.
            # This might not immediately stop the serve() call.
            # A more robust stop might involve sending a signal or using a shutdown event.
            # For now, setting should_exit is the standard uvicorn way.
