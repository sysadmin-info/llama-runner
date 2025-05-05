import asyncio
import os
import subprocess
import logging
import re
from typing import Dict, Any

from llama_runner.config_loader import load_config, CONFIG_DIR, LOG_FILE

# Configure logging
logging.basicConfig(filename=LOG_FILE, level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class LlamaCppRunner:
    def __init__(self, model_name: str, model_path: str, llama_cpp_runtime: str = None, **kwargs):
        """
        Initializes the LlamaCppRunner.

        Args:
            model_name (str): The display name of the model.
            model_path (str): The full path to the model file.
            llama_cpp_runtime (str, optional): The path to the llama-server executable.
                Defaults to None, which uses the llama-server from the PATH.
            **kwargs: Additional arguments to pass to llama-server.
        """
        self.model_name = model_name
        self.model_path = model_path
        self.llama_cpp_runtime = llama_cpp_runtime or "llama-server"
        self.kwargs = kwargs
        self.process: subprocess.Popen = None
        self.startup_pattern = re.compile(r"main: server is listening on")  # Regex to detect startup
        self.port = None #Dynamically assigned port

    async def start(self):
        """
        Starts the llama.cpp server.
        """
        if self.process and self.process.poll() is None:
            print(f"llama.cpp server for {self.model_name} is already running.")
            return

        command = [
            self.llama_cpp_runtime,
            "--model", self.model_path,
            "--alias", self.model_name,
            "--host", "127.0.0.1",  # Bind to localhost
            "--port", "0"             # Dynamically assigned port
        ]

        # Add additional arguments from kwargs
        for key, value in self.kwargs.items():
            arg_name = key.replace("_", "-")  # Convert snake_case to kebab-case
            command.append(f"--{arg_name}")
            command.append(str(value))

        print(f"Starting llama.cpp server with command: {' '.join(command)}")
        logging.info(f"Starting llama.cpp server with command: {' '.join(command)}")

        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=CONFIG_DIR  # Run in the config directory
            )

            await self.wait_for_server_startup()

        except OSError as e:
            print(f"Error starting llama.cpp server: {e}")
            logging.error(f"Error starting llama.cpp server: {e}")

    async def wait_for_server_startup(self):
        """
        Waits for the llama.cpp server to start up, using a regex pattern.
        """
        if not self.process or not self.process.stdout:
            return False

        while True:
            line = await asyncio.get_event_loop().run_in_executor(None, self.process.stdout.readline)
            if not line:
                break  # Process ended

            print(f"llama.cpp[{self.model_name}]: {line.strip()}")
            match = self.startup_pattern.search(line)
            if match:
                print(f"llama.cpp server for {self.model_name} started successfully.")
                 # Extract port from the startup message
                match = re.search(r'http://127\.0\.0\.1:(\d+)', line)
                if match:
                    self.port = int(match.group(1))
                    print(f"llama.cpp server for {self.model_name} is listening on port {self.port}")
                return True
            if self.process.poll() is not None:
                print(f"llama.cpp server for {self.model_name} exited before startup.")
                return False

        return False

    async def stop(self):
        """
        Stops the llama.cpp server.
        """
        if self.process and self.process.poll() is None:
            print(f"Stopping llama.cpp server for {self.model_name}.")
            self.process.terminate()
            try:
                await asyncio.wait_for(asyncio.get_event_loop().run_in_executor(None, self.process.wait), timeout=15)
            except asyncio.TimeoutError:
                print(f"llama.cpp server for {self.model_name} did not terminate in 15 seconds, killing it.")
                self.process.kill()
            self.process = None
            print(f"llama.cpp server for {self.model_name} stopped.")
        else:
            print(f"llama.cpp server for {self.model_name} is not running.")

    def is_running(self):
        """
        Checks if the llama.cpp server is running.
        """
        return self.process is not None and self.process.poll() is None

    def get_port(self):
        """
        Returns the port number of the llama.cpp server.
        """
        return self.port

async def main():
    """
    Main function to demonstrate the usage of LlamaCppRunner.
    """
    config = load_config()
    llama_runtimes = config.get("llama-runtimes", {})
    default_runtime = "llama-server"  # Default to llama-server from PATH

    # Load model-specific config, if available
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
    # await asyncio.sleep(10)  # Run for 10 seconds # REMOVE THIS LINE
    # await runner.stop() # REMOVE THIS LINE

# if __name__ == "__main__": # REMOVE THIS LINE
#    asyncio.run(main()) # REMOVE THIS LINE
