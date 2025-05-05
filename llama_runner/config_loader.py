import os
import json
import logging

CONFIG_DIR = os.path.expanduser("~/.llama-runner")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
LOG_FILE = os.path.join(CONFIG_DIR, "error.log")

# Ensure the log directory exists
if not os.path.exists(CONFIG_DIR):
    os.makedirs(CONFIG_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(filename=LOG_FILE, level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_config_exists():
    """
    Ensures that the configuration directory and file exist.
    Creates them if they don't.
    """
    if not os.path.exists(CONFIG_DIR):
        try:
            os.makedirs(CONFIG_DIR)
        except OSError as e:
            print(f"Error creating config directory: {e}")
            logging.error(f"Error creating config directory: {e}")
            return False

    if not os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump({}, f)  # Create an empty JSON file
        except OSError as e:
            print(f"Error creating config file: {e}")
            logging.error(f"Error creating config file: {e}")
            return False
    return True

def load_config():
    """
    Loads the configuration from the JSON file.
    Returns a dictionary containing the configuration.
    """
    if not ensure_config_exists():
        return {}

    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
            print(f"Loaded config: {config}")  # Print loaded config
            return config
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error loading config file: {e}")
        logging.error(f"Error loading config file: {e}")
        return {}

if __name__ == '__main__':
    # Example usage:
    config = load_config()
    print(f"Loaded config: {config}")
