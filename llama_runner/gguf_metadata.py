import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# Attempt to import the gguf library
try:
    from gguf import GGUFReader
    GGUF_AVAILABLE = True
except ImportError:
    logging.warning("The 'gguf' library is not installed. Metadata extraction will be disabled.")
    GGUF_AVAILABLE = False

from llama_runner.config_loader import CONFIG_DIR # Assuming CONFIG_DIR is defined here

METADATA_CACHE_DIR = os.path.join(CONFIG_DIR, "metadata_cache")
Path(METADATA_CACHE_DIR).mkdir(parents=True, exist_ok=True)

def ensure_cache_dir_exists():
    """Ensures the metadata cache directory exists."""
    Path(METADATA_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    logging.info(f"Ensured metadata cache directory exists: {METADATA_CACHE_DIR}")

def calculate_file_hash(filepath: str) -> str:
    """Calculates the SHA256 hash of a file."""
    hasher = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            # Read the file in chunks to avoid large memory usage
            while chunk := f.read(4096):
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        logging.error(f"File not found for hashing: {filepath}")
        return ""
    except Exception as e:
        logging.error(f"Error calculating hash for {filepath}: {e}")
        return ""

def get_metadata_cache_path(model_name: str, file_hash: str) -> str:
    """Generates a cache file path based on model name and file hash."""
    # Sanitize model name for filename
    safe_model_name = "".join(c if c.isalnum() or c in (' ', '.', '-') else '_' for c in model_name).replace(' ', '_')
    # Use a portion of the hash to keep filenames shorter but still unique per file version
    return os.path.join(METADATA_CACHE_DIR, f"{safe_model_name}_{file_hash[:8]}.json")

def load_metadata_from_cache(model_name: str, file_hash: str) -> Optional[Dict[str, Any]]:
    """Loads metadata from the cache file if it exists and is valid."""
    cache_path = get_metadata_cache_path(model_name, file_hash)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                metadata = json.load(f)
            logging.info(f"Loaded metadata from cache for {model_name}")
            return metadata
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from cache file {cache_path}: {e}")
            # In case of error, treat cache as invalid
            return None
        except Exception as e:
            logging.error(f"Error loading metadata from cache {cache_path}: {e}")
            return None
    return None

def save_metadata_to_cache(model_name: str, file_hash: str, metadata: Dict[str, Any]):
    """Saves metadata to a cache file."""
    cache_path = get_metadata_cache_path(model_name, file_hash)
    try:
        with open(cache_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logging.info(f"Saved metadata to cache for {model_name} at {cache_path}")
    except Exception as e:
        logging.error(f"Error saving metadata to cache {cache_path}: {e}")

def extract_gguf_metadata(model_path: str) -> Optional[Dict[str, Any]]:
    """Extracts relevant metadata from a GGUF file."""
    if not GGUF_AVAILABLE:
        logging.error("GGUF library not available. Cannot extract metadata.")
        return None
    if not os.path.exists(model_path):
        logging.error(f"Model file not found for metadata extraction: {model_path}")
        return None

    try:
        reader = GGUFReader(model_path, 'r')
        metadata = {}

        # Iterate through all fields and extract key-value pairs
        for key, field in reader.fields.items():
            try:
                # Access the value using the method shown in the example reader.py
                # field.data[0] is the index into field.parts
                value = field.parts[field.data[0]]
                metadata[key] = value
            except (IndexError, TypeError, AttributeError) as e:
                logging.warning(f"Could not extract value for key '{key}' from {model_path}: {e}")
                metadata[key] = None # Store None or skip? Store None for now.


        # Attempt to determine quantization from file type or name
        quantization = "Unknown"
        file_type = metadata.get('general.file_type')
        if file_type is not None:
             # Map GGUF file_type integer to string representation (based on gguf.py)
             # This mapping might need refinement based on actual gguf library values
             # Using a simplified map for common types, as the full map is large
             file_type_map = {
                 0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1",
                 5: "q8_0", 6: "q5_0", 7: "q5_1", 8: "q2_k", 9: "q3_k_s",
                 10: "q3_k_m", 11: "q3_k_l", 12: "q4_k_s", 13: "q4_k_m",
                 14: "q5_k_s", 15: "q5_k_m", 16: "q6_k", 17: "q8_k",
                 24: "bf16",
                 # Add more mappings if needed based on gguf.py source
             }
             quantization = file_type_map.get(file_type, f"Type_{file_type}")
        elif "q4_k_m" in os.path.basename(model_path).lower():
             quantization = "Q4_K_M" # Common convention
        # Fallback: Check if quantization info is directly in metadata keys (e.g., "quantization.method")
        elif metadata.get("quantization.method"):
             quantization = metadata["quantization.method"]
        elif metadata.get("quantization_version"): # Sometimes just a version number
             quantization = f"Q{metadata['quantization_version']}"


        # Attempt to determine type (llm, vlm, embeddings)
        model_type = "llm" # Default to llm
        # Check metadata keys first
        if metadata.get("ggml.model.type") == "embedding":
             model_type = "embeddings"
        elif metadata.get("ggml.model.type") == "vlm":
             model_type = "vlm"
        # Fallback to filename heuristic if metadata key is missing
        elif "embedding" in os.path.basename(model_path).lower() or "embed" in os.path.basename(model_path).lower():
             model_type = "embeddings"
        # VLM detection is harder, might need specific metadata keys or filename patterns
        # For now, stick to llm/embeddings based on metadata/filename heuristic


        # Construct LM Studio format
        lmstudio_format = {
            "id": metadata.get('general.name', os.path.basename(model_path)), # Use GGUF name or filename
            "object": "model",
            "type": model_type,
            "publisher": metadata.get('general.url', 'local'), # Use URL if available, else local
            "arch": metadata.get('general.architecture', 'unknown'),
            "compatibility_type": "gguf", # Assuming all are gguf
            "quantization": quantization,
            # State will be added later based on runtime status
            "max_context_length": metadata.get('llama.context_length', metadata.get('phi3.context_length', metadata.get('qwen2.context_length', 4096))) # Check common keys, default if not found
        }

        logging.info(f"Successfully extracted metadata for {model_path}")
        return lmstudio_format

    except Exception as e:
        logging.error(f"Error extracting GGUF metadata from {model_path}: {e}")
        return None

def get_model_lmstudio_format(model_name: str, model_path: str, is_running: bool) -> Optional[Dict[str, Any]]:
    """
    Gets model metadata in LM Studio format, using cache if available.
    Includes the current running state.
    """
    if not GGUF_AVAILABLE:
        return {
            "id": model_name,
            "object": "model",
            "type": "llm", # Assume LLM if no metadata
            "publisher": "local",
            "arch": "unknown",
            "compatibility_type": "unknown",
            "quantization": "unknown",
            "state": "loaded" if is_running else "not-loaded",
            "max_context_length": 4096 # Default if no metadata
        }

    file_hash = calculate_file_hash(model_path)
    if not file_hash:
        logging.error(f"Could not get hash for {model_path}. Cannot use cache.")
        # Fallback to extracting without caching if hashing fails
        metadata = extract_gguf_metadata(model_path)
        if metadata:
             metadata["state"] = "loaded" if is_running else "not-loaded"
        return metadata


    cached_metadata = load_metadata_from_cache(model_name, file_hash)

    if cached_metadata:
        # Update the state based on current runtime status
        cached_metadata["state"] = "loaded" if is_running else "not-loaded"
        return cached_metadata
    else:
        logging.info(f"Cache miss or invalid for {model_name}. Extracting metadata...")
        extracted_metadata = extract_gguf_metadata(model_path)
        if extracted_metadata:
            # Add state and save to cache
            extracted_metadata["state"] = "loaded" if is_running else "not-loaded"
            save_metadata_to_cache(model_name, file_hash, extracted_metadata)
            return extracted_metadata
        else:
            logging.error(f"Failed to extract metadata for {model_name} at {model_path}")
            # Return a minimal structure if extraction fails
            return {
                "id": model_name,
                "object": "model",
                "type": "llm", # Assume LLM if no metadata
                "publisher": "local",
                "arch": "unknown",
                "compatibility_type": "unknown",
                "quantization": "unknown",
                "state": "loaded" if is_running else "not-loaded",
                "max_context_length": 4096 # Default if no metadata
            }

def get_all_models_lmstudio_format(models_config: Dict[str, Dict[str, Any]], is_model_running_callback) -> List[Dict[str, Any]]:
    """
    Gets metadata for all configured models in LM Studio format.
    Uses the running status callback to determine the state.
    """
    all_models_data = []
    for model_name, model_config in models_config.items():
        model_path = model_config.get("model_path")
        if not model_path:
            logging.warning(f"Model '{model_name}' has no 'model_path' in config. Skipping metadata.")
            continue

        is_running = is_model_running_callback(model_name)
        metadata = get_model_lmstudio_format(model_name, model_path, is_running)
        if metadata:
            all_models_data.append(metadata)

    return all_models_data

def get_single_model_lmstudio_format(model_name: str, models_config: Dict[str, Dict[str, Any]], is_model_running_callback) -> Optional[Dict[str, Any]]:
    """
    Gets metadata for a single configured model in LM Studio format.
    Uses the running status callback to determine the state.
    """
    model_config = models_config.get(model_name)
    if not model_config:
        return None # Model not found

    model_path = model_config.get("model_path")
    if not model_path:
        logging.warning(f"Model '{model_name}' has no 'model_path' in config. Cannot get metadata.")
        return None

    is_running = is_model_running_callback(model_name)
    metadata = get_model_lmstudio_format(model_name, model_path, is_running)

    return metadata

# Ensure cache directory exists on import
ensure_cache_dir_exists()
