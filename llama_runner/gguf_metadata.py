import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# Attempt to import the gguf library and specific components
try:
    from gguf import GGUFReader
    from gguf.gguf_reader import GGMLQuantizationType # Import the quantization enum
    GGUF_AVAILABLE = True
except ImportError:
    logging.warning("The 'gguf' library is not installed or GGMLQuantizationType is missing. Metadata extraction will be disabled.")
    GGUF_AVAILABLE = False
except Exception as e:
    logging.warning(f"Error importing gguf components: {e}. Metadata extraction may be limited.")
    GGUF_AVAILABLE = False # Treat as unavailable if import fails unexpectedly


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


        # --- Construct LM Studio format based on requested fields ---

        # id: general.name (fallback to filename)
        model_id = metadata.get('general.name', os.path.basename(model_path))

        # object: "model" (static)
        obj_type = "model" # Static value as per LM Studio API

        # type: "llm", "vlm", "embeddings" (detected)
        # Check metadata keys first
        model_type = "llm" # Default to llm
        if metadata.get("ggml.model.type") == "embedding":
             model_type = "embeddings"
        elif metadata.get("ggml.model.type") == "vlm":
             model_type = "vlm"
        # Fallback to filename heuristic if metadata key is missing
        elif "embedding" in os.path.basename(model_path).lower() or "embed" in os.path.basename(model_path).lower():
             model_type = "embeddings"
        # VLM detection is harder, might need specific metadata keys or filename patterns
        # For now, stick to llm/embeddings based on metadata/filename heuristic


        # publisher: general.quantized_by (fallback to general.url, then local)
        publisher = metadata.get('general.quantized_by', metadata.get('general.url', 'local'))

        # arch: general.architecture (fallback to unknown)
        architecture = metadata.get('general.architecture', 'unknown')

        # compatibility_type: "gguf" (static)
        compatibility_type = "gguf" # Static value

        # quantization: GGMLQuantizationType(general.file_type).name (fallback to heuristic)
        quantization = "Unknown"
        file_type = metadata.get('general.file_type')
        if GGUF_AVAILABLE and file_type is not None:
            try:
                # Attempt to use the enum name
                quantization = GGMLQuantizationType(file_type).name
            except ValueError:
                logging.warning(f"Unknown GGMLQuantizationType value: {file_type} for {model_path}")
                quantization = f"Type_{file_type}" # Fallback if enum value is unknown
            except Exception as e:
                 logging.warning(f"Error getting quantization name from enum for {model_path}: {e}")
                 # Fall through to heuristic
        # Fallback to heuristic if enum method fails or is not available
        if quantization == "Unknown":
             if "q4_k_m" in os.path.basename(model_path).lower():
                  quantization = "Q4_K_M" # Common convention
             # Fallback: Check if quantization info is directly in metadata keys (e.g., "quantization.method")
             elif metadata.get("quantization.method"):
                  quantization = metadata["quantization.method"]
             elif metadata.get("quantization_version"): # Sometimes just a version number
                  quantization = f"Q{metadata['quantization_version']}"


        # max_context_length: ${general.architecture}.context_length (fallback to common keys, then default)
        max_ctx = 4096 # Default value
        if architecture != 'unknown':
             # Try the architecture-specific key
             ctx_key = f'{architecture}.context_length'
             max_ctx = metadata.get(ctx_key, max_ctx)

        # If architecture-specific key wasn't found or architecture was unknown, check common keys
        if max_ctx == 4096 and architecture == 'unknown': # Only check common keys if default is still used and arch is unknown
             max_ctx = metadata.get('llama.context_length', max_ctx)
             max_ctx = metadata.get('phi3.context_length', max_ctx)
             max_ctx = metadata.get('qwen2.context_length', max_ctx)
             # Add other common architecture keys here if needed


        # Construct the final LM Studio format dictionary
        lmstudio_format = {
            "id": model_id,
            "object": obj_type,
            "type": model_type,
            "publisher": publisher,
            "arch": architecture,
            "compatibility_type": compatibility_type,
            "quantization": quantization,
            # State will be added later based on runtime status
            "max_context_length": max_ctx
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
        # Return a minimal structure if GGUF library is not available
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
