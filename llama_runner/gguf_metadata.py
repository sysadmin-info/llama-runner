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

        # Helper to get key value safely
        def get_key_value(key: str) -> Any:
            gguf_value = reader.get_key(key)
            return gguf_value.value if gguf_value else None

        # Extract common metadata using get_key().value
        metadata['general.architecture'] = get_key_value('general.architecture')
        metadata['general.name'] = get_key_value('general.name')
        metadata['llama.context_length'] = get_key_value('llama.context_length')
        metadata['llama.block_count'] = get_key_value('llama.block_count')
        metadata['llama.embedding_length'] = get_key_value('llama.embedding_length')
        metadata['tokenizer.model'] = get_key_value('tokenizer.model')
        metadata['general.file_type'] = get_key_value('general.file_type') # This might indicate quantization

        # Attempt to determine quantization from file type or name
        quantization = "Unknown"
        file_type = metadata.get('general.file_type')
        if file_type is not None:
             # Map GGUF file_type integer to string representation (based on gguf.py)
             # This mapping might need refinement based on actual gguf library values
             file_type_map = {
                 0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 4: "q4_1_f16",
                 5: "q8_0", 6: "q5_0", 7: "q5_1", 8: "q2_k", 9: "q3_k_s",
                 10: "q3_k_m", 11: "q3_k_l", 12: "q4_k_s", 13: "q4_k_m",
                 14: "q5_k_s", 15: "q5_k_m", 16: "q6_k", 17: "q8_k", 18: "iq4_nl",
                 19: "iq3_xxs", 20: "iq2_xs", 21: "iq2_s", 22: "iq1_s", 23: "iq1_m",
                 24: "bf16", 25: "ggml_iq2_xxs", 26: "ggml_iq2_xs", 27: "ggml_iq3_xxs",
                 28: "ggml_iq0_xs", 29: "ggml_iq0_s", 30: "ggml_iqn_s", 31: "ggml_iqn_m",
                 32: "ggml_iqn_l", 33: "ggml_iq4_nl", 34: "ggml_iq4_xs", 35: "ggml_iq1_s",
                 36: "ggml_iq1_m", 37: "ggml_iq2_s", 38: "ggml_iq2_m", 39: "ggml_iq2_l",
                 40: "ggml_iq3_s", 41: "ggml_iq3_m", 42: "ggml_iq3_l", 43: "ggml_iq4_s",
                 44: "ggml_iq4_m", 45: "ggml_iq4_l", 46: "ggml_iq5_s", 47: "ggml_iq5_m",
                 48: "ggml_iq5_l", 49: "ggml_iq6_s", 50: "ggml_iq6_m", 51: "ggml_iq6_l",
                 52: "ggml_iq1_xxs", 53: "ggml_iq1_xs", 54: "ggml_iq2_xxs", 55: "ggml_iq2_xs",
                 56: "ggml_iq3_xxs", 57: "ggml_iq0_xxs", 58: "ggml_iq0_xs", 59: "ggml_iq0_s",
                 60: "ggml_iqn_xxs", 61: "ggml_iqn_xs", 62: "ggml_iqn_s", 63: "ggml_iqn_m",
                 64: "ggml_iqn_l", 65: "ggml_iq4_xxs", 66: "ggml_iq4_xs", 67: "ggml_iq4_s",
                 68: "ggml_iq4_m", 69: "ggml_iq4_l", 70: "ggml_iq5_xxs", 71: "ggml_iq5_xs",
                 72: "ggml_iq5_s", 73: "ggml_iq5_m", 74: "ggml_iq5_l", 75: "ggml_iq6_xxs",
                 76: "ggml_iq6_xs", 77: "ggml_iq6_s", 78: "ggml_iq6_m", 79: "ggml_iq6_l",
                 80: "ggml_q2_k", 81: "ggml_q3_k_s", 82: "ggml_q3_k_m", 83: "ggml_q3_k_l",
                 84: "ggml_q4_k_s", 85: "ggml_q4_k_m", 86: "ggml_q5_k_s", 87: "ggml_q5_k_m",
                 88: "ggml_q6_k", 89: "ggml_q8_k", 90: "ggml_f32", 91: "ggml_f16", 92: "ggml_bf16"
             }
             quantization = file_type_map.get(file_type, f"Type_{file_type}")
        elif "q4_k_m" in os.path.basename(model_path).lower():
             quantization = "Q4_K_M" # Common convention

        # Attempt to determine type (llm, vlm, embeddings)
        model_type = "llm" # Default to llm
        if "embedding" in os.path.basename(model_path).lower() or "embed" in os.path.basename(model_path).lower():
             model_type = "embeddings"
        # VLM detection is harder, might need specific metadata keys or filename patterns
        # For now, stick to llm/embeddings based on filename heuristic

        # Construct LM Studio format
        lmstudio_format = {
            "id": metadata.get('general.name', os.path.basename(model_path)), # Use GGUF name or filename
            "object": "model",
            "type": model_type,
            "publisher": "local", # Or try to extract from metadata if available
            "arch": metadata.get('general.architecture', 'unknown'),
            "compatibility_type": "gguf", # Assuming all are gguf
            "quantization": quantization,
            # State will be added later based on runtime status
            "max_context_length": metadata.get('llama.context_length', 4096) # Default if not found
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
