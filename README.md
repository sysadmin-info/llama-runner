## llama-runner
Llama.cpp runner/swapper and proxy that emulates LMStudio / Ollama backends (for IntelliJ AI Assistant / GitHub Copilot)

# Installation

```
$ git clone https://github.com/pwilkin/llama-runner
$ cd llama-runner
$ mkdir dev-venv
$ python -m venv dev-venv
$ source dev-venv/bin/activate (or .\dev-venv\Scripts\Activate.ps1 on Windows)
$ pip install -r requirements.txt
... create ~/.llama-runner/config.json ...
$ python main.py
```

# Sample config file

```json
{
  "llama-runtimes": {
    "default": {
      "runtime": "llama-server"
    },
    "ik_llama": {
      "runtime": "/home/adrian/llama.cpp/build/bin/llama-server",
      "supports_tools": false
    }
  },
  "models": {
        "Qwen3 14B 128K": {
          "model_path": "/home/adrian/models/Qwen3-14B-128K-IQ4_NL.gguf",
          "llama_cpp_runtime": "default",
          "parameters": {
            "ctx_size": 80000,
            "gpu_layers": 99,
            "no_kv_offload": true,
            "cache-type-k": "f16",
            "cache-type-v": "q4_0",
            "flash-attn": true,
            "min_p": 0,
            "top_p": 0.9,
            "top_k": 20,
            "temp": 0.6,
            "rope-scale": 4,
            "yarn-orig-ctx": 32768,
            "jinja": true
          }
        }
      }
    },
    "Qwen3 8B": {
      "model_path": "/mnt/win/k/models/unsloth/Qwen3-8B-GGUF/Qwen3-8B-Q5_K_M.gguf",
      "llama_cpp_runtime": "default",
      "parameters": {
        "ctx_size": 25000,
        "gpu_layers": 99,
        "cache-type-k": "f16",
        "cache-type-v": "q4_0",
        "flash-attn": true,
        "min_p": 0,
        "top_p": 0.9,
        "top_k": 20,
        "temp": 0.6,
        "no_webui": true
      }
    },
    "Qwen3 30B MoE": {
      "model_path": "/mnt/win/k/models/unsloth/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-UD-Q3_K_XL.gguf",
      "llama_cpp_runtime": "default",
      "parameters": {
        "override_tensor": "(up_exps|down_exps)=CPU",
        "ctx_size": 22000,
        "gpu_layers": 99,
        "min_p": 0,
        "top_p": 0.9,
        "top_k": 20,
        "temp": 0.6
      }
    },
    "Hermes 3B": {
      "model_path": "/mnt/win/k/models/NousResearch/Hermes-3-Llama-3.2-3B-GGUF/Hermes-3-Llama-3.2-3B.Q5_K_M.gguf",
      "llama_cpp_runtime": "default",
      "parameters": {
        "ctx_size": 100000,
        "gpu_layers": 99,
        "cache-type-k": "q8_0",
        "cache-type-v": "q8_0",
        "flash-attn": true
      }
    }
  }
}
```

# Functionality
* support for different llama.cpp runtimes including ik_llama (for ik_llama, specify "port" in model configuration for runner)
* dynamically loads and unloads runtimes based on model string in request
* dynamically strips tool queries for ik_llama that doesn't support it
* double proxy: emulation for LM Studio-specific backend and OpenAI-compatible backends (running on port 1234) and for Ollama specific backends (running on port 11434)
* tested on GitHub Copilot (for Ollama emulation) and on IntelliJ AI Assistant (for LM Studio emulation)
* tested on Windows & Linux (Ubuntu 24.10)

# Disclaimer

Yes, this is mostly vibe-coded. Pull requests fixing glaring code issues / inefficiencies are welcome. Comments pointing out glaring code issues / inefficiencies are not welcome (unless it's security-critical).
