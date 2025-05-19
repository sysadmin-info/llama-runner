## llama-runner
Llama.cpp runner/swapper and proxy that emulates LMStudio / Ollama backends (for IntelliJ AI Assistant / GitHub Copilot)

# Installation

Complete guide for WSL2 on Windows 11 describing the installation and configuration of a local Large Language Model (LLM):

* `llamafile` and `llama-runner`,
* using a 30B Q4 model or Qwen3 14B 128K,
* running via `llama.cpp` with an 80K context patch,
* integrated with RooCode in VS Code,
* launched through WSL2 (and additionally via CMake on Windows as a bonus).

# Environment and Hardware Setup

The base machine is an **AtomMan X7 Ti** with 96 GB RAM and an **NVIDIA RTX 3060 (12 GB)** GPU connected via an eGPU DEG1 (OcuLink). We work under **Windows 11**, but the entire installation and LLM execution takes place in **WSL2** (Ubuntu 22.04).

* **GPU Drivers:** On Windows, install the NVIDIA driver compatible with WSL2 and CUDA (preferably the latest from the 525+ series). Make sure Windows detects the GPU, and WSL has access to it (the `nvidia-smi` command in WSL2 should show the RTX 3060).
* **Power Supply and eGPU:** A 650 W PSU is sufficient for an RTX 3060. Since there is no NVLink, we can't split computation across GPUs – the entire model must fit on one card (or partially into RAM). In practice, **gpu\_layers=99** (over 98%) puts most of the network on the GPU, requiring quantization and memory tuning to stay under the 12 GB VRAM limit.

Install Windows Subsystem for Linux via PowerShell:

```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
wsl --set-default-version 2
````

In Windows, enable the **WSL2 feature** (PowerShell: `wsl --install`) and add the **Ubuntu 22.04** distribution from the Microsoft Store. Yes, NVIDIA CUDA installation doesn't work on Debian 12 😂. Then in WSL:

```bash
sudo apt update
sudo apt install -y build-essential dkms cmake git python3 python3-pip nvidia-cuda-toolkit nvidia-cuda-dev libcurl4-openssl-dev curl jq unzip zipalign
```

After rebooting, run the `nvidia-smi` command; if you see a list of processors, the GPU is ready.

## Building llama.cpp with CUDA and long context support

In WSL2, clone the llama.cpp repository:

```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

*Important:* By default, llama.cpp supports up to 8K or 32K context depending on the model. To achieve *80,000 tokens*, use RoPE scaling and YaRN (eyrap) – described below. There is no official 80K patch, but you can tweak the RoPE and YaRN settings (the “rope-scaling yarn” approach).

Build the library with CUDA flags:

```bash
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FORCE_CUBLAS=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_CUDA_F16=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

* `-DGGML_CUDA=ON` enables GPU acceleration (cUBLAS and flash-attn).
* `-DGGML_CUDA_FA_ALL_QUANTS=ON` enables full support for all KV memory bit quant combinations with flash-attn (needed for mixing Q4/KV). It increases compile time but is required for our settings: `cache-type-k=f16`, `cache-type-v=q4_0`.
* `-DGGML_CUDA_FORCE_CUBLAS=ON` forces matrix multiplications to use cUBLAS (can be faster on newer GPUs at the cost of more memory).
* `-DGGML_CUDA_F16=ON` enables FP16 precision in some CUDA ops for better performance on modern cards.

It's also helpful to set the following environment variables when launching:

```bash
export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1     # fallback to RAM when VRAM is exhausted
export GGML_CUDA_FORCE_CUBLAS=1              # (just in case)
```

`GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` allows fallback to system RAM when the entire KV cache doesn't fit in GPU VRAM (prevents crashes with 80K context). Without it, the 12 GB VRAM can be easily exceeded.

After the build, the `build/bin` directory will contain binaries including `llama-server` (OpenAI-API server) and `llama-cli` (CLI tool). Verify GPU visibility (`./build/bin/llama-cli --version` should show your RTX 3060).

## Method 1: **Llama-runner (local OpenAI-API server using llama.cpp)**

This method runs the standard OpenAI-compatible server from the llama.cpp project. It offers full control over GGUF configuration. Steps:

1. **Prepare the model:** As before, download your selected GGUF model (30B Q4 or Qwen3 14B) and place it in e.g. `/home/user/models/model.gguf`.

```bash
cd ~
mkdir models && cd models
wget -O Qwen3-14B-128K-IQ4_NL.gguf https://huggingface.co/unsloth/Qwen3-14B-128K-GGUF/resolve/main/Qwen3-14B-128K-IQ4_NL.gguf
```

2. **Start the server:** Use the compiled `llama-server` binary from the `build/bin` directory (from the build step). Example:

**Remember to replace `user` with your actual home directory username!**

```bash
cd llama.cpp
./build/bin/llama-server \
  -m /home/user/models/Qwen3-14B-128K-IQ4_NL.gguf \
  --ctx-size 80000 \
  --gpu-layers 99 \
  --no-kv-offload \
  --flash-attn \
  --cache-type-k f16 \
  --cache-type-v q4_0 \
  --rope-scaling yarn \
  --rope-scale 4 \
  --jinja  --yarn-orig-ctx 32768 \
  --port 8080
```

Here:

* `-m` points to the GGUF model path,
* the rest of the flags (`--ctx-size`, `--gpu-layers`, etc.) are as explained above,
* `--port 8080` sets the listening port (changeable).

The server starts and waits for requests at `http://localhost:8080/v1/chat/completions` and other OpenAI endpoints. You can test it (e.g. with `curl` using JSON – see \[70] or \[120] for example request format).

3. **Local test (optional):** You can test it using `curl` (with `-d` JSON like OpenAI) or tools like Postman. Example:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "any-model",
        "messages": [
          {"role": "system", "content": "You are a code assistant."},
          {"role": "user", "content": "Write a Python function to sort a list of numbers."}
        ]
      }'
```

The model will return a JSON response with `"choices":[{"message":{"content":...}}]`.

> **Practical tips:** Due to the lack of NVLink and limited VRAM, using `--no-kv-offload` and the specified quant settings (`k=f16`, `v=q4_0`) is key to fitting the model within 12 GB. Enabling **FlashAttention** (`--flash-attn`) speeds up GPU computation. If memory errors occur, try setting `GGML_CUDA_FORCE_MMQ=1` (as an env var), which reduces VRAM use at the cost of slightly slower performance. Ensure *Unified Memory* (`GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`) is enabled as shown above.

## Method 2: **llama-runner** (standalone, simplified local launcher)

This method runs the launcher from the [llama-runner project](https://github.com/sysadmin-info/llama-runner.git). It’s a fork of [llama-runner by Piotr Wilkin](https://github.com/pwilkin/llama-runner.git), modified by me to work under WSL2.

Assumes all prior setup steps (before Method 1) have been completed.

Clone the repository:

```bash
git clone https://github.com/sysadmin-info/llama-runner.git
cd llama-runner
mkdir dev-venv
python3 -m venv dev-venv
source dev-venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

## Integration with VS Code (RooCode)

To use the model as an LLM backend in VS Code (e.g., with **RooCode**, formerly RooCline):

1. **Install RooCode:** Open Extensions in VS Code and install “Roo Code” by RooVeterinaryInc. A RooCode icon will appear in the sidebar.

2. **Connection setup:** RooCode allows connection to any "OpenAI-compatible" endpoint. In the settings (or in your `settings.json` file), enter:

   ```json
   "roocode.openai_api_base": "http://localhost:8080/v1",
   "roocode.openai_model": "qwen3-14b-128k",   // model name can be anything, the server ignores it and reads from config.json
   "roocode.openai_api_key": ""
   ```

   * `openai_api_base` points to your llama.cpp server (add `/v1` path).
   * `openai_model` must be provided in RooCode, but the llama.cpp server doesn’t use it (can be `any-model`).
   * `openai_api_key` should be left empty – the local server does not require auth.

   In some versions, you can configure this via GUI: choose *AI Provider → Add Provider → Generic OpenAI*, and enter `http://localhost:8080` and leave the key field blank.

3. **Usage:** Once configured, RooCode will send requests to the local server and the model will respond as if it were remote. You don’t need to use `curl` or a separate UI – all requests are generated internally (e.g., within RooCode’s chat interface).

> **Note:** In practice, users report that RooCode sometimes needs the server URL to be set manually. Ensure that the `llama-server` (or `llamafile --server`) is running before starting the VS Code session.

# Sample config file

```json
{
  "llama-runtimes": {
    "default": {
      "runtime": "llama-server"
    },
    "ik_llama": {
      "runtime": "/home/user/llama.cpp/build/bin/llama-server",
      "supports_tools": false
    }
  },
  "models": {
    "Qwen3 14B 128K": {
      "model_path": "/home/user/models/Qwen3-14B-128K-IQ4_NL.gguf",
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
