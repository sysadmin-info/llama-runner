import os
import sys
import argparse
import json
import subprocess
import shlex

CONFIG_PATH = os.path.expanduser('~/.llama-runner/config.json')

def load_config():
    if not os.path.exists(CONFIG_PATH):
        print(f"Config not found: {CONFIG_PATH}")
        sys.exit(1)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def choose_model(models):
    print("Available models:")
    for idx, model_name in enumerate(models):
        print(f"{idx + 1}. {model_name}")
    while True:
        choice = input("Choose model (number): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return list(models.keys())[int(choice)-1]
        print("Invalid choice. Try again.")

def model_to_args(model_info, extra_args=None):
    # Mapowanie parametrów z config.json na flagi CLI (poprawka na cache-type-k, cache-type-v)
    param_map = {
        "ctx_size": "--ctx-size",
        "gpu_layers": "--gpu-layers",
        "no_kv_offload": "--no-kv-offload",
        "cache-type-k": "--cache-type-k",
        "cache-type-v": "--cache-type-v",
        "flash-attn": "--flash-attn",
        "min_p": "--min-p",
        "top_p": "--top-p",
        "top_k": "--top-k",
        "temp": "--temp",
        "rope-scale": "--rope-scale",
        "yarn-orig-ctx": "--yarn-orig-ctx",
        "jinja": "--jinja"
    }
    args = []
    params = model_info.get("parameters", {})
    for key, cli_flag in param_map.items():
        if key in params:
            val = params[key]
            if isinstance(val, bool) and val:
                args.append(cli_flag)
            elif not isinstance(val, bool):
                args.append(cli_flag)
                args.append(str(val))
    if extra_args:
        args += extra_args
    return args

def main():
    parser = argparse.ArgumentParser(description="Llama Runner CLI (no GUI)")
    parser.add_argument('--host', type=str, default="0.0.0.0", help="Host to bind server (default: 0.0.0.0)")
    parser.add_argument('--port', type=int, default=8080, help="Port to bind server (default: 8080)")
    args = parser.parse_args()

    config = load_config()
    models = config.get("models", {})

    if not models:
        print("No models defined in config.json.")
        sys.exit(2)

    model_name = choose_model(models)
    model_info = models[model_name]

    print("\nSelected model:")
    print(f"  Name: {model_name}")
    print(f"  Model file: {model_info.get('model_path')}")
    print(f"  Runtime: {model_info.get('llama_cpp_runtime')}")
    print("  Parameters:")
    for k, v in model_info.get('parameters', {}).items():
        print(f"    {k}: {v}")

    # Pobierz ścieżkę do runtime
    runtimes = config.get("llama-runtimes", {})
    runtime_key = model_info.get('llama_cpp_runtime', 'default')
    runtime_path = runtimes.get(runtime_key, {}).get("runtime", "/home/adrian/llama.cpp/build/bin/llama-server")

    if not os.path.isabs(runtime_path):
        runtime_path = "/home/adrian/llama.cpp/build/bin/llama-server"

    # Zbuduj listę argumentów
    cli_args = [runtime_path, "-m", model_info["model_path"]]
    cli_args += ["--host", args.host, "--port", str(args.port)]
    cli_args += model_to_args(model_info)

    print("\nUruchamiam runner:")
    print(" ".join(shlex.quote(arg) for arg in cli_args))

    # Odpal serwer na stałe (z terminala, nie capture_output)
    try:
        subprocess.run(cli_args)
    except KeyboardInterrupt:
        print("\nZatrzymano serwer.")
    except Exception as e:
        print(f"\nRunner error: {e}")

if __name__ == "__main__":
    main()
