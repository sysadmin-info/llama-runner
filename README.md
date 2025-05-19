## llama-runner
Llama.cpp runner/swapper i proxy emulujące backend LMStudio / Ollama (dla IntelliJ AI Assistant / GitHub Copilot / VS COde za pomocą RooCode)

# Instalacja

Kompletny poradnik dla WSL2 w Windows 11 opisujący instalację i konfigurację lokalnego dużego modleu języukowego LLM:

* `llamafile` oraz `llama-runner`,
* z modelem 30B Q4 lub Qwen3 14B 128K,
* działających przez `llama.cpp` z patchem dla 80K kontekstu,
* z integracją do RooCode w VS Code,
* uruchamianych przez WSL2 (i dodatkowo przez CMake na Windows jako bonus).

# Środowisko i przygotowanie sprzętu

Komputer **AtomMan X7 Ti** z 96 GB RAM i kartą **NVIDIA RTX 3060 (12 GB)** podłączoną przez eGPU DEG1 (OcuLink) jest podstawą. Pracujemy pod **Windows 11**, ale cała instalacja i uruchamianie LLM odbywa się w **WSL2** (Ubuntu 22.04).

* **Sterowniki GPU:** Na Windows zainstaluj sterownik NVIDIA zgodny z WSL2 i CUDA (najlepiej najnowszy z serii 525+). Upewnij się, że Windows wykrywa GPU, a w WSL uzyskuje się dostęp do GPU (polecenie `nvidia-smi` w WSL2 powinno pokazać RTX 3060). .
* **Zasilacz i eGPU:** 650 W PSU starczy dla RTX 3060. Brak NVLink oznacza, że nie rozdzielamy obliczeń między karty - całość modelu musi zmieścić się w jednej karcie (lub częściowo na RAM). W praktyce **gpu\_layers=99** (ponad 98%) stawia większość sieci na GPU, co wymaga kwantyzacji i zarządzania pamięcią, by nie przekroczyć 12 GB VRAM.

Zainstaluj Windows Subsystem Linux z poziomu PowerShell

```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
wsl --set-default-version 2
```
W systemie Windows włącz **funkcję WSL2** (PowerShell: `wsl --install`) i dodaj dystrybucję **Ubuntu 22.04** z Microsoft Store. Tak, w Debian 12 nie działa instalacja NVIDIA CUDA 😂 Następnie w WSL:

```bash
sudo apt update
sudo apt install -y build-essential dkms cmake git python3 python3-pip nvidia-cuda-toolkit nvidia-cuda-dev libcurl4-openssl-dev curl jq unzip zipalign
```

Po restarcie uruchom polecenie `nvidia-smi`; jeśli widzisz listę procesorów, GPU jest gotowe.

## Kompilacja llama.cpp z obsługą CUDA i długiego kontekstu

W WSL2 pobierz repozytorium llama.cpp:

```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

*Ważne:* domyślnie llama.cpp obsługuje do 8K lub 32K kontekstu zależnie od modelu. Aby uzyskać *80 000 tokenów*, użyj mechanizmu RoPE-skali i YaRN (eyrap) - opisane poniżej. Nie ma oficjalnego automatycznego patcha 80K, ale można zmienić ustawienia RoPE i YaRN (podejście „rope-scaling yarn”).

Zbuduj bibliotekę z flagami CUDA:

```bash
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FORCE_CUBLAS=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_CUDA_F16=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

* `-DGGML_CUDA=ON` włącza akcelerację GPU (cUBLAS i flash-attn).
* `-DGGML_CUDA_FA_ALL_QUANTS=ON` pozwala na pełne wsparcie wszystkich kombinacji ilości bitów w pamięci KV dla flash-attn (potrzebne przy mieszaniu Q4/KV). Kompilacja jest wtedy dłuższa, ale niezbędna dla naszych ustawień `cache-type-k=f16`, `cache-type-v=q4_0`.
* `-DGGML_CUDA_FORCE_CUBLAS=ON` wymusza użycie cUBLAS do mnożeń macierzowych (może przyspieszyć na nowszych GPU kosztem większej pamięci).
* `-DGGML_CUDA_F16=ON` umożliwia użycie precyzji FP16 w pewnych operacjach CUDA, co poprawia wydajność na nowszych kartach.

Dodatkowo warto ustawić zmienne środowiskowe przy uruchomieniu:

```bash
export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1     # fallback do RAM, gdy VRAM zabraknie:contentReference[oaicite:3]{index=3}  
export GGML_CUDA_FORCE_CUBLAS=1             # (na wszelki wypadek)  
```

`GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` pozwala na używanie pamięci systemowej, gdy GPU nie mieści całego zestawu KV (zapobiega zrywaniu programu przy 80K kontekstu). Bez tego przy tak dużej pamięci KV możemy łatwo przekroczyć 12 GB VRAM.

Po kompilacji w katalogu `build/bin` powstaną pliki binarne, w tym `llama-server` (serwer OpenAI-API) i `llama-cli` (narzędzie CLI). Upewnij się, że program poprawnie widzi GPU (`./build/bin/llama-cli --version` powinien pokazać twoją kartę RTX 3060).

## Metoda 1: **Llama-runner (lokalny serwer OpenAI-API z llama.cpp)**

Ta metoda to uruchomienie standardowego serwera z projektu llama.cpp (kompatybilnego z OpenAI). Oferuje pełną kontrolę i konfigurację GGUF. Postępujemy tak:

1. **Przygotowanie modelu:** Jak wyżej, pobierz wybrany model GGUF (30B Q4 lub Qwen3 14B) i umieść np. w `/home/user/models/model.gguf`.

```bash
cd ~
mkdir models && cd models
wget -O Qwen3-14B-128K-IQ4_NL.gguf https://huggingface.co/unsloth/Qwen3-14B-128K-GGUF/resolve/main/Qwen3-14B-128K-IQ4_NL.gguf
```

2. **Uruchomienie serwera:** Użyj skompilowanego pliku `llama-server` z katalogu `build/bin` (powstałego w kroku budowy). Przykład:

**Pamiętaj, aby user zastąpić nazwą swojego użytkownika domowego!**

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

   Tutaj:

   * `-m` wskazuje ścieżkę do modelu GGUF,
   * reszta flag (`--ctx-size`, `--gpu-layers`, itd.) identycznie jak wyżej,
   * `--port 8080` ustawia nasłuch na porcie 8080 (można zmienić).

   Serwer startuje i czeka na zapytania na `http://localhost:8080/v1/chat/completions` i innych końcówkach OpenAI. Możesz go testować (np. `curl` z JSONem, patrz \[70] lub \[120] dla przykładu struktury API).

3. **Test lokalny (opcjonalnie):** Możesz sprawdzić działanie przez `curl` (opcja `-d` JSON z wiadomościami jak w OpenAI) lub za pomocą narzędzi typu Postman. Przykładowo:

   ```bash
   curl http://localhost:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
           "model": "any-model",
           "messages": [
             {"role": "system", "content": "Jesteś asystentem kodu."},
             {"role": "user", "content": "Napisz funkcję w Pythonie sortującą listę liczb."}
           ]
         }'
   ```

   Model zwróci JSON z odpowiedzią pod kluczem `"choices":[{"message":{"content":...}}]`.

> **Praktyczne uwagi:** Z racji braku NVLink i ograniczonej VRAM użycie `--no-kv-offload` i podanych kwantyzacji (`k=f16`, `v=q4_0`) jest kluczowe, by model „zmieścił się” w 12 GB. Włączenie **FlashAttention** (`--flash-attn`) przyspieszy obliczenia na GPU. Jeśli zauważysz błędy pamięci, spróbuj także zmienić `GGML_CUDA_FORCE_MMQ` na `1` (w środowisku), co zmniejszy zużycie VRAM kosztem nieco wolniejszych obliczeń. Upewnij się też, że masz włączoną obsługę *Unified Memory* (`GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`), jak pokazano wyżej.

## Metoda 2: **llama-runner** (samodzielne uruchomienie modelu w sposób bardziej uproszczony)

Ta metoda to uruchomienie runnera z projektu [llama-runner ](https://github.com/sysadmin-info/llama-runner.git). Jest to fork projektu [llama-runner by Piotr Wilkin](https://github.com/pwilkin/llama-runner.git), który został zmodyfikowany przeze mnie, aby móc uruchamiać go z poziomu WSL2.

Zakładam, że wszystkie kroki przed metodą pierwszą zostały wykonane.

Sklonuj repozytorium:

```bash
git clone https://github.com/sysadmin-info/llama-runner.git
cd llama-runner
mkdir dev-venv
python3 -m venv dev-venv
source dev-venv/bin/activate
pip install -r requirements.txt
python3 main.py
```


## Integracja z VS Code (RooCode)

Aby korzystać z modelu jako backendu LLM w VS Code (np. z rozszerzeniem **RooCode**, dawniej RooCline):

1. **Zainstaluj RooCode:** Wejdź w Extensions w VS Code i zainstaluj „Roo Code” autorstwa RooVeterinaryInc. Po instalacji pojawi się ikona RooCode w panelu bocznym.

2. **Konfiguracja połączenia:** RooCode pozwala podłączyć się do dowolnego „OpenAI-compatible” endpointu. W ustawieniach (lub pliku `settings.json` VS Code) wpisz:

   ```json
   "roocode.openai_api_base": "http://localhost:8080/v1",
   "roocode.openai_model": "qwen3-14b-128k",   // nazwa modelu może być dowolna, serwer ignoruje tę wartość bo bierze ją z config.json
   "roocode.openai_api_key": ""
   ```

   * `openai_api_base` wskazuje adres Twojego serwera llama.cpp (z plusem ścieżki `/v1`).
   * `openai_model` w RooCode musi być podane, ale serwer llama.cpp nie używa tej wartości (może być `any-model`).
   * `openai_api_key` zostaw puste – lokalny serwer nie wymaga autoryzacji.

   W niektórych wersjach może to być w GUI ustawień: wybierz *AI Provider → Add Provider → Generic OpenAI*, i tam podaj URL `http://localhost:8080` oraz klucz (pusty).

3. **Używanie:** Po konfiguracji RooCode będzie wysyłać zapytania do lokalnego serwera, a model odpowiadać jak zdalny. Nie musisz używać `curl` ani osobnego interfejsu – wszystkie zapytania generowane są wewnętrznie (np. w oknie czatu RooCode).

> **Uwaga:** W praktyce użytkownicy reportują, że konfiguracja RooCode z własnym serwerem czasem wymaga manualnego wskazania URL w ustawieniach. Ważne, by serwer `llama-server` (lub `llamafile --server`) działał przed uruchomieniem sesji w VS Code.




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
