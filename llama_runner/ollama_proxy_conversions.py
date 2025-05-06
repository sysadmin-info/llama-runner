def embeddingRequestFromOllama(ollama_req: dict) -> dict:
    """
    Converts an Ollama embedding request to OpenAI-compatible format.
    Ollama: { model, prompt }
    OpenAI: { model, input }
    """
    return {
        "model": ollama_req.get("model"),
        "input": ollama_req.get("prompt")
    }


def embeddingResponseToOllama(openai_resp: dict) -> dict:
    """
    Converts an OpenAI embedding response to Ollama format.
    OpenAI: { data: [ { embedding, index } ], model, usage }
    Ollama: { embedding }
    """
    first = openai_resp.get("data", [{}])[0]
    return {
        "embedding": first.get("embedding", [])
    }


def generateRequestFromOllama(ollama_req: dict) -> dict:
    """
    Converts an Ollama completion request to OpenAI-compatible format.
    Ollama: { model, prompt, options: { temperature, max_tokens? } }
    OpenAI: { model, prompt, temperature, max_tokens }
    """
    opts = ollama_req.get("options", {})
    return {
        "model": ollama_req.get("model"),
        "prompt": ollama_req.get("prompt"),
        "temperature": opts.get("temperature", 1.0),
        "max_tokens": opts.get("max_tokens")
    }


def generateResponseToOllama(openai_resp: dict) -> dict:
    """
    Converts an OpenAI completion response to Ollama format.
    OpenAI: { choices: [ { text, finish_reason } ], usage?, id?, ... }
    Ollama: { response, done }
    """
    choice = openai_resp.get("choices", [{}])[0]
    return {
        "response": choice.get("text", ""),
        "done": choice.get("finish_reason") == "stop"
    }


def chatRequestFromOllama(ollama_req: dict) -> dict:
    """
    Converts an Ollama chat request to OpenAI-compatible format.
    Ollama: { model, messages, options: { temperature, max_tokens? } }
    OpenAI: { model, messages, temperature, max_tokens }
    """
    opts = ollama_req.get("options", {})
    return {
        "model": ollama_req.get("model"),
        "messages": ollama_req.get("messages", []),
        "temperature": opts.get("temperature", 1.0),
        "max_tokens": opts.get("max_tokens")
    }


def chatResponseToOllama(openai_resp: dict) -> dict:
    """
    Converts an OpenAI chat completion response to Ollama format.
    OpenAI: { choices: [ { message: { role, content }, finish_reason } ], ... }
    Ollama: { message: { role, content }, done }
    """
    choice = openai_resp.get("choices", [{}])[0]
    return {
        "message": choice.get("message", {}),
        "done": choice.get("finish_reason") == "stop"
    }