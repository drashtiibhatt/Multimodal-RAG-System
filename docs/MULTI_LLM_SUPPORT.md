# Multi-LLM Support Documentation

This document explains how to use the multi-LLM provider system, which allows you to switch between OpenAI's cloud models and local models via Ollama.

## Table of Contents
1. [Overview](#overview)
2. [Supported Providers](#supported-providers)
3. [Quick Start](#quick-start)
4. [OpenAI Provider](#openai-provider)
5. [Ollama Provider (Local Models)](#ollama-provider-local-models)
6. [Configuration](#configuration)
7. [Usage Examples](#usage-examples)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The multi-LLM provider system provides a unified interface for using different LLM backends:

- **OpenAI Provider**: Cloud-based GPT models (GPT-4, GPT-3.5-turbo)
- **Ollama Provider**: Local open-source models (Llama 2, Mistral, CodeLlama, etc.)

### Benefits

✅ **Cost Savings**: Use free local models for development/testing
✅ **Privacy**: Keep sensitive data local with Ollama
✅ **Flexibility**: Switch providers without code changes
✅ **Offline Support**: Work without internet using local models

---

## Supported Providers

### 1. OpenAI (Cloud)
- **Models**: GPT-4, GPT-4-turbo, GPT-3.5-turbo
- **Requires**: OpenAI API key
- **Internet**: Required
- **Cost**: Pay-per-token

### 2. Ollama (Local)
- **Models**: Llama 2, Mistral, CodeLlama, Phi, Gemma, and more
- **Requires**: Ollama installed locally
- **Internet**: Not required (after model download)
- **Cost**: Free

---

## Quick Start

### Option 1: Use OpenAI (Default)

```bash
# Set in .env file
LLM_PROVIDER=openai
LLM_MODEL=gpt-4-turbo-preview
OPENAI_API_KEY=your_api_key_here
```

### Option 2: Use Ollama (Local)

1. **Install Ollama**:
   ```bash
   # Download from https://ollama.ai
   # Or use package manager:
   # macOS: brew install ollama
   # Linux: curl https://ollama.ai/install.sh | sh
   ```

2. **Start Ollama server**:
   ```bash
   ollama serve
   ```

3. **Pull a model**:
   ```bash
   # Llama 2 (7B)
   ollama pull llama2

   # Or Mistral (7B)
   ollama pull mistral

   # Or CodeLlama (for code tasks)
   ollama pull codellama
   ```

4. **Configure in .env**:
   ```bash
   LLM_PROVIDER=ollama
   OLLAMA_MODEL=llama2
   OLLAMA_BASE_URL=http://localhost:11434
   ```

---

## OpenAI Provider

### Configuration

```python
from src.generation import UseCaseGenerator

# Use OpenAI with default config
generator = UseCaseGenerator(provider="openai")

# Or specify model
generator = UseCaseGenerator(
    provider="openai",
    model="gpt-4-turbo-preview",
    temperature=0.3
)
```

### Supported Models

- `gpt-4` - Most capable, best quality
- `gpt-4-turbo-preview` - Faster, cheaper GPT-4
- `gpt-3.5-turbo` - Fast, cost-effective

### Features

✅ JSON mode support
✅ High-quality outputs
✅ Function calling
✅ Vision capabilities (GPT-4V)

---

## Ollama Provider (Local Models)

### Installation

**Windows**:
1. Download from https://ollama.ai
2. Run installer
3. Ollama runs as a service automatically

**macOS**:
```bash
brew install ollama
```

**Linux**:
```bash
curl https://ollama.ai/install.sh | sh
```

### Starting Ollama

```bash
# Start server (runs on http://localhost:11434)
ollama serve

# In another terminal, pull a model
ollama pull llama2
```

### Available Models

| Model | Size | Best For | Command |
|-------|------|----------|---------|
| **Llama 2** | 7B | General tasks | `ollama pull llama2` |
| **Mistral** | 7B | Fast, accurate | `ollama pull mistral` |
| **CodeLlama** | 7B-34B | Code generation | `ollama pull codellama` |
| **Phi-2** | 2.7B | Lightweight | `ollama pull phi` |
| **Gemma** | 2B-7B | Google's model | `ollama pull gemma` |

Full list: https://ollama.ai/library

### Configuration

```python
from src.generation import UseCaseGenerator

# Use Ollama with default config
generator = UseCaseGenerator(provider="ollama")

# Or specify model and URL
generator = UseCaseGenerator(
    provider="ollama",
    model="mistral",  # or llama2, codellama, etc.
    temperature=0.3
)

# Custom Ollama URL (if running on different host)
from src.llm_providers import OllamaProvider

provider = OllamaProvider(
    model="llama2",
    base_url="http://192.168.1.100:11434"
)
```

### Features

✅ JSON mode support
✅ Completely free
✅ Privacy-focused (data stays local)
✅ No API key needed
✅ Offline capable
⚠️ Slower than cloud models
⚠️ Lower quality than GPT-4

---

## Configuration

### Environment Variables

Update your `.env` file:

```bash
# Choose provider
LLM_PROVIDER=ollama  # or "openai"

# OpenAI settings (if using OpenAI)
OPENAI_API_KEY=your_key_here
LLM_MODEL=gpt-4-turbo-preview

# Ollama settings (if using Ollama)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2  # or mistral, codellama, etc.

# Common settings
TEMPERATURE=0.3
```

### Programmatic Configuration

```python
from src.config import get_settings

settings = get_settings()
print(f"Provider: {settings.llm_provider}")
print(f"Model: {settings.llm_model}")
```

---

## Usage Examples

### Example 1: Basic Usage

```python
from src.generation import UseCaseGenerator

# Initialize with Ollama
generator = UseCaseGenerator(provider="ollama", model="llama2")

# Generate use cases
output = generator.generate(
    query="Create test cases for user login",
    context="User can login with email and password",
    debug=True
)

print(f"Generated {len(output['use_cases'])} use cases")
```

### Example 2: Switching Providers

```python
# Test with both providers
providers = [
    ("OpenAI GPT-4", "openai", "gpt-4"),
    ("Local Llama 2", "ollama", "llama2")
]

for name, provider, model in providers:
    print(f"\nTesting {name}...")

    gen = UseCaseGenerator(
        provider=provider,
        model=model
    )

    if gen.provider.is_available():
        output = gen.generate(query, context)
        print(f"✓ Success: {len(output['use_cases'])} use cases")
    else:
        print(f"✗ Provider not available")
```

### Example 3: Direct Provider API

```python
from src.llm_providers import OllamaProvider

provider = OllamaProvider(model="mistral")

# Check availability
if not provider.is_available():
    print("Ollama not running!")
    exit(1)

# List available models
models = provider.list_models()
print(f"Available models: {models}")

# Generate
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain what a test case is."}
]

response = provider.generate(messages, json_mode=False)
print(response.content)
```

### Example 4: Full Pipeline with Ollama

```python
from src.main import RAGPipeline

# Initialize pipeline with Ollama
pipeline = RAGPipeline(llm_provider="ollama", llm_model="llama2")

# Ingest documents (works same as before)
pipeline.ingest_directory("data/sample_dataset/user-signup")

# Query with local model
results = pipeline.query(
    "Create test cases for password validation",
    use_safe_generator=True,
    debug=True
)

print(f"Generated {len(results['use_cases'])} use cases using local model!")
```

---

## Troubleshooting

### Ollama Issues

**Problem**: `Cannot connect to Ollama server`

**Solution**:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it
ollama serve

# Check status
ps aux | grep ollama  # Linux/macOS
```

**Problem**: `Model not found`

**Solution**:
```bash
# List installed models
ollama list

# Pull missing model
ollama pull llama2
```

**Problem**: `Generation is very slow`

**Solution**:
- Use smaller models (phi, llama2 instead of llama2:70b)
- Increase system RAM
- Close other applications
- Use GPU acceleration if available

### OpenAI Issues

**Problem**: `OpenAI client not initialized`

**Solution**:
```bash
# Check API key in .env
OPENAI_API_KEY=sk-...

# Or set environment variable
export OPENAI_API_KEY=sk-...
```

### JSON Mode Issues

**Problem**: `JSON parsing failed with Ollama`

**Solution**:
- Some smaller models struggle with JSON format
- Try using `mistral` or `codellama` instead
- Reduce complexity of the request
- Increase temperature slightly (0.3 → 0.5)

### Comparison: When to Use Each Provider

| Scenario | Recommended Provider |
|----------|---------------------|
| Production deployments | OpenAI |
| Development/testing | Ollama |
| Sensitive/private data | Ollama |
| Best quality outputs | OpenAI (GPT-4) |
| Cost optimization | Ollama |
| Offline work | Ollama |
| Fast iterations | OpenAI (GPT-3.5) |
| Code generation | Ollama (CodeLlama) or OpenAI |

---

## Testing

Run the test suite to verify both providers:

```bash
# Test Ollama setup
python test_ollama.py

# Test full pipeline
python -m src.main --query "test query" --llm-provider ollama
```

---

## Performance Comparison

Based on average performance:

| Metric | OpenAI GPT-4 | OpenAI GPT-3.5 | Ollama Llama2 | Ollama Mistral |
|--------|--------------|----------------|---------------|----------------|
| Speed | ~5-10s | ~2-5s | ~15-30s | ~10-20s |
| Quality | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Cost | $$$ | $ | Free | Free |
| JSON Mode | ✅ | ✅ | ✅ | ✅ |

*Times are approximate for 500-token responses on average hardware*

---

## Advanced Configuration

### Custom Provider Implementation

You can add support for other LLM providers by implementing the `BaseLLMProvider` interface:

```python
from src.llm_providers import BaseLLMProvider, LLMResponse

class CustomProvider(BaseLLMProvider):
    def generate(self, messages, **kwargs):
        # Your implementation
        pass

    def is_available(self):
        # Check availability
        pass

    def get_provider_name(self):
        return "custom"
```

### Provider Selection Logic

The system selects providers in this order:
1. Explicit `provider` parameter
2. `LLM_PROVIDER` environment variable
3. Default to OpenAI

---

## Best Practices

1. **Development**: Use Ollama to save costs during development
2. **Production**: Use OpenAI for best quality
3. **Testing**: Cache results to avoid re-generating
4. **Privacy**: Use Ollama for sensitive documents
5. **Fallback**: Implement fallback from Ollama to OpenAI if needed

---

## Resources

- **Ollama Documentation**: https://ollama.ai
- **Ollama Models**: https://ollama.ai/library
- **OpenAI Documentation**: https://platform.openai.com/docs
- **Provider Code**: `src/llm_providers/`

---

## Summary

Multi-LLM support gives you flexibility to:
- ✅ Use free local models for development
- ✅ Switch to cloud models for production
- ✅ Keep sensitive data private
- ✅ Work offline when needed
- ✅ Optimize costs

Choose the provider that best fits your needs!
