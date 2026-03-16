# CLI Reference

Conscribe provides a command-line interface for generating config stubs and inspecting registries.

## Commands

### `generate-config`

Generate a Python config stub file from a registry.

```bash
conscribe generate-config \
  --registrar "my_app.llm._registrar:LLMRegistrar" \
  --discover "my_app.llm.providers" \
  --output "generated/llm_config.py"
```

**Options:**

| Flag | Required | Description |
|------|----------|-------------|
| `--registrar` | Yes | Dotted path to registrar (`module:attribute`) |
| `--discover` | No | Package paths to import before generation |
| `--output` | No | Output file path (prints to stdout if omitted) |
| `--json-schema` | No | Also generate JSON Schema to this path |
| `--config` | No | Path to batch config YAML file |

### `inspect`

Display registry contents with config fields.

```bash
conscribe inspect \
  --registrar "my_app.llm._registrar:LLMRegistrar" \
  --discover "my_app.llm.providers"
```

Output:

```
Layer: llm
Discriminator: provider
Registered keys (2):

  anthropic (ChatAnthropic):
    model_id: str
    max_tokens: int = 4096

  open_ai (ChatOpenAI):
    model_id: str  # Model identifier, e.g. gpt-4o
    temperature: float = 0.0  # Sampling temperature, 0-2
```

### `update-stubs`

Force regenerate all stubs, ignoring the fingerprint cache.

```bash
conscribe update-stubs --config conscribe.yaml
```

## Batch Config File

For projects with multiple layers, use a YAML config file:

```yaml
# conscribe.yaml
discover:
  - my_app.agents
  - my_app.llm.providers
  - my_app.evaluators

output_dir: generated

layers:
  - registrar: my_app.agents._registrar:AgentRegistrar
    output: generated/agent_config.py
    json_schema: generated/agent_config.schema.json

  - registrar: my_app.llm._registrar:LLMRegistrar
    output: generated/llm_config.py
    json_schema: generated/llm_config.schema.json
```

Then run:

```bash
conscribe generate-config --config conscribe.yaml
```

This discovers all packages, then generates stubs for each layer.

## Programmatic Equivalent

Every CLI command has a Python equivalent:

```python
from conscribe import discover, build_layer_config
from conscribe import generate_layer_config_source, generate_layer_json_schema

# discover
discover("my_app.llm.providers")

# generate-config
result = build_layer_config(LLMRegistrar)
source = generate_layer_config_source(result)
schema = generate_layer_json_schema(result)

# inspect
for key in LLMRegistrar.keys():
    cls = LLMRegistrar.get(key)
    print(f"{key}: {cls.__name__}")
```
