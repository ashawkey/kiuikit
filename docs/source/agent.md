# Agent

A simple-to-use LLM agent that supports multi-turn tool calling.

## API keys
The API keys are stored in the `~/.kiui.yaml` file, which will be automatically loaded as `kiui.conf`.

```yaml
openai:
  <name>: # alias name for this model
    model: <model_name> # the name used in the API call
    api_key: <api_key>
    base_url: <base_url>
```

## CLI

```bash
python -m kiui.agent.cli --help
# equals
kia --help

# list all available models defined above
kia list

# start interactive chat mode
kia chat --model <name>
kia chat --model <name> --system_prompt <path/to/system_prompt.txt> --verbose

# execute a single query
kia exec --model <name> "Tell me a joke."
```