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

# custom query for image/text files
kia exec --model <name> "Please describe the following image @path/to/image.png."
kia exec --model <name> "Please summarize the following file @path/to/file.txt."
```

## Tools

We support defining tools as python functions with strictly formatted docstrings:

```python
def function_name(param1: type1, param2: type2, ...):
    """ <one-line description>
    <more detailed description>
    Args:
       <param1>: <param1_description>
       <param2>: <param2_description>
       ...
    """
    # implementation
    # it should return a simple string, or the openai content dict, e.g. {"type": "text", "text": "..."}
    # after the string, optionally an image content dict, e.g. {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    return "..."

# an example
def read_file(file_path: str):
    """Read a file from the local filesystem.
    Args:
       file_path: the path to the file to read.
    """
    with open(file_path, "r") as f:
        return f.read()
```

You can implement your own toolset in a python file, and use `--tools <path/to/tools.py>` to load the tools into the agent.

We also provide some built-in toolsets and prompts (under `kiui/agent/prompts/` and `kiui/agent/tools/`), such as a general-purpose coder adapted from codex:
```bash
# use gpt-5 series models for the best performance as we use apply_patch tool.
kia exec --model gpt-5.2 --tools coder --system_prompt coder --verbose "Please review the project under the current directory and write a summary in summary.md"
```

**WARNING:** It will not ask for confirmation before executing shell command or editing files, use at your own risk!