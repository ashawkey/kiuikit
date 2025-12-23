import os
import re
import time
import json
import importlib
import inspect
from typing import Callable, Literal
from openai import OpenAI
from rich.console import Console
from rich.theme import Theme

from kiui.agent.terminal import TerminalInput
from kiui.agent.utils import parse_tool_docstring, get_text_content_dict, get_image_content_dict

def parse_custom_query(query: str) -> list:
    """Parse the user query into formatted content, optionally load images or text files (by using @filename).
    Supported file types:
    - @image.png/jpg/jpeg: load an image
    - @file.txt/md/json/yaml: read a local text file
    e.g. "Please describe the following image @path/to/image.png, and tell me the weather." will be parsed into: 
    [
        {"type": "text", "text": "Please describe the following image"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...", "detail": "low"}},
        {"type": "text", "text": "and tell me the weather."}
    ]
    """
    content = []

    # Match @filepath where filepath contains typical path characters
    pattern = re.compile(r"@([\w./-]+)")
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
    text_extensions = {'.txt', '.md', '.json', '.yaml', '.yml'}
    
    last_end = 0
    for match in pattern.finditer(query):
        # text before this tag
        pre_text = query[last_end:match.start()].strip()
        if pre_text:
            content.append(get_text_content_dict(pre_text))

        file_path = match.group(1)
        _, ext = os.path.splitext(file_path.lower())
        
        if ext in image_extensions:
            content.append(get_image_content_dict(file_path))
        elif ext in text_extensions:
            with open(file_path, "r") as f:
                file_content = f.read()
            content.append(get_text_content_dict(file_content))
        else:
            # Unknown extension, keep as literal text
            content.append(get_text_content_dict(f"@{file_path}"))
            
        last_end = match.end()

    # trailing text after the last tag
    tail = query[last_end:].strip()
    if tail:
        content.append(get_text_content_dict(tail))

    return content

class ContextManager:
    """Context manager for flexible conversation history."""
    def __init__(self, system_prompt: str):
        self.system_prompt = {
            "role": "system",
            "content": [get_text_content_dict(system_prompt)],
        }
        # round-based conversation history, each round can have multiple messages.
        self.rounds: dict = {}
        self.current_round_id = 0
    
    def add(self, content, round_id: int | None = None):
        # if round_id is not provided, use the last round id + 1
        if round_id is None:
            round_id = self.current_round_id
        if round_id not in self.rounds:
            self.rounds[round_id] = []
        
        self.rounds[round_id].append(content)

        # update the last round id
        self.current_round_id = round_id + 1
    
    def get(self, num_rounds: int | None = None, include_system: bool = True) -> list:
        res = []
        if include_system:
            res.append(self.system_prompt)
        
        if num_rounds is None:
            num_rounds = len(self.rounds) # all rounds
        
        for round_id in range(self.current_round_id - num_rounds, self.current_round_id):
            res.extend(self.rounds[round_id])
        
        return res
        
        

class LLMAgent:
    def __init__(
        self, 
        model: str,
        api_key: str,
        base_url: str,
        system_prompt: str = "You are a helpful assistant.",
        verbose: bool = True,
        max_tool_iter: int = 20,
        thinking_budget: Literal["low", "medium", "high"] = "low",
    ):

        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.thinking_budget = thinking_budget

        self.context = ContextManager(system_prompt)

        self.available_functions: dict = {}
        self.tools: list = []
        
        self.tool_iter = 0
        self.max_tool_iter = max_tool_iter # to avoid infinite loops

        self.round_id = 0

        # running token usage totals across the entire conversation
        self.token_totals = {
            "total": 0,
            "prompt": 0,
            "cached_prompt": 0,
            "completion": 0,
            "reasoning": 0,
        }

        # Rich console with color theme for different kinds of logs
        self.console = Console(
            theme=Theme(
                {
                    "debug": "dim cyan",
                    "input": "bold yellow",
                    "response": "bold green",
                    "error": "bold red",
                    "system": "bold blue",
                }
            )
        )
        self.console.print(f"[SYSTEM] Created Agent with model: {model}", style="system", markup=False)
        self.console.print(f"[SYSTEM] System prompt: {system_prompt[:100]}...", style="system", markup=False)

    # tool usage helpers
    def add_tool(
        self, name: str, func: Callable, description: str, parameters: dict
    ):
        if self.verbose:
            self.console.print(f"[DEBUG] Adding tool {name}: {description[:30]}...", style="debug", markup=False)

        self.available_functions[name] = {
            "function": func,
            "schema": {
                "type": "function",
                "function": {"name": name, "description": description, "parameters": parameters},
            },
        }

        self.tools.append(self.available_functions[name]["schema"])


    def load_tools(self, module_path: str):

        if self.verbose:
            self.console.print(f"[DEBUG] Loading tools from {module_path}", style="debug", markup=False)
    
        spec = importlib.util.spec_from_file_location("tools_module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        loaded_functions: list[str] = []

        # load all functions from the module
        for name in module.__all__:
            func = getattr(module, name)
            if callable(func):
                # parse the function signature and docstring
                sig = inspect.signature(func)
                docstring = func.__doc__
                description, param_description = parse_tool_docstring(docstring)

                properties = {}
                required_params = []

                for param_name, param in sig.parameters.items():
                    # param type
                    param_type = "string"
                    if param.annotation == int:
                        param_type = "integer"
                    elif param.annotation == float:
                        param_type = "number"
                    elif param.annotation == bool:
                        param_type = "boolean"
                    # param description
                    properties[param_name] = {
                        "type": param_type,
                        "description": f"Parameter {param_name}: {param_description[param_name]}",
                    }
                    # required params
                    if param.default == inspect.Parameter.empty:
                        required_params.append(param_name)
                
                parameters = {
                    "type": "object",
                    "properties": properties,
                    "required": required_params,
                }   

                self.add_tool(name, func, description, parameters)
                loaded_functions.append(name)
        
        if len(self.available_functions) == 0:
            self.console.print(f"[ERROR] No tools loaded from {module_path}!", style="error", markup=False)
        else:
            self.console.print(f"[SYSTEM] Loaded {len(loaded_functions)} tools from {module_path}", style="system", markup=False)


    def call_api(self, use_tools: bool = True):
        """Call the API using current context."""

        if self.verbose:
            self.console.print(f"[DEBUG] Calling API (round: {self.round_id}, tool iter: {self.tool_iter})", style="debug", markup=False)

        # build the messages
        messages = self.context.get()

        kwargs = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        if use_tools and len(self.tools) > 0:
            kwargs["tools"] = self.tools
            kwargs["tool_choice"] = "auto"
        
        # thinking budget (model-specific, unfortunately hard-coded...)
        if "gpt-5" in self.model.lower():
            kwargs["reasoning_effort"] = self.thinking_budget
        elif "gemini" in self.model.lower():
            kwargs["extra_body"] = {
                "extra_body": {
                    "google": {
                        "thinking_config": {
                            "thinking_budget": "low" if self.thinking_budget == "low" else "high", # only low/high
                            "include_thoughts": True,
                        },
                    },
                },
            }
            

        # call the API
        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        usage = response.usage

        # accumulate token usage totals
        self.token_totals["total"] += usage.total_tokens
        self.token_totals["prompt"] += usage.prompt_tokens
        if usage.prompt_tokens_details and usage.prompt_tokens_details.cached_tokens:
            self.token_totals["cached_prompt"] += usage.prompt_tokens_details.cached_tokens
        self.token_totals["completion"] += usage.completion_tokens
        if usage.completion_tokens_details and usage.completion_tokens_details.reasoning_tokens:
            self.token_totals["reasoning"] += usage.completion_tokens_details.reasoning_tokens

        # append the assistant message to the context
        self.context.add(message, round_id=self.round_id)

        # log the usage
        if self.verbose:
            self.console.print(
                f"[DEBUG] API Response total_tokens: {usage.total_tokens} = "
                f"output: {usage.completion_tokens} "
                f"(reasoning: {usage.completion_tokens_details.reasoning_tokens if usage.completion_tokens_details and usage.completion_tokens_details.reasoning_tokens else 'N/A'}) "
                f"input: {usage.prompt_tokens} "
                f"(cached: {usage.prompt_tokens_details.cached_tokens if usage.prompt_tokens_details and usage.prompt_tokens_details.cached_tokens else 'N/A'})", 
                style="debug",
                markup=False,
            )
            if message.tool_calls:
                self.console.print(f"[DEBUG] Requested tool calls: {len(message.tool_calls)}", style="debug", markup=False)

        return message


    def execute_tool_calls(self, tool_calls: list):
        """Execute the tool calls and update the conversation history."""

        for i, tool_call in enumerate(tool_calls):
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            if self.verbose:
                self.console.print(f"[DEBUG] Tool call {i+1}/{len(tool_calls)}: {function_name}({function_args})", style="debug", markup=False)

            # Execute the function
            try:
                func = self.available_functions[function_name]["function"]
                result = func(**function_args) # a string or a list of message dict

                # Append tool result to conversation history
                if isinstance(result, str):
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": [get_text_content_dict(result)],
                    }
                    self.context.add(tool_message, round_id=self.round_id)
                else:
                    # special case for tools that also return an image, we need an extra user message to upload the image.
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": [get_text_content_dict(result[0])],
                    }
                    self.context.add(tool_message, round_id=self.round_id)
                    user_message = {
                        "role": "user",
                        "content": result[1], # the image content dict
                    }
                    self.context.add(user_message, round_id=self.round_id)
                # don't print tool message if it succeeds (as it may return long content or an image)
                
            except Exception as e:
                # if the function call fails (maybe due to wrong usage), we add an error message to the conversation history so the LLM can try again
                error_msg = f"Error executing {function_name}: {str(e)}"
                error_result_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": [get_text_content_dict(error_msg)],
                }

                self.context.add(error_result_message, round_id=self.round_id)
                self.console.print(f"[ERROR] Tool failed: {error_msg}", style="error", markup=False)
        
        
    def get_response(self):
        """Process the context and update the current response."""
       
        # increment the current iteration
        use_tools = True
        if self.tool_iter >= self.max_tool_iter:
            self.console.print(f"[ERROR] tool call iterations reached ({self.max_tool_iter}), force not using tools in this iteration.", style="error", markup=False)
            use_tools = False

        # call the API
        message = self.call_api(use_tools=use_tools)

        if message.content:
            self.console.print(f"{message.content}", style="response", markup=False)
        
        # if there are tool calls, we need to execute them and call the API recursively
        if message.tool_calls:
            self.tool_iter += 1 
            self.execute_tool_calls(message.tool_calls)
            self.get_response()
        
        return message.content if message.content else None

    
    def chat_loop(self):
            
        self.console.print("[SYSTEM] Type 'quit' or 'exit' to exit.", style="system", markup=False)
        self.console.print("[SYSTEM] `Enter` to send. `Escape` then `Enter` for a newline.", style="system", markup=False)
        self.console.print("[SYSTEM] Current working directory: " + os.getcwd(), style="system", markup=False)
        
        terminal = TerminalInput(history_path=os.path.expanduser("~/.agent_history"))

        while True:
            # use prompt_toolkit session for rich, multiline input
            query = terminal.prompt().strip()
            # exit logic
            if query.lower() in ["quit", "exit"]:
                break
            # parse user query and append to conversation history
            user_message = {"role": "user", "content": parse_custom_query(query)}
            self.context.add(user_message, round_id=self.round_id)
            # call API for the response (may be multi-turn with tool calls)
            self.tool_iter = 0
            self.get_response()
            self.round_id += 1
        # print total token usage upon exit
        self.console.print(
            f"[SYSTEM] Total tokens used: {self.token_totals['total']} (input: {self.token_totals['prompt']}, cached input: {self.token_totals['cached_prompt']}, output: {self.token_totals['completion']}, reasoning: {self.token_totals['reasoning']})",
            style="system",
            markup=False,
        )
        
    def execute(self, query: str):
        # exec mode, no interactive terminal and just finish the task.
        self.console.print(f"[SYSTEM] Executing query: {query}", style="system", markup=False)
        t0 = time.time()

        user_message = {"role": "user", "content": parse_custom_query(query)}
        self.context.add(user_message, round_id=self.round_id)
        self.get_response()
        
        t1 = time.time()
        self.console.print(f"[SYSTEM] Execution time: {t1 - t0:.2f} seconds", style="system", markup=False)
        # print total token usage upon exit
        self.console.print(
            f"[SYSTEM] Total tokens used: {self.token_totals['total']} (input: {self.token_totals['prompt']}, cached input: {self.token_totals['cached_prompt']}, output: {self.token_totals['completion']}, reasoning: {self.token_totals['reasoning']})",
            style="system",
            markup=False,
        )