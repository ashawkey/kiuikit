import os
import argparse
from rich.console import Console
from rich.table import Table

from kiui.config import conf
from kiui.agent.backend import LLMAgent


def get_agent(args) -> LLMAgent | None:
    """Create an LLMAgent from args and config. Returns None if model not found."""
    console = Console()
    openai_conf = conf.get("openai", {})
    
    if args.model not in openai_conf:
        console.print(f"[bold red]Model '{args.model}' not found in ~/.kiui.yaml[/bold red]")
        console.print("Available models:", list(openai_conf.keys()))
        return None
    
    model_conf = openai_conf[args.model]
    
    # Load system prompt from file or use as string
    system_prompt = args.system_prompt
    if os.path.exists(args.system_prompt):
        with open(args.system_prompt, "r") as f:
            system_prompt = f.read()
    
    agent = LLMAgent(
        model=model_conf.get("model", args.model),
        api_key=model_conf.get("api_key", ""),
        base_url=model_conf.get("base_url", ""),
        system_prompt=system_prompt,
        verbose=args.verbose,
    )
    
    if args.tools:
        agent.load_tools(args.tools)
    
    return agent


def add_common_args(parser):
    """Add common arguments for chat and exec commands."""
    parser.add_argument("--model", type=str, required=True, help="Model name from ~/.kiui.yaml")
    parser.add_argument("--tools", type=str, default=None, help="Path to tools module")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.", help="System prompt or path to file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logs")


def cmd_list(args):
    """List all available models from kiui.conf['openai']."""
    console = Console()
    openai_conf = conf.get("openai", {})
    
    if not openai_conf:
        console.print("[bold red]No models found in ~/.kiui.yaml[/bold red]")
        console.print("Please add model configurations under the 'openai' key.")
        return
    
    table = Table(title="Available Models", show_header=True, header_style="bold magenta")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Model", style="green")
    table.add_column("Base URL", style="yellow")
    
    for name, model_conf in openai_conf.items():
        table.add_row(
            name,
            model_conf.get("model", "N/A"),
            model_conf.get("base_url", "N/A"),
        )
    
    console.print(table)


def cmd_chat(args):
    """Start interactive chat mode with the specified model."""
    if agent := get_agent(args):
        agent.chat_loop()


def cmd_exec(args):
    """Execute a single query and return the result."""
    if agent := get_agent(args):
        agent.execute(args.prompt)


def main():
    parser = argparse.ArgumentParser(description="LLM Agent CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # list command
    parser_list = subparsers.add_parser("list", help="List all available models")
    parser_list.set_defaults(func=cmd_list)
    
    # chat command
    parser_chat = subparsers.add_parser("chat", help="Start interactive chat mode")
    add_common_args(parser_chat)
    parser_chat.set_defaults(func=cmd_chat)
    
    # exec command
    parser_exec = subparsers.add_parser("exec", help="Execute a single query")
    parser_exec.add_argument("prompt", type=str, help="The query to execute")
    add_common_args(parser_exec)
    parser_exec.set_defaults(func=cmd_exec)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
    else:
        args.func(args)


if __name__ == "__main__":
    main()
