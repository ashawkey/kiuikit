from __future__ import annotations

from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.history import FileHistory
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.validation import Validator, ValidationError
from pygments.lexers.markup import MarkdownLexer


class NonEmptyInputValidator(Validator):
    def validate(self, document):
        if not document.text.strip():
            raise ValidationError(message="Please enter a message.")


class TerminalInput:
    def __init__(
        self,
        history_path: str | Path | None = None,
        prompt_label: str = "[QUERY] ",
    ):
        self._prompt_label = prompt_label
        self.style = Style.from_dict({
            "prompt": "bold ansiyellow",
        })

        self.history = FileHistory(str(history_path)) if history_path else None
        self._session = PromptSession(
            multiline=False,
            style=self.style,
            history=self.history,
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=self._create_keybindings(),
            lexer=PygmentsLexer(MarkdownLexer),
            validator=NonEmptyInputValidator(),
            validate_while_typing=False,
        )

    def _create_keybindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("escape", "enter")
        def _(event):
            event.current_buffer.insert_text("\n")

        return kb

    def prompt(self) -> str:
        """Read a line of input from the terminal.

        Raises:
            KeyboardInterrupt: When the user presses Ctrl+C.
            EOFError: When the user presses Ctrl+D (end of input).
        """
        return self._session.prompt([("class:prompt", self._prompt_label)])