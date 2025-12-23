from typing import Optional

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
            raise ValidationError(message="Input cannot be empty.")


class TerminalInput:
    def __init__(self, history_path: Optional[str] = None):
        self.style = Style.from_dict({
            "prompt": "bold ansiyellow",
        })

        self.history = FileHistory(history_path) if history_path else None
        self._session = PromptSession(
            message=[("class:prompt", "[QUERY] ")],
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

        # Escape then Enter to insert a newline (instead of sending)
        @kb.add("escape", "enter")
        def _(event):
            event.current_buffer.insert_text("\n")

        return kb

    def prompt(self) -> str:
        return self._session.prompt()