"""Persistent Chrome DevTools Protocol tools for the browser skill."""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

_MAX_OUTPUT_CHARS = 20_000
_DEFAULT_CDP_URL = "http://127.0.0.1:9222"


class CDPConnection:
    def __init__(self, url: str):
        try:
            from websockets.sync.client import connect
        except ImportError as exc:
            raise RuntimeError("browser tools require the 'websockets' package") from exc

        self.endpoint = _resolve_endpoint(url)
        self._ws = connect(
            self.endpoint,
            open_timeout=30,
            close_timeout=3,
            max_size=16 * 1024 * 1024,
        )
        self._lock = threading.Lock()
        self._next_id = 0
        self._sessions: dict[str, str] = {}
        self.target_id: str | None = None

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass

    def send(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        session_id: str | None = None,
        timeout: float = 15,
    ) -> dict[str, Any]:
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            message: dict[str, Any] = {"id": request_id, "method": method}
            if params:
                message["params"] = params
            if session_id:
                message["sessionId"] = session_id
            self._ws.send(json.dumps(message))

            while True:
                response = json.loads(self._ws.recv(timeout=timeout))
                if response.get("id") != request_id:
                    continue
                if "error" in response:
                    error = response["error"]
                    raise RuntimeError(error.get("message", str(error)))
                return response.get("result", {})

    def targets(self) -> list[dict[str, Any]]:
        infos = self.send("Target.getTargets").get("targetInfos", [])
        return [target for target in infos if target.get("type") == "page"]

    def resolve_target(self, tab_id: str) -> str:
        matches = [
            target["targetId"]
            for target in self.targets()
            if target["targetId"] == tab_id or target["targetId"].endswith(tab_id)
        ]
        if len(matches) != 1:
            raise ValueError(f"tab_id matched {len(matches)} tabs; use a unique id from browser_status")
        return matches[0]

    def select(self, target_id: str) -> None:
        self.send("Target.activateTarget", {"targetId": target_id})
        self.target_id = target_id

    def _session_for(self, target_id: str) -> str:
        if target_id not in self._sessions:
            result = self.send("Target.attachToTarget", {"targetId": target_id, "flatten": True})
            self._sessions[target_id] = result["sessionId"]
        return self._sessions[target_id]

    def current_target(self) -> dict[str, Any]:
        targets = self.targets()
        ids = {target["targetId"] for target in targets}
        if self.target_id not in ids:
            if not targets:
                created = self.send("Target.createTarget", {"url": "about:blank"})
                self.target_id = created["targetId"]
                targets = self.targets()
            else:
                self.target_id = targets[0]["targetId"]
                for target in targets:
                    result = self.send(
                        "Runtime.evaluate",
                        {"expression": "document.visibilityState", "returnByValue": True},
                        self._session_for(target["targetId"]),
                    )
                    if result.get("result", {}).get("value") == "visible":
                        self.target_id = target["targetId"]
                        break
        return next(target for target in targets if target["targetId"] == self.target_id)

    def session(self) -> str:
        return self._session_for(self.current_target()["targetId"])

    def evaluate(self, expression: str, *, await_promise: bool = False) -> Any:
        result = self.send(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
                "awaitPromise": await_promise,
                "userGesture": True,
            },
            self.session(),
        )
        if "exceptionDetails" in result:
            details = result["exceptionDetails"]
            raise RuntimeError(details.get("text", "JavaScript evaluation failed"))
        remote = result.get("result", {})
        if remote.get("subtype") == "error":
            raise RuntimeError(remote.get("description", "JavaScript evaluation failed"))
        return remote.get("value")


def _resolve_endpoint(url: str) -> str:
    if url.startswith(("ws://", "wss://")):
        return url
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    parsed = urllib.parse.urlsplit(url)
    path = parsed.path.rstrip("/")
    if not path.endswith("/json/version"):
        path += "/json/version"
    version_url = urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment)
    )
    try:
        with urllib.request.urlopen(version_url, timeout=5) as response:
            version = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            scheme = "wss" if parsed.scheme == "https" else "ws"
            return urllib.parse.urlunsplit(
                (scheme, parsed.netloc, "/devtools/browser", parsed.query, parsed.fragment)
            )
        raise
    endpoint = version.get("webSocketDebuggerUrl")
    if not endpoint:
        raise RuntimeError(f"CDP endpoint returned no webSocketDebuggerUrl: {version_url}")
    return endpoint


def _cdp_server(url: str) -> tuple[bool, str | None, int | None]:
    """Return the transport security, host, and port identifying a CDP server."""
    if not url.startswith(("http://", "https://", "ws://", "wss://")):
        url = f"http://{url}"
    parsed = urllib.parse.urlsplit(url)
    secure = parsed.scheme in {"https", "wss"}
    port = parsed.port or (443 if secure else 80)
    host = parsed.hostname
    if host in {"localhost", "127.0.0.1", "::1"}:
        host = "loopback"
    return secure, host, port


def _discover_cdp_url(explicit: str | None) -> str:
    if explicit:
        return explicit
    configured = os.environ.get("KIA_BROWSER_CDP_URL")
    if configured:
        return configured

    candidates = [
        Path.home() / ".config/google-chrome/DevToolsActivePort",
        Path.home() / ".config/chromium/DevToolsActivePort",
        Path.home() / "Library/Application Support/Google/Chrome/DevToolsActivePort",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/User Data/DevToolsActivePort",
    ]
    for candidate in candidates:
        if candidate.is_file():
            lines = candidate.read_text(encoding="utf-8").splitlines()
            if len(lines) >= 2:
                return f"ws://127.0.0.1:{lines[0]}{lines[1]}"
    return _DEFAULT_CDP_URL


def _connection(executor, cdp_url: str | None = None) -> CDPConnection:
    connection = getattr(executor, "_browser_connection", None)
    if connection is not None:
        if cdp_url and _cdp_server(connection.endpoint) != _cdp_server(cdp_url):
            executor.close_tool_resource("browser")
            connection = None
        else:
            return connection

    url = _discover_cdp_url(cdp_url)
    try:
        connection = CDPConnection(url)
    except Exception as exc:
        raise RuntimeError(
            f"Could not attach to Chrome at {url}: {exc}. Enable remote debugging in "
            "chrome://inspect/#remote-debugging or pass its CDP URL to browser_status."
        ) from exc
    executor._browser_connection = connection
    resource = connection

    def cleanup() -> None:
        resource.close()
        if getattr(executor, "_browser_connection", None) is resource:
            del executor._browser_connection

    executor.register_tool_resource("browser", cleanup)
    return connection


def _tabs(connection: CDPConnection) -> list[dict[str, Any]]:
    current = connection.current_target()["targetId"]
    return [
        {
            "tab_id": target["targetId"],
            "active": target["targetId"] == current,
            "title": target.get("title", "")[:500],
            "url": target.get("url", "")[:2000],
        }
        for target in connection.targets()
    ]


def _ok(*, ui_summary: str | None = None, **values: Any) -> dict[str, Any]:
    result = {**values, "success": True}
    if ui_summary:
        result["_ui_summary"] = ui_summary
    return result


def _label(text: str, fallback: str = "") -> str:
    value = " ".join((text or fallback).split())
    return value if len(value) <= 100 else value[:97] + "..."


def browser_status(executor, cdp_url: str | None = None) -> dict[str, Any]:
    """Attach if necessary and report the selected page and open tabs."""
    executor.console.tool("browser_status")
    connection = _connection(executor, cdp_url)
    tabs = _tabs(connection)
    active = next(tab for tab in tabs if tab["active"])
    summary = f"Attached to Chrome · {len(tabs)} tabs · active: {_label(active['title'], active['url'])}"
    return _ok(
        ui_summary=summary,
        connected=True,
        endpoint=connection.endpoint,
        active=active,
        tabs=tabs,
    )


def browser_open(executor, url: str, new_tab: bool = False) -> dict[str, Any]:
    """Navigate the selected tab or create a visible tab."""
    action = "new tab" if new_tab else "current tab"
    executor.console.tool(f"browser_open {url} ({action})")
    connection = _connection(executor)
    if new_tab:
        target_id = connection.send("Target.createTarget", {"url": url})["targetId"]
        connection.select(target_id)
    else:
        target_id = connection.current_target()["targetId"]
        result = connection.send("Page.navigate", {"url": url}, connection.session(), timeout=30)
        if result.get("errorText"):
            raise RuntimeError(result["errorText"])
    summary = f"Opened new tab: {_label(url)}" if new_tab else f"Navigated current tab: {_label(url)}"
    return _ok(ui_summary=summary, tab_id=target_id, url=url, new_tab=new_tab)


_OBSERVE_SCRIPT = r"""
(() => {
  const maxElements = __MAX_ELEMENTS__;
  const maxText = __MAX_TEXT__;
  const roots = [document];
  const all = [];
  for (let r = 0; r < roots.length; r++) {
    const root = roots[r];
    for (const el of root.querySelectorAll('*')) {
      all.push(el);
      if (el.shadowRoot) roots.push(el.shadowRoot);
    }
  }
  const visible = (el) => {
    const s = getComputedStyle(el), rect = el.getBoundingClientRect();
    return s.display !== 'none' && s.visibility !== 'hidden' && Number(s.opacity) !== 0 &&
      rect.width > 0 && rect.height > 0 && rect.bottom >= 0 && rect.right >= 0 &&
      rect.top <= innerHeight && rect.left <= innerWidth;
  };
  const roles = new Set(['button','link','textbox','checkbox','radio','combobox','listbox','option','menuitem','tab','switch','slider','spinbutton']);
  const candidates = all.filter(el => {
    const tag = el.tagName.toLowerCase();
    const role = (el.getAttribute('role') || '').toLowerCase();
    return visible(el) && !el.disabled && (
      ['a','button','input','textarea','select','summary'].includes(tag) ||
      el.isContentEditable || roles.has(role) || el.tabIndex >= 0 || typeof el.onclick === 'function'
    );
  });
  const interactive = candidates.slice(0, maxElements);
  window.__kiaBrowserElements = interactive;
  const elements = interactive.map((el, i) => {
    const rect = el.getBoundingClientRect();
    const text = (el.innerText || el.value || el.getAttribute('aria-label') || el.getAttribute('title') || '').trim().replace(/\s+/g, ' ').slice(0, 200);
    return {
      index: i + 1,
      tag: el.tagName.toLowerCase(),
      role: el.getAttribute('role') || undefined,
      type: el.getAttribute('type') || undefined,
      text,
      placeholder: el.getAttribute('placeholder') || undefined,
      href: el.href?.slice(0, 1000) || undefined,
      rect: {x: Math.round(rect.x), y: Math.round(rect.y), width: Math.round(rect.width), height: Math.round(rect.height)}
    };
  });
  return {
    url: location.href,
    title: document.title,
    text: (document.body?.innerText || '').trim().replace(/\n{3,}/g, '\n\n').slice(0, maxText),
    elements,
    elements_truncated: candidates.length > interactive.length,
    scroll: {x: scrollX, y: scrollY, viewport_width: innerWidth, viewport_height: innerHeight,
             page_width: document.documentElement.scrollWidth, page_height: document.documentElement.scrollHeight}
  };
})()
"""


def browser_observe(
    executor,
    max_elements: int = 150,
    max_text_chars: int = 12000,
) -> dict[str, Any]:
    """Return page text and indexed visible interactive elements."""
    if not 1 <= max_elements <= 500:
        raise ValueError("max_elements must be between 1 and 500")
    if not 0 <= max_text_chars <= _MAX_OUTPUT_CHARS:
        raise ValueError(f"max_text_chars must be between 0 and {_MAX_OUTPUT_CHARS}")
    executor.console.tool("browser_observe")
    connection = _connection(executor)
    script = _OBSERVE_SCRIPT.replace("__MAX_ELEMENTS__", str(max_elements)).replace(
        "__MAX_TEXT__", str(max_text_chars)
    )
    result = connection.evaluate(script)
    while result["elements"] and len(json.dumps(result, ensure_ascii=False)) > _MAX_OUTPUT_CHARS:
        result["elements"].pop()
        result["elements_truncated"] = True
    remaining = _MAX_OUTPUT_CHARS - len(json.dumps({**result, "text": ""}, ensure_ascii=False))
    result["text"] = result["text"][: max(0, remaining)]
    count = len(result["elements"])
    count_label = f"{count}+" if result["elements_truncated"] else str(count)
    summary = (
        f"Observed {_label(result['title'], result['url'])} · "
        f"{count_label} interactive elements · {len(result['text']):,} text chars"
    )
    return _ok(ui_summary=summary, **result)


def _element_geometry(connection: CDPConnection, index: int) -> dict[str, Any]:
    script = f"""
(() => {{
  const el = window.__kiaBrowserElements?.[{index - 1}];
  if (!el || !el.isConnected) return {{error: 'stale'}};
  el.scrollIntoView({{block: 'center', inline: 'center'}});
  const r = el.getBoundingClientRect();
  const text = (el.innerText || el.value || el.getAttribute('aria-label') || el.getAttribute('title') || '').trim().replace(/\s+/g, ' ').slice(0, 200);
  return {{x: r.left + r.width / 2, y: r.top + r.height / 2, tag: el.tagName.toLowerCase(), text, disabled: !!el.disabled}};
}})()
"""
    geometry = connection.evaluate(script)
    if not geometry or geometry.get("error") == "stale":
        raise RuntimeError("Element index is stale; call browser_observe again")
    if geometry.get("disabled"):
        raise RuntimeError("Element is disabled")
    return geometry


def browser_click(executor, index: int) -> dict[str, Any]:
    """Perform a real mouse click on an element from the latest observation."""
    if index < 1:
        raise ValueError("index must be positive")
    executor.console.tool(f"browser_click #{index}")
    connection = _connection(executor)
    geometry = _element_geometry(connection, index)
    session = connection.session()
    point = {"x": geometry["x"], "y": geometry["y"], "button": "left", "clickCount": 1}
    connection.send("Input.dispatchMouseEvent", {**point, "type": "mousePressed"}, session)
    connection.send("Input.dispatchMouseEvent", {**point, "type": "mouseReleased"}, session)
    description = f"{geometry['tag']} #{index}"
    if geometry["text"]:
        description += f": {_label(geometry['text'])}"
    return _ok(
        ui_summary=f"Clicked {description}",
        index=index,
        tag=geometry["tag"],
        text=geometry["text"],
        clicked=True,
    )


def browser_type(executor, index: int, text: str, clear: bool = True) -> dict[str, Any]:
    """Focus an observed editable element and insert text."""
    if index < 1:
        raise ValueError("index must be positive")
    executor.console.tool(f"browser_type #{index} ({len(text)} chars)")
    connection = _connection(executor)
    script = f"""
(() => {{
  const el = window.__kiaBrowserElements?.[{index - 1}];
  if (!el || !el.isConnected) return {{error: 'stale'}};
  const editable = el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement || el.isContentEditable;
  if (!editable || el.disabled || el.readOnly) return {{error: 'not_editable'}};
  el.scrollIntoView({{block: 'center', inline: 'center'}});
  el.focus();
  if ({json.dumps(clear)}) {{
    if (el.isContentEditable) {{ el.textContent = ''; }}
    else if ('value' in el) {{
      const proto = el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
      const setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
      if (setter) setter.call(el, ''); else el.value = '';
    }}
    el.dispatchEvent(new InputEvent('input', {{bubbles: true, inputType: 'deleteContentBackward'}}));
  }}
  return {{tag: el.tagName.toLowerCase()}};
}})()
"""
    result = connection.evaluate(script)
    if not result or result.get("error") == "stale":
        raise RuntimeError("Element index is stale; call browser_observe again")
    if result.get("error") == "not_editable":
        raise RuntimeError("Element is not an editable input, textarea, or contenteditable element")
    if text:
        connection.send("Input.insertText", {"text": text}, connection.session())
    connection.evaluate(
        "document.activeElement?.dispatchEvent(new Event('change', {bubbles: true})); true"
    )
    cleared = ", cleared first" if clear else ""
    return _ok(
        ui_summary=f"Typed {len(text)} characters into {result['tag']} #{index}{cleared}",
        index=index,
        tag=result["tag"],
        typed=True,
        characters=len(text),
        cleared=clear,
    )


def browser_scroll(
    executor,
    direction: str = "down",
    pages: float = 1.0,
    index: int | None = None,
) -> dict[str, Any]:
    """Scroll the page or an observed scrollable element."""
    if direction not in {"up", "down", "left", "right"}:
        raise ValueError("direction must be up, down, left, or right")
    if not 0.1 <= pages <= 10:
        raise ValueError("pages must be between 0.1 and 10")
    target_label = "page" if index is None else f"element #{index}"
    executor.console.tool(f"browser_scroll {target_label} {direction} {pages:g} pages")
    target = "window" if index is None else f"window.__kiaBrowserElements?.[{index - 1}]"
    sign = -1 if direction in {"up", "left"} else 1
    axis = "x" if direction in {"left", "right"} else "y"
    script = f"""
(() => {{
  const target = {target};
  if (!target || (target !== window && !target.isConnected)) return {{error: 'stale'}};
  const amount = Math.round(({axis == 'x' and 'innerWidth' or 'innerHeight'}) * {pages} * {sign});
  if (target === window) window.scrollBy({{left: {axis == 'x' and 'amount' or '0'}, top: {axis == 'y' and 'amount' or '0'}, behavior: 'auto'}});
  else target.scrollBy({{left: {axis == 'x' and 'amount' or '0'}, top: {axis == 'y' and 'amount' or '0'}, behavior: 'auto'}});
  return {{amount}};
}})()
"""
    result = _connection(executor).evaluate(script)
    if not result or result.get("error") == "stale":
        raise RuntimeError("Element index is stale; call browser_observe again")
    return _ok(
        ui_summary=f"Scrolled {target_label} {direction} {pages:g} pages ({abs(result['amount']):,} px)",
        direction=direction,
        pages=pages,
        index=index,
        pixels=result["amount"],
    )


def browser_extract(
    executor,
    selector: str | None = None,
    include_links: bool = False,
    max_chars: int = _MAX_OUTPUT_CHARS,
) -> dict[str, Any]:
    """Extract bounded visible text and optional links without another LLM."""
    if not 1 <= max_chars <= _MAX_OUTPUT_CHARS:
        raise ValueError(f"max_chars must be between 1 and {_MAX_OUTPUT_CHARS}")
    target = selector or "page"
    executor.console.tool(f"browser_extract {target}")
    script = f"""
(() => {{
  const root = {f'document.querySelector({json.dumps(selector)})' if selector else 'document.body'};
  if (!root) return {{error: 'selector_not_found'}};
  const text = (root.innerText || root.textContent || '').trim().replace(/\\n{{3,}}/g, '\\n\\n');
  const links = {str(include_links).lower()} ? [...root.querySelectorAll('a[href]')].map(a => ({{text: (a.innerText || '').trim().replace(/\\s+/g, ' ').slice(0, 200), url: a.href.slice(0, 1000)}})).slice(0, 200) : [];
  return {{url: location.href, title: document.title, text: text.slice(0, {max_chars}), truncated: text.length > {max_chars}, links}};
}})()
"""
    result = _connection(executor).evaluate(script)
    if result.get("error") == "selector_not_found":
        raise ValueError(f"selector not found: {selector}")
    while result["links"] and len(json.dumps(result, ensure_ascii=False)) > _MAX_OUTPUT_CHARS:
        result["links"].pop()
        result["links_truncated"] = True
    remaining = _MAX_OUTPUT_CHARS - len(json.dumps({**result, "text": ""}, ensure_ascii=False))
    original_text = result["text"]
    result["text"] = original_text[: max(0, remaining)]
    result["truncated"] = result["truncated"] or len(result["text"]) < len(original_text)
    summary = f"Extracted {len(result['text']):,} text chars from {_label(result['title'], result['url'])}"
    if include_links:
        summary += f" · {len(result['links'])} links"
    if result["truncated"]:
        summary += " · truncated"
    return _ok(ui_summary=summary, **result)


def browser_tabs(executor, action: str = "list", tab_id: str | None = None) -> dict[str, Any]:
    """List, switch, or close browser tabs."""
    if action not in {"list", "switch", "close"}:
        raise ValueError("action must be list, switch, or close")
    suffix = f" {tab_id}" if tab_id else ""
    executor.console.tool(f"browser_tabs {action}{suffix}")
    connection = _connection(executor)
    if action == "switch":
        if not tab_id:
            raise ValueError("tab_id is required for switch")
        connection.select(connection.resolve_target(tab_id))
    elif action == "close":
        if not tab_id:
            raise ValueError("tab_id is required for close")
        target_id = connection.resolve_target(tab_id)
        connection.send("Target.closeTarget", {"targetId": target_id})
        connection._sessions.pop(target_id, None)
        if connection.target_id == target_id:
            connection.target_id = None
    tabs = _tabs(connection)
    if action == "list":
        summary = f"Listed {len(tabs)} browser tabs"
    elif action == "switch":
        active = next(tab for tab in tabs if tab["active"])
        summary = f"Switched tab · active: {_label(active['title'], active['url'])}"
    else:
        summary = f"Closed tab · {len(tabs)} tabs remain"
    return _ok(ui_summary=summary, action=action, tabs=tabs)


def browser_stop(executor) -> dict[str, Any]:
    """Detach Kia from Chrome without closing the user's browser."""
    executor.console.tool("browser_stop")
    connection = getattr(executor, "_browser_connection", None)
    if connection is None:
        message = "Browser was already detached"
        return _ok(ui_summary=message, connected=False, message=message)
    executor.close_tool_resource("browser")
    message = "Detached from browser; Chrome remains open"
    return _ok(ui_summary=message, connected=False, message=message)


def _schema(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


TOOLS = [
    {
        "permission": "safe",
        "run": browser_status,
        "schema": _schema(
            "browser_status",
            "Attach to an already-running Chrome browser through CDP and report its active page and tabs.",
            {"cdp_url": {"type": "string", "description": "Optional HTTP or WebSocket CDP endpoint."}},
        ),
    },
    {
        "permission": "risky",
        "run": browser_open,
        "schema": _schema(
            "browser_open",
            "Navigate the current visible browser tab or open a new visible tab.",
            {"url": {"type": "string"}, "new_tab": {"type": "boolean", "default": False}},
            ["url"],
        ),
    },
    {
        "permission": "safe",
        "run": browser_observe,
        "schema": _schema(
            "browser_observe",
            "Read the active page and assign indexes to visible interactive elements for click, type, and scroll.",
            {
                "max_elements": {"type": "integer", "minimum": 1, "maximum": 500, "default": 150},
                "max_text_chars": {"type": "integer", "minimum": 0, "maximum": _MAX_OUTPUT_CHARS, "default": 12000},
            },
        ),
    },
    {
        "permission": "risky",
        "run": browser_click,
        "schema": _schema(
            "browser_click",
            "Click an element index from the latest browser_observe call with a real mouse event.",
            {"index": {"type": "integer", "minimum": 1}},
            ["index"],
        ),
    },
    {
        "permission": "risky",
        "run": browser_type,
        "schema": _schema(
            "browser_type",
            "Type text into an editable element index from the latest browser_observe call.",
            {
                "index": {"type": "integer", "minimum": 1},
                "text": {"type": "string"},
                "clear": {"type": "boolean", "default": True},
            },
            ["index", "text"],
        ),
    },
    {
        "permission": "safe",
        "run": browser_scroll,
        "schema": _schema(
            "browser_scroll",
            "Scroll the page or an indexed scrollable element by viewport-sized pages.",
            {
                "direction": {"type": "string", "enum": ["up", "down", "left", "right"], "default": "down"},
                "pages": {"type": "number", "minimum": 0.1, "maximum": 10, "default": 1},
                "index": {"type": "integer", "minimum": 1, "description": "Optional element index; omit to scroll the page."},
            },
        ),
    },
    {
        "permission": "safe",
        "run": browser_extract,
        "schema": _schema(
            "browser_extract",
            "Extract bounded text and optional links from the active page or a CSS-selected region.",
            {
                "selector": {"type": "string", "description": "Optional CSS selector."},
                "include_links": {"type": "boolean", "default": False},
                "max_chars": {"type": "integer", "minimum": 1, "maximum": _MAX_OUTPUT_CHARS, "default": _MAX_OUTPUT_CHARS},
            },
        ),
    },
    {
        "permission": "risky",
        "run": browser_tabs,
        "schema": _schema(
            "browser_tabs",
            "List, switch, or close tabs in the attached browser. Closing a tab is irreversible.",
            {
                "action": {"type": "string", "enum": ["list", "switch", "close"], "default": "list"},
                "tab_id": {"type": "string", "description": "Required for switch or close; full id or unique suffix."},
            },
        ),
    },
    {
        "permission": "safe",
        "run": browser_stop,
        "schema": _schema(
            "browser_stop",
            "Detach Kia from Chrome without closing the user's browser or tabs.",
            {},
        ),
    },
]
