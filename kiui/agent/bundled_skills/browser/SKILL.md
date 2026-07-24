---
name: browser
description: Attach to an already-running Chrome or Chromium browser and directly navigate, inspect, click, type, scroll, extract page content, and manage tabs through CDP. Use for interactive website tasks where the user should see each agent action in their browser.
compatibility: Requires a Chromium-based browser with remote debugging enabled and the websockets package provided by kiui[kia].
---

# Browser Control

Loading this skill enables:

- `browser_status`: attach to Chrome and report the active page and tabs;
- `browser_open`: navigate the current tab or open a new tab;
- `browser_observe`: inspect visible text and indexed interactive elements;
- `browser_click`: click an element index from the latest observation;
- `browser_type`: type into an indexed editable element;
- `browser_scroll`: scroll the page or an indexed scrollable element;
- `browser_extract`: return bounded page or element text and links;
- `browser_tabs`: list, switch, or close tabs;
- `browser_stop`: detach the agent without closing Chrome.

## Connect

1. Ensure remote debugging is enabled in the user-owned Chrome/Chromium instance. Recent Chrome versions expose this approval at `chrome://inspect/#remote-debugging`.
2. Call `browser_status`. It uses `KIA_BROWSER_CDP_URL` when set, otherwise checks Chrome's `DevToolsActivePort` file and `http://127.0.0.1:9222`.
3. If discovery fails, ask the user for the HTTP or WebSocket CDP endpoint and pass it as `cdp_url` to `browser_status`.

The connection and active-tab selection persist on the Kia tool executor across calls. Never launch, restart, or kill the user's browser. `browser_stop` closes only Kia's CDP connection.

## Operate

1. Call `browser_observe` before interacting. Element indexes belong to that observation and may become stale after navigation or dynamic page updates.
2. Use `browser_click`, `browser_type`, or `browser_scroll` with an observed index.
3. Observe again after every action that can materially change the page. Do not reuse an index after navigation.
4. Use `browser_extract` for larger text reads; use `browser_observe` for choosing interactive elements.
5. Use `browser_tabs(action="list")` only when tab details beyond `browser_status` are needed. Switch explicitly if the intended tab is not current.

The tools interact with the live visible browser. Treat clicks, typing, navigation, and tab closure as real user actions. Before consequential actions such as purchases, submissions, messages, account changes, or destructive operations, follow the user's authorization and Kia's permission policy.

## Limitations

- Only Chromium browsers exposing CDP are supported.
- Element inspection covers the main document and open shadow roots. Cross-origin iframe contents and canvas-only interfaces are not indexed.
- `browser_extract` performs deterministic text extraction; it does not invoke another LLM.
- If an index is stale, observe again rather than retrying it blindly.
