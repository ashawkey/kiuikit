---
name: batch
description: Apply one repeated captioning, classification, extraction, scoring, or summarization task to independent files, rows, or URLs without accumulating results in context.
---

# Batch Processing over Independent Items

Loading this skill enables `run_batch`: it runs one task per item, each in a fresh
context, and appends per-item results to `.kia/batch/<name>.jsonl`. The reply
carries only counts, the output path, and a few sample failures — never the
results.

Use it when the items are **independent**: nothing an item produces should change
how the next one is handled.

## Choose the cheapest mechanism first

1. **One model call per item, no tools needed** (plain captioning or classification
   at scale) — do **not** use `run_batch`. Write a script that calls the API
   directly with a thread pool, start it with the `monitor` skill, and read only
   its summary. That is parallel, resumable, and has no agentic overhead.
2. **A few tool calls per item, same shape every time** — `run_batch`.
3. **Items differ in structure, or need judgment about which tools to use** —
   do not use `run_batch`. For a small number of substantial, self-contained tasks,
   use the `subagent` skill; otherwise handle them in the normal conversation.
4. **Fewer than roughly 20 short items, or items that build on each other** — just
   do them in the normal conversation. Isolation is not worth its overhead.

## Build the item list without paying for it

Anything you type into a tool call stays in the conversation for the rest of the
session, so never write the list out yourself:

- **Do:** build it with a command — `exec_command("ls images/*.png > items.txt")`
  — or use a manifest that already exists, then pass `items_file`.
- **Do not:** paste a long `items` array, `write_file` a long literal list, or
  enumerate with `glob_files` first. All three cost the same as typing it.

Items are identified by their text, so duplicate lines resume as a single item.
Pipe the list through `sort -u` when that matters.

Use `items` only for a short list you already know (max 100).

## Write the task

Each item runs against its own prompt alone, and its turn is discarded
afterwards. So:

- Make `task` **self-contained**. It sees your system prompt and can call every
  tool you can — but not this conversation, not any earlier item, and not the
  instructions of a skill you loaded. If an item needs a skill, tell the task to
  `load_skill` it itself; each item loads it fresh.
- Put `{item}` where the item belongs. Without it the item is appended on its own
  line.
- State the expected output shape explicitly. The final assistant text of the
  turn becomes the item's `result`.
- If items write files, have the task write them itself and keep the returned
  text short — a per-item path or status, not the content.

## Run and report

1. Call `run_batch(task, items_file|items, name, label)`. `name` identifies the
   run and is also its resume key, so pick something descriptive and specific
   (`caption-product-photos`, not `run`) — reusing a name continues that run.
   Results go to `.kia/batch/<name>.jsonl`.
2. Read the returned counts. Per-item results are only in the output file — reach
   for it with `grep_files` or `read_file` when you need them.
3. To retry after an interruption or partial failure, re-issue the **identical**
   call. Successful items are skipped; failed and unattempted ones are retried.
   Only pass `resume=false` when every item must genuinely be redone; it starts
   the run over, keeping the old results as `<name>.jsonl.bak`.
4. Report the counts and the output path. Do not restate per-item results unless
   the user asks for specific ones.
5. If the user wants the results elsewhere or in another format, convert the
   JSONL with a command afterwards.

If the run aborts early after consecutive failures with no successes, the task
instructions or a required tool are wrong. Fix the task and re-run — do not
retry the same call unchanged.

Items run sequentially, so warn the user before starting a run large enough to
take a long time.
