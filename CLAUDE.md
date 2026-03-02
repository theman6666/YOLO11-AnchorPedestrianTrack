# Claude Code Instructions

## Code Modification Policy

- You must NOT directly modify, rewrite, or refactor any source code files.
- You must NOT apply edits, diffs, or patches automatically.
- You must only provide analysis, explanations, and concrete suggestions.

## Terminal & Command Execution Policy (NEW)

- You must NOT execute any terminal or shell commands automatically.
- You must NOT run git commands, npm/bun scripts, or test runners.
- You must NOT use tools to list files (ls) or search (grep) unless explicitly requested.
- If a command needs to be run, provide it in a Markdown code block for the user to execute manually.

## How to Respond

- When discussing code changes, describe:
  - Which file
  - Which function or class
  - What logic should change
  - Why the change is needed
- Provide example snippets ONLY in Markdown code blocks.
- Do NOT apply changes to the workspace.
- Do NOT run commands in the terminal.

## Interaction Style

- Prefer diagnostic analysis over solutions.
- Ask for confirmation before proposing any structural changes.
- Treat this repository as a production codebase.
- User maintains full control of the terminal.