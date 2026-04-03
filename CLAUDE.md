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

<!-- GSD:project-start source:PROJECT.md -->
## Project

**YOLO11 Frontend Refactoring Project**

Refactoring the existing single-file HTML frontend for the YOLO11 pedestrian detection and tracking system into a modern, component-based Vue 3 application with an industrial dark mode UI.

**Core Value:** A maintainable, scalable frontend architecture that preserves all existing detection/tracking functionality while providing a professional user experience.

### Constraints

- **Backend compatibility**: Must work with existing Flask API endpoints unchanged
- **Visual parity**: Preserve all existing functionality (camera feed, image/video upload, results display)
- **Deployment**: Must be deployable alongside Flask backend
- **Performance**: Video stream performance must match existing implementation
- **Dark mode**: Industrial aesthetic with Tailwind CSS
<!-- GSD:project-end -->

<!-- GSD:stack-start source:STACK.md -->
## Technology Stack

Technology stack not yet documented. Will populate after codebase mapping or first phase.
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
