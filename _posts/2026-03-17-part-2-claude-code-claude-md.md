---
title: "CLAUDE.md: Your Project's Constitution"
date: 2026-03-17 09:00:00 +0300
categories: tutorials
tags: claude-code cli
series: "Building With Claude Code"
series_order: 2
---

In the last post, we built Markd — a bookmarks API with a CLI — by just prompting Claude Code in an empty folder. It worked. Everything ran. But if you looked closely, the codebase had drift: inconsistent error handling, response shapes that varied between endpoints, and decisions Claude made silently that you never discussed.

Today we fix that with a single file.

### The file Claude reads first

When Claude Code opens a project, one of the first things it does is look for `CLAUDE.md` files. If it finds one at the root of your project, it reads it before doing anything else. Think of it as the onboarding document you'd hand a new developer on their first day — except Claude re-reads it every single session.

Let's write one and immediately see the difference.

### Writing your first CLAUDE.md

Open your `markd` project in Claude Code and ask it to help:

```
Look at the current codebase. I want to write a CLAUDE.md that
captures the conventions and patterns this project uses, so that
future code is consistent. Look at the existing code and help me
draft one.
```

Claude will scan your files and propose something. It'll probably be decent but too long — a common first instinct is to document *everything*. That's the wrong move, and we'll talk about why in a moment.

Instead of accepting whatever Claude drafts wholesale, let's think about what actually belongs here. Open a `CLAUDE.md` at the project root and write it yourself, using what Claude found as input. Here's roughly what a good one looks like for Markd at this stage:

```markdown
# Markd

A personal bookmarks manager. FastAPI + SQLite REST API with a Typer CLI.

## Project structure

- `src/markd/` — API package (routes, models, database)
- `cli/` — CLI client (Typer)
- `tests/` — pytest tests

## Development

- Python 3.11+, dependencies in pyproject.toml
- Run API: `uvicorn src.markd.main:app --reload`
- Run tests: `pytest`
- Lint: `ruff check . --fix`

## Conventions

### API patterns
- All endpoints return Pydantic response models, never raw dicts.
- Error responses use `HTTPException` with this shape:
  `{"detail": "Human-readable message"}`.
- Database access goes through functions in `db.py`, not inline SQL in routes.
- Route functions use FastAPI dependency injection for the DB connection.

### Models
- Request bodies use a `Create` or `Update` prefix (e.g. `CreateBookmark`).
- Response models match the DB schema plus computed fields.
- `Update` models make all fields optional for PATCH support.

### Testing
- Use pytest with httpx `AsyncClient` via the `client` fixture in `conftest.py`.
- Every endpoint gets at least: one happy path test, one validation error test,
  and one not-found test.

### Git
- Commit messages: imperative mood, under 72 chars. E.g. "Add search endpoint".
- Don't commit the SQLite database file or __pycache__.
```

That's about 40 lines. Let's talk about what's in there, what's not, and why.

### The mental model: always-true vs. sometimes-relevant

The most common mistake with CLAUDE.md is treating it like a dumping ground. People cram in everything: coding patterns, deployment procedures, API documentation, style preferences, troubleshooting tips, architectural decision records. The file balloons to 500+ lines and, paradoxically, becomes less effective — Claude has to sift through a wall of text to find what's relevant to the current task, and important rules get buried.

Here's the filter: **CLAUDE.md is for things that are always true, regardless of what task Claude is working on.**

If Claude is adding an endpoint, it needs to know your error handling conventions. If Claude is writing a test, it needs to know your testing patterns. If Claude is making a commit, it needs to know your commit message format. These are always-on rules — no matter what the task is, these conventions apply.

Things that are *sometimes* relevant — how to deploy, how to run database migrations, your API specification, the architecture decision log — belong somewhere else. We'll get to where in a moment.

A useful thought experiment: if you forced yourself to keep CLAUDE.md under 200 lines, what would survive the cut? That's your actual constitution. Everything else is reference material.

### The test: redo a task from last time

Now let's see if it makes a difference. Start a fresh Claude Code session in the same project:

```bash
claude
```

Ask Claude to add an endpoint — the same kind of task that produced drift in Post 1:

```
Add a GET /bookmarks/export endpoint that returns all bookmarks as a
downloadable JSON file.
```

Watch what happens. Claude reads your CLAUDE.md at the start of the session. Now when it writes the endpoint, it should:

- Use a Pydantic response model (not a raw dict)
- Follow the same error handling pattern
- Put database logic in `db.py`
- Use dependency injection for the DB connection

Compare this endpoint's code with the ones from Post 1. Is it more consistent? It should be. The conventions are no longer things Claude has to infer from examples — they're explicitly stated.

Now ask Claude to write tests for the new endpoint:

```
Write tests for the export endpoint.
```

This time, Claude knows to use pytest + httpx `AsyncClient`, to use the `client` fixture, and to write happy path, validation, and not-found tests. You didn't have to explain any of that.

### What doesn't belong in CLAUDE.md

Knowing what to leave *out* is just as important as knowing what to put in. Here are the common things people over-include and why they hurt:

**Code snippets and examples.** They go stale fast. If you include a "here's how a typical route looks" code block and then refactor your routes, the CLAUDE.md example is now lying to Claude. Instead, point Claude at the actual code: "See `src/markd/routes/bookmarks.py` for the canonical endpoint pattern." Claude can read the file; let it get the truth from the source.

**Task-specific workflows.** "When adding a new endpoint, do steps 1-7..." is a workflow, not a convention. It changes depending on context. This is what *skills* are for — we'll build one in the next post.

**Exhaustive API documentation.** Your OpenAPI spec or a reference doc can hold this. CLAUDE.md just needs to say where to find it: "See `docs/api-spec.md` for the full endpoint reference."

**Linting and formatting rules.** If you have a ruff config or a `.editorconfig`, Claude will read those directly. Duplicating them in CLAUDE.md means two sources of truth that can drift apart. Instead: "Linting is configured in `pyproject.toml` under `[tool.ruff]`. Always run `ruff check` before committing."

**Anything you'd want to enforce 100% of the time.** If a rule is critical enough that "Claude followed it 70% of the time" isn't acceptable, CLAUDE.md isn't the right mechanism. Hooks (covered in Post 4) give you deterministic guarantees. CLAUDE.md gives you strong guidance. Know which one you need.

### Reference docs: CLAUDE.md's escape valve

As your project grows, you'll accumulate knowledge that's important but not always-on. The solution is simple: keep it in separate docs and point to it from CLAUDE.md.

Add a section to your CLAUDE.md like this:

```markdown
## Reference docs

These files have detailed information. Read them when working on the
relevant area:

- `docs/api-spec.md` — Full endpoint reference with request/response examples
- `docs/database.md` — Schema details, migration approach, query patterns
- `docs/deployment.md` — How to deploy, environment variables, production config
```

This pattern keeps CLAUDE.md lean while making sure Claude knows where to look when it needs deeper context. When you ask Claude to work on deployment, it'll see the pointer and read the deployment doc. When you ask it to add a feature, it won't waste context on deployment details it doesn't need.

Create the `docs/` directory in your Markd project and move any detailed documentation there. For now, it might just be a database doc:

```
Create a docs/database.md that documents our SQLite schema, how tags
are stored, and the query patterns we use in db.py. Then update
CLAUDE.md to point to it.
```

### Subdirectory CLAUDE.md files

As projects grow, different areas develop their own conventions. The CLI has different patterns than the API. The tests have their own setup. You can add CLAUDE.md files in subdirectories, and Claude reads them when working in that area.

For Markd, this might mean adding one for the CLI:

```
Create a cli/CLAUDE.md that documents the CLI conventions: that we use
Typer, that output should use rich tables for list views, that errors
should print to stderr, and that every command should have a --json
flag for script-friendly output.
```

And one for the tests:

```
Create a tests/CLAUDE.md that documents our test conventions: fixture
usage, how we set up the test database, naming patterns, and that we
always clean up test data between tests.
```

Subdirectory files don't override the root — they *add to* it. Claude sees both when working in that directory. This is how you scale conventions without the root file becoming unwieldy.

A rule of thumb: if you find yourself adding a section to root CLAUDE.md that starts with "When working on the CLI..." or "For tests...", that's a signal it belongs in a subdirectory CLAUDE.md instead.

### The decision framework so far

After this post, you have two tools. Here's when to use each:

| Situation | Tool | Why |
|---|---|---|
| Convention that always applies | CLAUDE.md | Claude reads it every session |
| Convention for a specific area | Subdirectory CLAUDE.md | Only loads when relevant |
| Detailed reference material | Docs pointed from CLAUDE.md | Keeps the constitution lean |
| Specific task workflow | ??? | Next post — this is what skills solve |

That last row is the gap you'll feel next. Your CLAUDE.md says *what* the patterns are, but when you ask Claude to add a new endpoint, it still has to figure out the *steps* — which files to create, what order to do things in, what to check afterward. Sometimes it gets the sequence right, sometimes it doesn't.

### Before next time

Make sure your Markd project has:

- A root CLAUDE.md with your conventions (aim for under 80 lines at this stage)
- At least one reference doc in `docs/` that CLAUDE.md points to
- Optionally, a subdirectory CLAUDE.md for `cli/` or `tests/`

Then try this experiment: **start a new Claude Code session and ask Claude to add a feature you haven't built yet.** Something like:

```
Add a POST /bookmarks/{id}/archive endpoint that soft-deletes a
bookmark by setting an archived_at timestamp. Archived bookmarks
should be excluded from list and search results by default, with an
optional include_archived query parameter to show them.
```

This is a multi-file change — it touches the database schema, the models, multiple routes, and tests. Pay attention to whether Claude follows your conventions consistently across all those files. It should be much better than Post 1. But also notice: you're still explaining the *task* in detail. Claude knows your patterns but doesn't have a repeatable playbook for "add a new feature that touches the DB, routes, and tests."

That's what we fix next.

### What's next

In the next post, we'll build our first skill — a reusable playbook that teaches Claude the *steps* for adding a new endpoint to Markd. We'll cover the key distinction between skills as reference material vs. skills as workflow instructions, when to let Claude auto-invoke a skill vs. requiring explicit triggers, and how skills and CLAUDE.md divide responsibilities.

The goal: Claude should know not just your conventions (CLAUDE.md) but your *processes* (skills).

---

*This is Part 2 of the "Building With Claude Code" series. Previously: [Part 1 — Jump In, Break Things](/tutorials/claude-code/). Next up: [Part 3 — Skills: Teaching Claude How Your Project Works](/tutorials/claude-code/)*
