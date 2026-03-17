---
title: "Skills: Teaching Claude How Your Project Works"
date: 2026-03-17 10:00:00 +0300
categories: tutorials
tags: claude-code cli
series: "Building With Claude Code"
series_order: 3
---

If you tried the exercise at the end of Post 2 — adding the archive feature — you probably noticed something. Claude followed your conventions. The Pydantic models were consistent, the error handling matched, the database logic went in the right place. CLAUDE.md did its job.

But the *process* was messy. Maybe Claude updated the routes before creating the database migration function. Maybe it wrote the endpoint but forgot to update the list and search routes to exclude archived bookmarks. Maybe it added the feature but didn't write tests until you reminded it. Every time you ask for a multi-step task, Claude has to reinvent the workflow from scratch.

CLAUDE.md tells Claude what your code should look like. Skills tell Claude *how to build it*.

### What is a skill?

A skill is a markdown file with optional supporting files that Claude can load when it's relevant. At its simplest, it's a `SKILL.md` file inside `.claude/skills/<skill-name>/`. It has a description in its frontmatter, and Claude uses that description to decide whether to load the skill for the current task.

That's the mechanical definition. The more useful one is this: **a skill is a reusable playbook that captures a process you'd otherwise explain in a prompt every time.**

Think about what you repeated across Post 1's prompts. Every time you asked for a new endpoint, you had to think about (or forget to mention) the same things: create a Pydantic model, write the route function, add database logic, write tests. A skill captures that sequence once.

### Two kinds of skills

This is the most important distinction in this post, and one that trips people up early. Skills serve two fundamentally different purposes:

**Reference skills** contain knowledge. They describe patterns, conventions, or domain information that Claude should be aware of when working in a certain area. They're like a specialist handbook — Claude reads them for context, then applies that knowledge using its own judgment.

**Workflow skills** contain instructions. They lay out a specific sequence of steps Claude should follow to complete a task. They're like a recipe — Claude follows them in order, checking off each step.

The difference matters because it determines how you write the skill, how Claude uses it, and when it breaks down.

Here's the same concept — "how we build endpoints in Markd" — expressed both ways:

**As a reference skill:**

```markdown
---
name: markd-api-patterns
description: API conventions and patterns for the Markd bookmarks service.
  Reference when building or modifying API endpoints.
---
# Markd API Patterns

## Endpoint anatomy
- Route functions live in `src/markd/routes/` grouped by resource.
- Each endpoint has a Pydantic response model in `src/markd/models.py`.
- Request bodies use `Create` or `Update` prefix naming.
- Database operations go through `src/markd/db.py`, never inline SQL in routes.

## Error handling
- Use `HTTPException` for all error responses.
- Always include a human-readable `detail` string.
- Common: 404 for missing resources, 422 for validation (handled by FastAPI).

## Response shapes
- Single item: return the model directly.
- Lists: return `list[Model]`.
- No envelope wrapping. Let FastAPI handle serialization.
```

Claude reads this and *knows things*. It doesn't tell Claude what to do step by step — it trusts Claude to apply the knowledge. This is the right choice when the task varies enough that a rigid sequence would be limiting. You might use this when modifying existing endpoints, debugging, or refactoring.

**As a workflow skill:**

```markdown
---
name: add-endpoint
description: Step-by-step workflow for adding a new API endpoint to Markd.
  Use when creating a new route from scratch.
---
# Add Endpoint Workflow

When adding a new endpoint to Markd, follow these steps in order:

## 1. Database layer
- Add any new query functions to `src/markd/db.py`.
- If the schema changes, add a migration function and call it in `init_db()`.
- Follow existing patterns in db.py for parameter handling and return types.

## 2. Models
- Add Pydantic models to `src/markd/models.py`:
  - `Create*` model for POST request bodies (required fields only).
  - `Update*` model for PATCH request bodies (all fields Optional).
  - Response model matching the DB row plus any computed fields.

## 3. Route
- Add the route function in the appropriate file under `src/markd/routes/`.
- Use dependency injection for the database connection.
- Use the response model in the decorator: `@router.get("/path", response_model=Model)`.
- Handle errors with `HTTPException`.

## 4. Register
- If this is a new router file, register it in `src/markd/main.py`.

## 5. Tests
- Add tests in `tests/` covering:
  - Happy path (valid input → expected output).
  - Validation error (invalid input → 422).
  - Not found (missing resource → 404), if applicable.
- Use the `client` fixture from `conftest.py`.

## 6. Verify
- Run `pytest` and confirm all tests pass.
- Run `ruff check .` and fix any lint issues.
- Check `/docs` to verify the OpenAPI schema looks correct.
```

Claude reads this and *follows a process*. Each step is explicit and ordered. This is the right choice when you want consistency across repetitions of the same kind of task — you want every new endpoint to go through the same checklist, every time.

### When to use which

Here's the decision framework:

**Use a reference skill when:**
- The task is varied and Claude needs judgment, not a recipe.
- You're documenting domain knowledge (how your auth system works, what the database schema means, how your deployment pipeline is configured).
- The information helps Claude make better decisions across many different tasks.

**Use a workflow skill when:**
- The task is repeatable — you do it the same way each time.
- You've noticed Claude missing steps or doing them in the wrong order.
- You'd explain the same sequence of instructions to a junior developer.

Many projects end up with a handful of each. For Markd, we might have:

- `markd-api-patterns` — reference skill for general API knowledge
- `add-endpoint` — workflow skill for creating new endpoints
- `markd-testing` — reference skill for test conventions and fixtures

### Building your first skill

Let's create the `add-endpoint` workflow skill. You could write it by hand, but let's have Claude help — it already knows your codebase:

```
I want to create a skill for adding new API endpoints to this project.
Look at the existing endpoints, models, db functions, and tests.
Create a step-by-step workflow skill at .claude/skills/add-endpoint/SKILL.md
that captures the process. It should cover database changes, Pydantic
models, route creation, test writing, and verification steps.
```

Claude will generate a draft. Review it critically — skills are one of those things worth editing by hand. Ask yourself:

- **Is the sequence right?** Database first, then models, then routes, then tests — or does your project flow differently?
- **Is each step specific enough?** "Add tests" is too vague. "Add tests covering happy path, validation error, and not-found cases using the `client` fixture" is actionable.
- **Is it too long?** Anthropic recommends keeping SKILL.md under 500 lines. For a single workflow, aim for well under 100. If it's ballooning, you're probably mixing reference material and workflow instructions — split them.

### Auto-invocation vs. explicit triggers

Skills have a `description` field in their frontmatter. Claude reads all skill descriptions at the start of a session to know what's available. When you give Claude a task, it matches the task against those descriptions and loads relevant skills automatically.

This means the description is important. It determines *when* Claude reaches for the skill. Compare:

```yaml
# Vague — Claude might load this for anything API-related
description: Helps with API development.

# Specific — Claude loads this when creating new endpoints
description: Step-by-step workflow for adding a new API endpoint to Markd.
  Use when creating a new route from scratch.
```

The second description tells Claude both what the skill does and when to use it. That specificity means Claude won't waste context loading it when you're just fixing a bug or refactoring.

Sometimes you want a skill that Claude should *never* auto-invoke — only when you explicitly trigger it. Add this to the frontmatter:

```yaml
---
name: deploy
description: Production deployment checklist for Markd.
disable-model-invocation: true
---
```

With `disable-model-invocation: true`, this skill only runs when you type `/deploy`. Claude won't load it on its own. Use this for high-stakes workflows (deployment, database migrations, release processes) where you want the human to be in the driver's seat.

For the `add-endpoint` skill, auto-invocation makes sense — when you say "add an endpoint for bookmarks export," Claude should recognize that as a match and load the playbook.

### Test it: the "same task, better result" pattern

Let's replay the kind of task that was messy in earlier posts. Start a fresh session:

```bash
claude
```

Now ask for a new feature:

```
Add a GET /bookmarks/stats endpoint that returns:
- total bookmark count
- count by favorite status
- top 5 most-used tags
```

Watch how Claude works through it. With the skill loaded, it should follow the sequence: database function first, then models, then route, then tests, then verification. It shouldn't skip steps or do them in a random order.

Compare this with how Claude handled similar requests in Post 1 (no configuration at all) and Post 2 (CLAUDE.md but no skills). The progression should be clear:

- **Post 1:** Claude writes code that works but drifts in style.
- **Post 2:** Claude writes code that matches your conventions but invents its own process.
- **Post 3:** Claude writes code that matches your conventions *and* follows your process.

### Skills vs. CLAUDE.md: dividing responsibilities

Now that you have both tools, it's worth being precise about who owns what:

**CLAUDE.md owns facts.** Conventions, standards, project structure, development commands, pointers to reference docs. These are true regardless of the task. They're short, stable, and always loaded.

**Skills own processes.** Step-by-step workflows, task-specific playbooks, specialized knowledge that's only relevant for certain kinds of work. They load on demand. They can be long and detailed because they only consume context when needed.

If you find your CLAUDE.md growing past 100 lines, look at what you're adding. If it reads like a step-by-step guide for a specific task, extract it into a skill. If it reads like a convention or a fact about the project, keep it.

The reverse is also true: if a skill contains a lot of general conventions that apply to many tasks, that information probably belongs in CLAUDE.md where it's always visible, not locked inside a skill that only loads sometimes.

### A note on supporting files

Skills can include more than just SKILL.md. The skill directory can hold templates, scripts, or reference files. For example:

```
.claude/skills/add-endpoint/
├── SKILL.md              # The workflow instructions
└── endpoint-template.py  # A template file the skill references
```

Inside your SKILL.md, you can reference these files: "Use `endpoint-template.py` as the starting point for the new route function." Claude will read the template and use it.

This is useful for enforcing structure. Instead of describing the exact boilerplate in prose (which Claude might interpret loosely), you provide a literal template that Claude adapts. But keep it simple — one or two supporting files at most. If you're packaging a dozen files with a skill, you're building something more complex than a skill is meant to be.

### Skills that run in isolation

There's one more skill feature worth knowing about now, even though we'll explore it properly in Post 5 when we cover subagents. You can add `context: fork` to a skill's frontmatter, which makes it run in a separate context — its own isolated session with its own tools, disconnected from your current conversation.

```yaml
---
name: research-feature
description: Research how to implement a feature before writing code.
context: fork
agent: Explore
---
Research $ARGUMENTS in the codebase. Find related patterns, existing
implementations, and any edge cases to consider. Summarize findings.
```

When this skill runs, it spins up an Explore subagent that reads your codebase without polluting your main conversation's context. The results get summarized and returned to you.

This is powerful for tasks where you want Claude to do deep research, analysis, or planning *without* burning through context in your main session. But it only makes sense for skills that contain an actionable task — if your skill is just conventions with no instructions, a forked agent will load the conventions, have nothing to do, and return empty-handed.

We'll build on this in Post 5. For now, just know the option exists and that it bridges the gap between skills and subagents.

### The decision framework so far

| Situation | Tool | Why |
|---|---|---|
| Convention that always applies | CLAUDE.md | Always loaded, keeps code consistent |
| Convention for a specific area | Subdirectory CLAUDE.md | Only loads in that directory |
| Detailed reference material | Docs pointed from CLAUDE.md | Keeps the constitution lean |
| Domain knowledge for certain tasks | Reference skill | Loads when relevant, provides context |
| Repeatable multi-step workflow | Workflow skill | Consistent process every time |
| Something that must happen 100% of the time | ??? | Next post |

### Before next time

Make sure your Markd project has at least one skill — the `add-endpoint` workflow is the most useful starting point. If you want to go further, try creating:

- A `markd-api-patterns` reference skill with your API conventions (and notice how it overlaps with CLAUDE.md — then decide what to move where).
- A `write-tests` workflow skill that captures your testing checklist.

Then try this: **add a deliberately wrong pattern** somewhere. Put a raw dict response in one endpoint instead of a Pydantic model. Then ask Claude to add a new endpoint and see whether it copies the bad pattern or follows the skill. If your skill is well-written, Claude follows the skill. If it's vague, Claude might default to whatever it sees in the existing code.

That experiment reveals something important: skills work best when they're specific. Vague guidance loses to concrete examples in the codebase. Specific step-by-step instructions win.

One more thing to notice: Claude follows the skill's workflow *most of the time*. But "most of the time" isn't "every time." If you have a step like "run pytest after every change" and Claude occasionally skips it, that's not a skill problem — that's the boundary between probabilistic guidance and deterministic enforcement. The next post is about crossing that boundary.

### What's next

In the next post, we tackle hooks — the layer that gives you guarantees instead of guidance. We'll set up automatic test runs, linting, and other checks that fire deterministically, not just when Claude remembers to do them. It's a short post because the concept is simple, but it closes an important gap.

The mental model: CLAUDE.md and skills are *probabilistic* — Claude follows them using judgment. Hooks are *deterministic* — they run every time, no exceptions.

---

*This is Part 3 of the "Building With Claude Code" series. Previously: [Part 2 — CLAUDE.md: Your Project's Constitution](/tutorials/claude-code/). Next up: [Part 4 — Hooks: When You Need Guarantees, Not Suggestions](/tutorials/claude-code/)*
