---
title: "Jump In, Break Things, Learn Why Configuration Matters"
date: 2026-03-17 08:00:00 +0300
categories: tutorials
tags: claude-code cli
series: "Building With Claude Code"
series_order: 1
---

This is the first post in a series about using Claude Code for real software development. This is not a detailed feature guide but rather a decision guide. By the end of the series, you'll know *when* to reach for CLAUDE.md vs. a skill vs. a hook vs. a subagent, and more importantly, *why*.

We'll build a small app along the way: a bookmarks API with a CLI. But the app isn't the point. The **decisions** are. Each post is structured around a moment where you have to choose how to work with Claude, and we'll talk through the tradeoffs.

The whole series is designed as a project you can complete in a week or over the weekend. Today we'll set up and start building — no configuration, just prompting — and see exactly where things start to get interesting.

### What you'll need

- Claude Code installed ([install guide](https://code.claude.com/docs/en/overview))
- Python 3.11+
- A terminal you're comfortable in
- About 2 hours for this first session

If you already have Claude Code running, skip ahead to "Let's build something."

**Important Note** about this guide. Since agents by their nature are probabilistic, your responses and outputs will most likely be different from what is in this guide. This is ok. The idea is to reflect on the outputs and how they influence decisions about which Claude Code feature to use.

### The app: Markd

We're building **Markd**, a personal bookmarks manager. It's a REST API built with FastAPI and SQLite, with a small CLI client. Here's what it looks like when it's done:

```bash
$ markd add https://explained.re/elf --title "ELF Explained" --tags systems,linux
✓ Bookmark saved (id: 1)

$ markd add https://ciechanow.ski/gps --title "How GPS Works" --tags explainers,science
✓ Bookmark saved (id: 2)

$ markd list
┌────┬──────────────────────────┬──────────────────────┬───┐
│ ID │ Title                    │ Tags                 │ ★ │
├────┼──────────────────────────┼──────────────────────┼───┤
│  1 │ ELF Explained            │ systems, linux       │   │
│  2 │ How GPS Works            │ explainers, science  │   │
└────┴──────────────────────────┴──────────────────────┴───┘

$ markd fav 2
✓ Marked as favorite

$ markd search gps
┌────┬──────────────────────────┬──────────────────────┬───┐
│ ID │ Title                    │ Tags                 │ ★ │
├────┼──────────────────────────┼──────────────────────┼───┤
│  2 │ How GPS Works            │ explainers, science  │ ★ │
└────┴──────────────────────────┴──────────────────────┴───┘
```

Simple app. But the app isn't really the point — the **development workflow** is. Here's what building with Claude Code looks like at the end of this series, compared to where we'll start today:

**Today (no configuration):**
```
You:  Add a PATCH endpoint for updating bookmarks.
      Make sure it validates input and returns consistent
      error responses like the other endpoints. Use the
      same patterns as the existing routes. Run the tests
      after. Also the error format should be
      {"detail": "...", "status_code": ...} like we
      discussed earlier...

Claude: *writes endpoint with slightly different error
         handling than the others, forgets to run tests*
```

**By the end of this series:**
```
You:  Add a PATCH endpoint for updating bookmarks.

Claude: *reads CLAUDE.md for conventions, loads the
         "add-endpoint" skill for your project's patterns,
         writes endpoint matching existing style, hook
         auto-runs tests and linter before committing*
```

Same task. But in the second version, you said less and got more consistent results. The conventions, patterns, testing, and quality checks are all configured once and applied every time. That's the gap we'll close over the next few posts.

Why this app? It's simple enough to build in a weekend but has enough surface area — multiple endpoints, a database, Pydantic models, tests, a CLI — that every Claude Code concept will come up naturally. You won't have to squint to see why a skill or a hook matters.

---

## Let's build something

Create a new directory and open Claude Code in it:

```bash
mkdir markd && cd markd
claude
```

That's it. No configuration files. No CLAUDE.md. No skills. Just you and Claude in an empty folder. This is intentional — we want to see what happens when you start from zero, because that's how you'll understand what each layer of configuration actually solves.

### Round 1: Scaffold the project

Give Claude your first prompt:

```
Create a bookmarks manager app called Markd. It should be a REST API
using FastAPI with a SQLite database. I need CRUD endpoints for
bookmarks — each bookmark has a url, title, description, tags (list
of strings), and a favorite boolean. Set up the project with a
pyproject.toml and a clean package structure.
```

Watch what Claude does. It'll create files, set up a virtual environment, install dependencies, and give you a working starting point. For a one-shot scaffold like this, Claude Code is genuinely impressive. You'll probably get something runnable on the first try.

Take a moment to look at what it produced. Note a few things:

- **How did it structure the project?** A flat `main.py`? A package with `app/` and submodules? Did it separate routes, models, and database logic?
- **What patterns did it use?** Raw SQL or an ORM? Pydantic models for request/response, or plain dicts? Dependency injection for the database connection, or a global?
- **What did it assume?** The port number, the database filename, how tags are stored in SQLite (JSON column? comma-separated? a join table?), sync or async.

All of these are fine choices. The point isn't that Claude got something wrong — it's that it *made decisions you didn't discuss*. Remember that. It'll matter in about ten minutes.

Start the server and make sure it works:

```
Start the server and test that I can create and retrieve a bookmark.
Show me the curl commands.
```

You should have a running API. Celebrate briefly. Now let's stress-test the workflow.

### Round 2: Add a feature

With the server working, ask Claude to add search functionality:

```
Add a GET /bookmarks/search endpoint that accepts a query parameter q
and searches across title, description, and tags. It should be
case-insensitive.
```

Claude will implement this. It'll probably work. But look closely:

- **Did it follow the same patterns as the scaffolded code?** Maybe. Maybe not. If the original code used one style of exception handling and the search endpoint uses `HTTPException` differently, that's *drift*. It's subtle now, but it compounds.
- **Did it add tests?** Probably not, unless you asked.
- **Did it add a Pydantic response model?** Maybe for this endpoint, but did it match the shape of the others?

None of this is Claude's fault. You didn't tell it your conventions because you don't have any written down yet. Claude is working from its general training, not from your project's specific standards.

### Round 3: Where things get interesting

Now ask for something a bit more involved:

```
Add a CLI tool at cli/markd.py that uses click or typer to interact
with the API from the terminal. I want commands like:
  markd add <url> --title "..." --tags tag1,tag2
  markd list
  markd search <query>
  markd fav <id>
Make it pip-installable as a console script.
```

After Claude builds this, ask it to add one more endpoint to the API:

```
Add a PATCH /bookmarks/{id} endpoint for partially updating a bookmark.
```

Now look at the full codebase. Here's what you'll likely notice:

**1. Pattern inconsistency.** The endpoints built during scaffolding might use slightly different error handling, response shapes, or validation approaches than the ones added later. Maybe the early endpoints return `{"id": 1, "url": "..."}` and the later ones return `{"data": {"id": 1, "url": "..."}, "status": "ok"}`. When Claude writes multiple things across multiple prompts, each prompt starts with good intentions but no memory of the micro-decisions made in earlier prompts.

**2. Missing cross-cutting concerns.** Did every endpoint get proper Pydantic validation? Consistent error responses? Logging? Did it use the same response model pattern everywhere? Claude addresses what you ask about and makes reasonable guesses about the rest, but "reasonable guesses" vary from prompt to prompt.

**3. Context fade.** If your session is getting long, Claude's awareness of the full codebase starts to thin. It might re-read files before editing them (good), but it won't necessarily remember that you care about a specific error format or that you wanted all responses wrapped in a standard envelope.

Try this experiment — ask Claude to add *another* endpoint:

```
Add a GET /bookmarks/tags endpoint that returns all unique tags with
a count of how many bookmarks use each tag.
```

Compare the style of this endpoint with the first ones Claude wrote. Are the response models consistent? Is error handling identical? Did it use the same database access pattern? Chances are there's drift.

### The insight

Here's the thing: everything Claude built *works*. If you're prototyping, this is fantastic. You went from an empty folder to a working API with a CLI in maybe 30 minutes.

But if you're building something you'll maintain — something where a teammate might open a PR next week, or where you'll come back in a month — the inconsistencies start to matter. And the solution isn't to write longer prompts. It's to stop relying on prompts for things that should be *configured*.

Think about it this way:

| What you're doing now | What you'll learn to do |
|---|---|
| Explaining conventions in every prompt | Writing them once in CLAUDE.md |
| Hoping Claude follows the same patterns | Teaching patterns via skills |
| Asking Claude to run tests | Making tests run automatically via hooks |
| Watching context get cluttered | Offloading research to subagents |

This is the roadmap for the rest of the series. Each post introduces one of these tools at the moment you *need* it — not as a feature demo, but as a solution to a real problem you've just felt.

### Before next time

Keep your `markd` project as-is. The inconsistencies are useful — we'll fix them in the next post when we introduce CLAUDE.md, and the contrast will make the value click immediately.

If you want to poke around further, try a few things:

- **Ask Claude to add tests** for all your endpoints. See if it picks pytest with httpx's `AsyncClient`, or something else entirely. See if it creates fixtures or repeats setup in every test function.
- **Start a new Claude Code session** in the same project (just exit and run `claude` again). Ask it to add a feature. Notice how it has to re-learn the project from scratch — reading files, inferring patterns. That re-learning time is real cost.
- **Read through your code as if you were a new team member.** What would you need to know that isn't written down anywhere? That list is roughly what your CLAUDE.md should contain.
- **Check the `/docs` endpoint** that FastAPI gives you for free. Compare what the auto-generated OpenAPI docs show against what you'd actually want. Inconsistent response models become very visible here.

### What's next

In the next post, we'll add a CLAUDE.md file to the project and immediately see the difference. We'll talk about what belongs in it (and, critically, what doesn't), how to think about root vs. subdirectory files, and when to point to a separate reference doc instead of inlining everything.

The goal: after one file, Claude should produce code that looks like *your* code, not generic code that happens to work.

---

*This is Part 1 of the "Building With Claude Code" series. Next up: [Part 2 — CLAUDE.md: Your Project's Constitution](/tutorials/claude-code/)*
