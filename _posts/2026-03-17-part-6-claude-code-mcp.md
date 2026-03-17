---
title: "MCP & The Bigger Picture: Claude in Your Toolchain"
date: 2026-03-17 13:00:00 +0300
categories: tutorials
tags: claude-code cli
series: "Building With Claude Code"
series_order: 6
---

Over the past five posts, we've built a layered system for working with Claude Code. CLAUDE.md sets conventions. Skills teach processes. Hooks enforce rules. Subagents manage context. Together, they turn Claude from a capable assistant into one that knows *your* project.

But so far, Claude has only interacted with two things: your codebase and your terminal. Real development doesn't happen in a vacuum. You read issues in GitHub, reference documentation on the web, check CI pipelines, update project boards. Every time you manually copy-paste information from those tools into a Claude prompt, you're doing the kind of bridging work that should be automated.

That's what MCP — the Model Context Protocol — solves. And once Claude is connected to your external tools, everything we've built across this series comes together into a workflow that goes from issue to merged PR with minimal manual bridging.

### What MCP actually is

MCP is a protocol that lets AI tools connect to external services through a standardized interface. An MCP server exposes a set of capabilities — reading GitHub issues, searching documentation, fetching Slack messages — and Claude Code can use those capabilities as naturally as it uses bash commands or file reads.

The mental model: **MCP makes external tools feel like local tools.** Instead of "go to GitHub, find issue #42, copy the description, paste it into your prompt," you say "look at GitHub issue #42" and Claude reads it directly.

Claude Code supports MCP servers out of the box. You configure them in your settings, and their capabilities become available in every session.

### Setting up your first MCP server: GitHub

Let's connect Markd to GitHub. If your project isn't already a GitHub repo, set one up now — we'll need it for the capstone exercise at the end of this post.

```
Help me set up the GitHub MCP server for this project. I want Claude
to be able to read issues, create branches, and open pull requests.
```

Claude will walk you through the configuration. The setup involves adding the GitHub MCP server to your Claude Code settings, which looks something like:

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "<your-token>"
      }
    }
  }
}
```

Once configured, Claude can interact with GitHub natively. You can say things like:

```
What are the open issues on this repo?
```

```
Read issue #3 and summarize what's being requested.
```

```
Create a branch called feature/bookmark-export and open a draft PR
with what I've built so far.
```

No copy-pasting. No switching windows. Claude reads from and writes to GitHub as part of its normal workflow.

### When MCP vs. when bash

MCP isn't the only way to connect Claude to external tools. Claude can already run bash commands, which means it can use the `gh` CLI, `curl` APIs, or run any script you give it. So when should you use MCP versus just letting Claude run a command?

**Use MCP when:**
- The integration is ongoing and conversational. You want Claude to fluidly read issues, check PR status, and reference external data throughout a session — not as a one-off command but as a natural part of how it works.
- The tool has a well-maintained MCP server. GitHub, Slack, Linear, Sentry, and many others have community or official servers.
- You want the tool's data to be available without you thinking about it. MCP capabilities show up as tools Claude can use proactively, not just reactively.

**Use bash/scripts when:**
- The integration is simple and one-directional. If you just need to post a Slack message or fetch a URL, `curl` in a hook or skill is simpler than setting up an MCP server.
- No MCP server exists for the tool. Not everything has one. Writing a quick bash wrapper is faster than building a server.
- The action is part of a hook. Hooks run scripts, not MCP calls. If you want to post to Slack every time Claude finishes a task, that's a hook running a `curl` command.

For Markd, GitHub MCP is worth the setup because we'll use it throughout the development workflow — reading issues, creating branches, opening PRs. If we just needed to post a deployment notification to Slack, a bash command in a hook would be simpler.

### Beyond GitHub: other MCP servers worth knowing about

MCP is an open standard, and the ecosystem is growing. A few servers that are particularly useful for development workflows:

- **Web search / documentation** — let Claude search the web or read docs pages directly, useful when implementing features that require external research.
- **Sentry / error tracking** — Claude can read error reports and stack traces, then trace them to the relevant code.
- **Linear / Jira / Asana** — project management integration, so Claude can read tasks and update status.
- **Postgres / database** — direct database access for debugging or data exploration.

You don't need all of these. Start with the one that removes the most manual copy-pasting from your workflow. For most developers, that's GitHub.

### CI/CD: your configuration travels with you

Here's something that becomes clear once you have the full stack set up: everything you've built — CLAUDE.md, skills, hooks — lives in your repository. That means it works anywhere your repo goes, including CI/CD.

Claude Code can run in GitHub Actions and GitLab CI as a headless agent. When it does, it reads the same CLAUDE.md, loads the same skills, and follows the same conventions as your local sessions. This is powerful because it means the agent that reviews PRs in CI uses the same standards as the agent that wrote the code locally.

A common setup:

- **PR review**: Claude Code runs on every PR, reads the diff, checks it against CLAUDE.md conventions, and leaves review comments. It knows your standards because it reads the same CLAUDE.md you've been building all series.
- **Issue triage**: Claude reads new issues via the GitHub MCP integration, labels them, and optionally drafts an implementation plan using your skills.
- **Automated fixes**: For certain kinds of issues (lint errors, dependency updates, simple bug fixes), Claude can create a branch, make the fix, run tests, and open a PR — all headless.

The details of CI/CD setup change frequently, so I won't walk through the YAML here. The conceptual point is what matters: your investment in CLAUDE.md, skills, and hooks isn't just for your local terminal. It's a configuration layer that applies everywhere Claude runs against your codebase. The time you spent writing that CLAUDE.md pays off every time a CI agent reviews a PR, not just when you're coding locally.

### Putting it all together: from issue to PR

Let's close the series by doing what we promised in Post 1 — building a feature using the complete stack. We'll take a feature from GitHub issue to merged PR, and you'll see every layer we've built in action.

**Step 0: Create the issue.**

Go to your Markd repo on GitHub and create an issue:

> **Title:** Add bookmark deduplication
>
> **Body:** When adding a bookmark, if a bookmark with the same URL already exists, return the existing bookmark instead of creating a duplicate. Add a `GET /bookmarks/duplicates` endpoint that finds bookmarks with the same URL. Include tests.

Or, if you have GitHub MCP set up, just ask Claude:

```
Create a GitHub issue titled "Add bookmark deduplication" with this
description: When adding a bookmark, if a bookmark with the same URL
already exists, return the existing bookmark instead of creating a
duplicate. Add a GET /bookmarks/duplicates endpoint that finds
bookmarks with the same URL. Include tests.
```

**Step 1: Read the issue and plan.**

Start a fresh Claude Code session:

```
Read GitHub issue #<number> and plan the implementation. Research our
codebase first to understand how bookmark creation currently works.
```

Watch what happens. Claude reads the issue via MCP — no copy-paste needed. It spawns an Explore subagent to research the codebase (how `POST /bookmarks` works, how the database layer is structured, what the current uniqueness constraints are). The subagent returns a summary. Claude proposes a plan.

**Layers at work:** MCP (reading the issue), subagent (isolated codebase research), CLAUDE.md (Claude knows the project structure and conventions from the start).

**Step 2: Implement.**

```
Go ahead and implement the plan. Create a feature branch first.
```

Claude creates a branch (via MCP or git commands), then starts implementing. It loads the `add-endpoint` skill for the new `/bookmarks/duplicates` endpoint and follows the step-by-step workflow: database function, Pydantic model, route, tests. For the deduplication logic on `POST /bookmarks`, it modifies the existing route, following the error handling conventions from CLAUDE.md.

**Layers at work:** Skill (the add-endpoint workflow), CLAUDE.md (conventions for error handling, response models, database patterns), MCP (branch creation).

**Step 3: Automatic verification.**

As Claude writes code, hooks fire silently in the background. Every Python file gets auto-linted by ruff. When Claude tries to commit, the pre-commit hook runs pytest. If any test fails, the commit is blocked and Claude fixes the issue before trying again.

**Layers at work:** Hooks (auto-lint, pre-commit test gate).

**Step 4: Open the PR.**

```
Open a pull request for this feature. Reference the issue in the
description.
```

Claude opens a PR via MCP, linking it to the original issue. The PR description includes what was changed and why, formatted according to whatever conventions you've set.

**Layers at work:** MCP (creating the PR, linking the issue).

**The full picture.** You typed four prompts. Claude read the issue, researched the codebase, followed your endpoint workflow, maintained your conventions, auto-linted, auto-tested, and opened a PR — all using the configuration you've built across this series. No prompt mentioned testing, linting, response models, error handling patterns, or branch naming. All of that was handled by the stack.

This is the difference between Post 1 and now. In Post 1, you had to explain everything in every prompt and results still drifted. Now, your prompts describe *what* you want, and the configuration handles *how* it gets done.

### The complete decision framework

Here's the final version of the table we've been building all series. Print it, bookmark it, tape it to your monitor — it's the cheat sheet for every decision you'll make when setting up Claude Code for a project.

| Situation | Tool | Why |
|---|---|---|
| Convention that always applies | CLAUDE.md (root) | Always loaded, short, stable |
| Convention for a specific area | CLAUDE.md (subdirectory) | Loads only in that directory |
| Detailed reference material | Docs pointed from CLAUDE.md | Keeps the constitution lean |
| Domain knowledge for certain tasks | Reference skill | Loads on demand, provides context |
| Repeatable multi-step workflow | Workflow skill | Consistent process every time |
| Must happen 100% of the time | Hook | Deterministic enforcement |
| Heavy research or isolated task | Subagent | Clean context, focused work |
| Integrating external tools | MCP server | Native access to GitHub, etc. |
| Simple one-off external action | Bash in a hook or skill | Faster than an MCP server |

And the quick decision tree:

- **"Should Claude always know this?"** → CLAUDE.md
- **"Should Claude know this for certain tasks?"** → Skill (reference)
- **"Should Claude follow these steps for a specific kind of task?"** → Skill (workflow)
- **"Must this happen without exception?"** → Hook
- **"Is this polluting my context?"** → Subagent
- **"Does Claude need ongoing access to an external tool?"** → MCP server

### Where to go from here

You now have the mental model for every major Claude Code concept. The specifics — exact configuration syntax, new features, MCP server options — will evolve. The thinking won't. When you encounter a new project or a new team, the process is the same:

1. **Start with CLAUDE.md.** Write down what's always true about the project. Keep it short.
2. **Notice what you repeat.** When you explain the same process twice, extract it into a skill.
3. **Notice what gets skipped.** When Claude forgets a step that matters, make it a hook.
4. **Notice what clutters context.** When research or investigation bloats your session, use subagents.
5. **Notice what you copy-paste.** When you're bridging between Claude and an external tool, set up MCP.

Each layer solves a specific problem. Add them as you feel the problem, not before. The worst Claude Code setups are the ones built all at once by someone who read every feature doc and configured everything preemptively. The best ones are grown organically — one CLAUDE.md rule at a time, one skill at a time, one hook at a time — in response to real friction.

Your Markd project is small. The patterns you've learned scale to any codebase. A team monorepo uses the same layers — just more of them, with more subdirectory CLAUDE.md files, more specialized skills, and probably a CI/CD integration running the same configuration headless.

The tools will change. The thinking stays.

---

*This is Part 6, the final post in the "Building With Claude Code" series. The whole series: [Part 1: Jump In, Break Things](/tutorials/claude-code/) · [Part 2: CLAUDE.md](/tutorials/claude-code/) · [Part 3: Skills](/tutorials/claude-code/) · [Part 4: Hooks](/tutorials/claude-code/) · [Part 5: Subagents](/tutorials/claude-code/) · Part 6: MCP & The Bigger Picture*
