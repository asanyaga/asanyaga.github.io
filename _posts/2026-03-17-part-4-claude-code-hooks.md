---
title: "Hooks: Automated Enforcement for Your Workflow"
date: 2026-03-17 11:00:00 +0300
categories: tutorials
tags: claude-code cli
series: "Building With Claude Code"
series_order: 4
---

At the end of Post 3, we left you with an observation: skills work *most of the time*. Your `add-endpoint` skill tells Claude to run pytest after writing code, and usually Claude does. But "usually" isn't "always." Sometimes Claude gets confident, skips the verification step, and moves on. Sometimes a long session means the skill's instructions fade slightly and Claude improvises.

For many things, that's fine. Claude writing a response model with a slightly different field order than your template isn't going to break production. But some things need to happen every time, without exception. Tests should run before every commit. The linter should catch issues before they make it into a PR. Dangerous commands should be blocked, period.

This is the gap between *guidance* and *enforcement*. CLAUDE.md and skills are guidance — Claude follows them using judgment, and judgment varies. Hooks are enforcement — they're code that runs at specific moments in Claude Code's lifecycle, and code doesn't have off days.

### The mental model

A hook is a script that fires automatically when a specific event happens inside Claude Code. You don't ask Claude to run it. You don't hope Claude remembers it. It just runs.

The events you can hook into include:

- **PreToolUse** — before Claude runs a tool (like writing a file or executing a bash command)
- **PostToolUse** — after a tool finishes
- **Notification** — when Claude wants your attention
- **Stop** — when Claude finishes a response

That's the full lifecycle. Most practical hooks use PreToolUse (to block or modify actions before they happen) and PostToolUse (to react to what just happened).

Hooks are configured in your Claude Code settings, not in CLAUDE.md or skills. You set them up via the `/hooks` command inside Claude Code, or by editing the settings JSON directly.

### Your first hook: auto-lint after file changes

Let's start with something immediately useful. Every time Claude writes or edits a Python file, we want ruff to run automatically. No skipping, no forgetting.

In your Claude Code session, set up the hook:

```
Set up a hook that runs ruff check --fix on any Python file that
Claude creates or modifies. It should run after the file write
completes. Use the PostToolUse event on the Write tool.
```

Claude will help you configure this. Under the hood, what's being created is a hook definition that says: "After Claude uses the Write or Edit tool on a `.py` file, run `ruff check --fix` on that file."

The result is a settings entry that looks something like:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "command": "ruff check --fix $CLAUDE_FILE_PATH"
      }
    ]
  }
}
```

Now every time Claude writes Python code, ruff auto-fixes formatting issues immediately. Claude doesn't decide whether to lint. It just happens.

### The critical distinction: probabilistic vs. deterministic

This is worth stating plainly because it drives every decision about whether to put something in CLAUDE.md, a skill, or a hook:

| Layer | Type | Reliability | Best for |
|---|---|---|---|
| CLAUDE.md | Probabilistic | ~80-90% | Conventions, preferences, context |
| Skills | Probabilistic | ~85-95% | Workflows, patterns, domain knowledge |
| Hooks | Deterministic | 100% | Safety rules, quality gates, automation |

The percentages are rough — they depend on prompt complexity, context length, and how specific your instructions are. The point is the category difference. CLAUDE.md and skills are suggestions that Claude's judgment interprets. Hooks are code that a computer executes. There's no interpretation.

This means the decision is straightforward: **if "Claude followed this 90% of the time" is acceptable, use CLAUDE.md or a skill. If it needs to be 100%, use a hook.**

### Practical hooks for Markd

Let's set up the hooks that matter for our project. You can configure these interactively:

```
I want to set up the following hooks for this project:

1. After any Python file is written or edited, run ruff check --fix
   on the changed file.

2. Before any git commit, run pytest. If tests fail, block the commit.

3. Send me a desktop notification when Claude finishes a task or
   needs my input.
```

Let's walk through what each of these does and why it's a hook rather than a CLAUDE.md rule or a skill instruction.

**Auto-lint (PostToolUse on Write/Edit).** We had "run ruff before committing" in our CLAUDE.md. That worked most of the time. But when Claude edits three files in quick succession, it sometimes lints the last one and forgets the first two. A hook runs on every single file write, individually. No files slip through.

**Pre-commit test gate (PreToolUse on Bash).** This watches for `git commit` commands and runs pytest first. If any test fails, the hook blocks the commit — Claude literally cannot commit broken code. This is the kind of guardrail you'd set up in CI, but having it locally means Claude catches failures before they even reach a PR.

**Notification (Notification event).** This one isn't about code quality — it's about workflow. When Claude finishes a long task or hits a permission prompt, you get a desktop notification instead of having to watch the terminal. If you're running Claude in a background tab while reviewing code, this keeps you in the loop without context-switching.

### What hooks should NOT do

Hooks are powerful precisely because they're rigid. That rigidity is also their limitation. Here's what doesn't belong in a hook:

**Complex decision-making.** A hook that says "if the file is in the routes/ directory and it's a new file but not a test file, then run the integration tests instead of unit tests" is fragile. Hooks should be simple: match a pattern, run a command, check the exit code. If the logic requires judgment, it belongs in a skill.

**Code generation or modification.** Hooks run *scripts*, not prompts. A hook can run ruff to auto-fix formatting. It shouldn't try to rewrite code based on conventions — that's what CLAUDE.md and skills are for.

**Anything slow.** Hooks run synchronously. A hook that takes 30 seconds to complete blocks Claude for that entire time. Keep them fast — lint a single file, run a quick test suite, send a notification. If you need a long-running check, consider whether it belongs in a CI pipeline instead.

### Safety hooks: blocking dangerous commands

One of the most valuable uses of hooks is preventing Claude from running commands you never want it to run. The canonical example:

```
Set up a PreToolUse hook on the Bash tool that blocks any command
containing "rm -rf", "DROP TABLE", or "git push --force". The hook
should exit with an error message explaining why the command was blocked.
```

This creates a hook that inspects every bash command before Claude executes it. If the command matches a dangerous pattern, the hook returns a non-zero exit code and Claude sees the error message instead of running the command.

The reason this is a hook and not a CLAUDE.md rule: a community member tested both approaches. With "never run rm -rf" in CLAUDE.md, Claude followed the rule about 70% of the time. With a hook, it's blocked 100% of the time. For safety-critical rules, that difference is everything.

### Hooks and the rest of the stack

Here's how hooks fit with everything we've built so far:

**CLAUDE.md says:** "Always run tests before committing."
**The skill says:** "Step 6: Run pytest and confirm all tests pass."
**The hook enforces:** Tests actually run before every commit, even if Claude skips step 6.

These aren't redundant — they're defense in depth. The CLAUDE.md convention means Claude *intends* to test. The skill step means Claude has a specific place in the workflow to test. The hook means testing happens regardless. Each layer catches what the one above might miss.

In practice, once you have hooks handling the mechanical stuff (linting, testing, blocking dangerous commands), you can simplify your skills. The `add-endpoint` skill from Post 3 had a "Verify" step that said "run pytest and ruff." With hooks in place, you can remove that step — the hooks handle verification automatically. The skill can focus on the creative, judgment-based steps that actually need Claude's intelligence.

### The decision framework so far

| Situation | Tool | Why |
|---|---|---|
| Convention that always applies | CLAUDE.md | Always loaded, keeps code consistent |
| Convention for a specific area | Subdirectory CLAUDE.md | Only loads in that directory |
| Detailed reference material | Docs pointed from CLAUDE.md | Keeps the constitution lean |
| Domain knowledge for certain tasks | Reference skill | Loads when relevant, provides context |
| Repeatable multi-step workflow | Workflow skill | Consistent process every time |
| Must happen 100% of the time | Hook | Deterministic, code-based enforcement |
| Context is getting cluttered | ??? | Next post |

### Before next time

Set up at least two hooks in your Markd project:

1. Auto-lint on file write (PostToolUse)
2. Pre-commit test gate (PreToolUse)

The notification hook is optional but worth trying if you ever run Claude on longer tasks.

Then try this experiment: **ask Claude to add a feature and deliberately don't mention testing or linting.** Something like:

```
Add a DELETE /bookmarks/{id} endpoint. Hard delete, not soft delete.
```

Don't say "and write tests." Don't say "and lint." Just the bare feature request. Watch what happens. Claude will write the endpoint — and the hooks will auto-lint the file and block the commit if there are no tests. The guardrails work even when nobody asks for them.

Once you have this working, pay attention to how your prompts change. You stop adding "remember to test" and "make sure to lint" to every request. Your prompts get shorter and more focused on *what* you want, because the *how* is handled by configuration. That's the compounding payoff of the whole stack we've built: CLAUDE.md handles conventions, skills handle processes, hooks handle enforcement. You just describe the feature.

### What's next

We've now covered the three layers that shape how Claude writes code in your project. But we haven't addressed a problem that grows with session length: **context clutter**. As you work with Claude on multiple tasks in a single session, the conversation accumulates context from each task. By the time you're on your fifth feature, Claude is carrying the memory of four previous features it doesn't need anymore.

In the next post, we'll cover subagents — Claude Code's mechanism for running isolated tasks in their own context. We'll look at when to let Claude spawn its own subagents, when to define custom ones, and the tradeoff between isolation and shared context.

---

*This is Part 4 of the "Building With Claude Code" series. Previously: [Part 3 — Skills: Teaching Claude How Your Project Works](/tutorials/claude-code/). Next up: [Part 5 — Subagents: Managing Context, Not Just Tasks](/tutorials/claude-code/)*
