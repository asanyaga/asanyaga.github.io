---
title: "Subagents: Keeping Your Context Clean"
date: 2026-03-17 12:00:00 +0300
categories: tutorials
tags: claude-code cli
series: "Building With Claude Code"
series_order: 5
---

If you've been following along, your Markd project has grown. You've added a handful of endpoints, a CLI, tests, an archive feature, stats, and a delete endpoint. Over the course of building all that, you've probably noticed something about longer sessions: Claude gets slower and less precise.

Not because Claude is getting tired — because the conversation is getting heavy. Every prompt, every file read, every tool output accumulates in the context window. By the time you're on your third feature in a single session, Claude is carrying the full memory of two features it's already finished. That context isn't helping anymore. It's noise.

This is the problem subagents solve. Not parallelism (though they can do that). Not specialization (though that too). The core value is **isolation** — running a task in its own clean context so your main conversation stays focused.

### The context cost of a single conversation

Let's make this concrete. Imagine you're in a Claude Code session working on Markd. You ask Claude to:

1. Research how to implement full-text search in SQLite
2. Implement FTS on the bookmarks table
3. Update the search endpoint to use FTS
4. Write tests for the new search behavior

By the time you reach step 4, Claude's context contains: all the research output from step 1 (web searches, documentation excerpts, comparison of approaches), the implementation discussion from step 2, the refactoring from step 3, plus every file read and tool output along the way. Most of that context is dead weight — you don't need the FTS research while writing tests. But it's sitting there, taking up space and subtly diluting Claude's attention.

Now imagine you'd split it differently. An isolated agent handles step 1, researches FTS approaches, and returns a short summary: "Use SQLite FTS5 with a content table mirroring the bookmarks table. Here's the schema and the query pattern." Your main conversation receives that summary — maybe 200 tokens instead of 5,000 — and proceeds with steps 2-4 with a clean, focused context.

That's what subagents do.

### Built-in agents: Explore and Plan

Claude Code comes with built-in subagent types that cover the most common isolation needs. The two you'll use most:

**Explore** is a read-only agent. It can search your codebase, read files, and grep through code, but it can't modify anything. This makes it ideal for research tasks — understanding how something works, finding related patterns, checking what exists before you build something new.

**Plan** is similar but oriented toward producing a structured plan. Give it a task and it'll investigate, then return a proposed approach rather than raw findings.

You don't need to configure these. Claude Code can spawn them on its own using its Task tool. You can also invoke them explicitly.

### Letting Claude decide vs. directing it yourself

There are two ways subagents get used, and the choice between them matters.

**Claude spawns subagents automatically.** When Claude encounters a task that benefits from isolation — heavy research, exploring a large codebase, investigating an unfamiliar area — it can spawn Explore or Plan subagents on its own using the Task tool. You'll see a message like "running in background" in your terminal. The subagent works independently, returns results, and your main conversation continues.

This works well when:
- You give Claude a broad task and trust its judgment on how to break it down.
- The task naturally decomposes into research + implementation.
- You don't have strong opinions about what should be isolated.

Try it with Markd:

```
I want to add bookmark import/export — users should be able to export
their bookmarks as a standard Netscape bookmark HTML file and import
from that same format. Research the format first, then implement it.
```

Watch what Claude does. It'll likely spawn an Explore agent to research the Netscape bookmark format, get back a summary, then implement the feature in the main context. You didn't tell it to use a subagent — it decided isolation was the right call.

**You direct subagents explicitly.** Sometimes you know upfront that a task should be isolated. Maybe you've learned from experience that a particular kind of research bloats context, or you want parallel investigations.

```
Use an Explore agent to look through our codebase and find every place
where we directly construct SQL queries. I want to know if we have
any SQL injection risks. Don't fix anything yet — just report back.
```

By asking for an Explore agent explicitly, you're telling Claude to isolate this investigation. The security audit happens in a clean context, and your main conversation only receives the findings.

### When to use subagents (and when not to)

The decision comes down to one question: **is this task polluting my main conversation?**

**Use subagents when:**
- The task involves heavy reading or research that produces a lot of intermediate output (searching docs, exploring unfamiliar code, comparing approaches).
- You want to investigate something without committing to acting on it yet.
- You're doing multiple independent things and they don't need to share context.
- Your session is already long and you want to keep the main context clean for the next task.

**Don't use subagents when:**
- The task is simple enough that it won't meaningfully bloat context.
- The task requires back-and-forth conversation — subagents run autonomously and return results, they don't have a dialogue with you.
- The task needs to modify files based on recent conversation context — a subagent doesn't see your conversation history (unless you explicitly pass information to it).
- You're making changes where Claude needs to understand the full picture of what's been done so far in the session.

That last point is the key tradeoff: **isolation means independence, and independence means the subagent doesn't know what you've been discussing.** If you've spent 10 minutes explaining a design decision in your main conversation and then spawn a subagent to implement it, the subagent doesn't have that context. You'd need to include the relevant information in the task description.

### Custom subagents

Beyond the built-in Explore and Plan types, you can define your own subagents. These live in `.claude/agents/` as markdown files that describe the agent's purpose, constraints, and what tools it can use.

For Markd, a useful custom subagent might be a test writer:

```
Create a custom subagent at .claude/agents/test-writer.md that
specializes in writing tests for our project. It should:
- Have access to Read, Write, and Bash tools
- Know our testing conventions (pytest + httpx AsyncClient)
- Always read the endpoint implementation before writing tests
- Run pytest after writing tests to verify they pass
- Preload the markd-api-patterns skill for context
```

This creates an agent definition like:

```markdown
---
name: test-writer
description: Writes and verifies tests for Markd API endpoints.
tools:
  - Read
  - Write
  - Bash
skills:
  - markd-api-patterns
---
You are a test-writing specialist for the Markd bookmarks API.

When asked to write tests for an endpoint:
1. Read the endpoint implementation to understand the behavior.
2. Read conftest.py to understand available fixtures.
3. Write tests covering happy path, validation errors, and not-found cases.
4. Run pytest to verify all tests pass.
5. Fix any failures before returning results.

Use pytest with httpx AsyncClient. Follow the patterns in existing test files.
```

The `skills` field in the frontmatter means this agent starts with the `markd-api-patterns` skill already loaded — it has your API conventions from the moment it spins up.

Now when you ask Claude to write tests, you can direct it to use this agent:

```
Use the test-writer agent to write tests for the export endpoint.
```

The tests get written in an isolated context. The agent reads the implementation, writes tests, runs them, fixes failures, and returns the result. Your main conversation stays clean.

### The honest tradeoff: context isolation vs. context awareness

Custom subagents are Claude Code's most powerful feature on paper. In practice, they introduce a tension you should understand before over-investing.

The pitch: a complex task requires a lot of input context (conventions, existing code) and produces a lot of working context (research, intermediate edits). Running it in the main conversation means all of that accumulates. A subagent handles it in isolation and returns only the final result.

The cost: **you've now hidden context from your main agent.** If you define a `test-writer` subagent, your main Claude no longer sees how tests are written. It can't reason holistically about a change that affects both implementation and tests. If the test-writer subagent makes a decision (say, adding a new fixture), the main agent doesn't know about it.

For well-defined, repeatable tasks with clear boundaries — like running a security audit or writing tests for a single endpoint — this isolation is worth it. For tasks that are deeply intertwined with the rest of your work, the overhead of keeping agents in sync outweighs the context savings.

Some experienced Claude Code users have landed on what they call the "master-clone" approach: instead of defining specialized subagents, they put all the context in CLAUDE.md and skills, then let the main agent spawn general-purpose copies of itself via the Task tool when it needs isolation. The clones inherit all the project context through CLAUDE.md and skills, do their work, and return results. No custom agent definitions to maintain, no context getting locked away in a specialist.

There's no universally right answer here. The guideline: **start simple.** Let Claude spawn its own subagents via Task. If you find a specific pattern that's repeatable and well-bounded (like test writing), promote it to a custom subagent. Don't pre-build an army of specialists before you know what you need.

### Connecting back: skills with `context: fork`

In Post 3, we previewed skills that run in isolation via `context: fork`. Now you can see how that fits in. A skill with `context: fork` is just a convenient way to package a subagent invocation as a skill — the skill content becomes the task, and the `agent` field picks which subagent type runs it.

```yaml
---
name: research-feature
description: Research a feature before implementing it.
context: fork
agent: Explore
---
Research $ARGUMENTS in the codebase and on the web. Find related
patterns, existing implementations, edge cases, and potential pitfalls.
Return a structured summary with recommended approach.
```

When you say "I want to add bookmark deduplication" and this skill's description matches, Claude forks an Explore agent that does the research in isolation and returns a clean summary. It's a skill on the outside (auto-invocable, description-matched) and a subagent on the inside (isolated context, separate tools).

Use this pattern when you want the *convenience* of skills (auto-invocation, discoverability) with the *isolation* of subagents (clean context, no pollution). The research skill above is the textbook example — you always want research isolated, and you always want it to happen automatically when someone says "research" or "investigate."

### Parallel subagents

One more pattern worth knowing: Claude can spawn multiple subagents in parallel. If you have independent tasks that don't depend on each other, this can significantly speed up your workflow.

```
I want to add three things to Markd. These are independent — do them
in parallel:

1. Add a GET /bookmarks/random endpoint that returns a random bookmark.
2. Add a --format csv option to the CLI's list command.
3. Write a docs/api-reference.md from the current OpenAPI spec.
```

Claude can spawn three subagents, each handling one task, running simultaneously. Each works in its own context with its own tools. Results come back as they finish.

This works well when the tasks are truly independent. If task 3 needs to reference endpoints created in task 1, running them in parallel creates a race condition. Use judgment — or just let Claude decide. It's generally good at recognizing when tasks can be parallelized.

### The decision framework so far

| Situation | Tool | Why |
|---|---|---|
| Convention that always applies | CLAUDE.md | Always loaded, keeps code consistent |
| Convention for a specific area | Subdirectory CLAUDE.md | Only loads in that directory |
| Detailed reference material | Docs pointed from CLAUDE.md | Keeps the constitution lean |
| Domain knowledge for certain tasks | Reference skill | Loads when relevant, provides context |
| Repeatable multi-step workflow | Workflow skill | Consistent process every time |
| Must happen 100% of the time | Hook | Deterministic, code-based enforcement |
| Heavy research or isolated task | Subagent | Clean context, focused work |
| Connecting to external tools | ??? | Next post |

### Before next time

Experiment with subagents in your Markd project. Try two things:

**1. Let Claude decide.** Give it a complex task and see whether it spawns subagents on its own:

```
I want to add full-text search to Markd using SQLite FTS5. Research
the approach, implement it, update the search endpoint, and write tests.
```

Watch whether Claude isolates the research phase automatically.

**2. Direct a subagent yourself.** Ask Claude to run an isolated investigation:

```
Use an Explore agent to audit our codebase for any inconsistencies
with the conventions in CLAUDE.md. Check error handling, response
models, and database access patterns. Report what you find but don't
fix anything.
```

This is a genuinely useful exercise — it's like a code review powered by your own conventions. The Explore agent reads your CLAUDE.md, reads your code, and reports mismatches. And because it's isolated, the audit output doesn't clutter your next task.

### What's next

We've now covered every layer of how Claude Code works inside your project: CLAUDE.md for conventions, skills for processes, hooks for enforcement, and subagents for isolation. But so far, Claude has only interacted with your codebase and your terminal.

In the next post, we'll connect Claude to the outside world — MCP servers for integrating with tools like GitHub, and a look at how everything we've built carries over to CI/CD and automated workflows. We'll close the series with the full picture: a feature built from GitHub issue to merged PR, using the complete stack.

---

*This is Part 5 of the "Building With Claude Code" series. Previously: [Part 4 — Hooks: When You Need Guarantees, Not Suggestions](/tutorials/claude-code/). Next up: [Part 6 — MCP & The Bigger Picture](/tutorials/claude-code/)*
