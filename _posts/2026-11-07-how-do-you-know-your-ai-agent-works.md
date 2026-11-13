---
title: "How Do You Know Your AI Agent Works?"
date: 2026-11-07 08:00:00 +0300
categories: ai-strategy
tags: agents ai evals
excerpt: "Your agent runs, it answers, the demo lands. But a demo that works on the questions you thought to ask is not the same as a system you can put in front of real users. Evals are how you tell the difference."
---

You've built an agent. You ask it a few questions, it reasons through them, calls the right tools, and comes back with sensible answers. The demo lands. Everyone nods.

Then the doubt creeps in. How do you actually know it's ready for real users?

Real users ask things you never anticipated. Tools return data in shapes you didn't plan for. Retrieval pulls back the wrong document, or only half of the right one. And the failure mode that should worry you most isn't the agent throwing an error. It's the agent doing the wrong thing confidently, in fluent, well-formatted prose that looks exactly like a correct answer.

That is the problem evals exist to solve.

## What Are Agent Evals?

With traditional software tests, you assert that a fixed input produces a fixed output. `add(2, 2)` returns `4`, every time, or the test fails. Clean and deterministic.

Agents don't work like that. For the same question, an agent might take several different valid paths to a good answer. And the output is language, not a number you can match exactly. Worse, a response can be grammatically perfect, confident, and completely wrong.

So "did it return the expected string?" is the wrong question. Evals ask two better ones: **was the answer actually good?** And **was the behaviour along the way appropriate?** The second question matters as much as the first, because an agent that stumbles into a right answer by reading a file it shouldn't have access to is not an agent you can trust.

## What Should We Evaluate?

Let's make it concrete. Say you've built an agent that answers employees' questions about company policy. Someone asks: *"How many annual leave days can I carry forward into next year?"*

A single "did it work?" flag hides most of what you care about. Pull it apart and there are at least five separate things worth measuring:

1. **Did it retrieve the right information?** The answer has to come from the current leave policy, not last year's version and not the parental-leave document that happened to look similar.
2. **Is the answer correct?** The right source still has to be read and summarised accurately. "Up to five days" and "up to fifteen days" are very different to the person planning their holiday.
3. **Did it use the right process?** The correct tools, in a sensible order, respecting access restrictions. An agent that can read any employee's records to answer a general policy question is a problem even when its answer is right.
4. **How did it handle uncertainty?** If the policy is ambiguous or the question is underspecified, did it ask a clarifying question or flag the uncertainty, or did it guess and present the guess as fact?
5. **Was it efficient?** Latency, number of tool calls, and cost. An answer that's correct but takes thirty seconds and a dozen tool calls may still be unusable.

These are genuinely separate questions. An agent can nail retrieval and still bungle the summary, or produce a perfect answer through a process you'd never approve of. One success metric can't see any of that.

## How Do We Measure Something So Subjective?

Some of these checks are mechanical, and you should lean on those first because they're cheap and unambiguous:

- Was the expected API or tool actually called?
- Is the output valid JSON, or whatever structure the next step needs?
- Did it finish within the time and cost budget?
- Did it stay away from data it shouldn't touch?

The harder part is judging answer quality, which really does require judgment. A few approaches, each with trade-offs:

- **Reference answers.** Have a subject-matter expert write the ideal answer for each test case, then compare against it. Accurate, but expensive to produce and to keep current as policies change.
- **Rule-based checks.** Assert that certain facts, phrases, or constraints appear (or don't). Fast and deterministic, but brittle, and blind to anything you didn't think to encode.
- **Human review.** The gold standard for nuance. It doesn't scale, and it's slow.
- **LLM-as-a-judge.** Use another model to grade the answer against a rubric. Scales well and catches nuance rule-based checks miss, but the judge has its own biases and needs validating against human judgment before you trust it.

In practice you mix these. Mechanical checks for the things that are objective, and some combination of the rest for the things that aren't.

## Evals Aren't Just for the Finished Product

It's tempting to think of evals as a final gate: build the thing, then test whether it's good enough to ship. But they earn their keep much earlier.

When an agent fails, the useful question is *why*. Was it retrieval that pulled the wrong document? Instructions the model misread? A bad tool choice? The model itself? Evals that target individual components let you isolate the failure instead of guessing.

They also catch regressions. You tweak a prompt to fix one behaviour and quietly break three others, which is almost a law of working with these systems. Without evals you find out from users. With them you find out before you ship.

And they let you compare. Swap the model, change the retrieval strategy, restructure the prompt, try a different architecture. Without evals, improving the agent is just changing things and hoping. With them, you have numbers that tell you whether you moved forward or backward.

## You Don't Need an Elaborate System to Start

None of this requires a platform or a framework. The honest starting point is a spreadsheet.

Collect 20 to 30 representative tasks your agent should handle. Make sure the set includes the easy cases, the genuinely hard ones, and the cases where the right move is to refuse or ask for clarification rather than answer. That last category is the one most people forget, and it's where confident-wrong failures hide.

For each task, write down what success looks like. Run the agent. Record what it did. Then review the failures and look for patterns. You'll usually find that the failures cluster. Fix the cluster, not the one-off.

That's it. That's a working eval set. You can grow it into something more automated later, but the thing that actually changes how you build is the habit of checking against a fixed set of cases every time you make a change.

## From Promising Demos to Dependable Systems

Evals don't guarantee your agent will never fail. Nothing does. What they give you is honesty: a clear picture of how often it succeeds, where it breaks, and whether your changes are making it better or worse. That's the difference between a demo that happens to work on the questions you thought to ask and a system you'd stake your name on.

So before you ship the next one, it's worth asking: how are you currently deciding whether your agent is good enough? If the answer is "it worked when I tried it," you don't yet know.
