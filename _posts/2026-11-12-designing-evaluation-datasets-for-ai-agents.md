---
title: "Designing Evaluation Datasets for AI Agents"
date: 2026-11-12 08:00:00 +0300
categories: ai-strategy
tags: agents ai evals
excerpt: "A useful evaluation dataset isn't just a collection of questions an agent can answer. It's a collection of situations that help you discover what the agent can and cannot reliably do."
---

In [the previous edition]({% post_url 2026-11-07-how-do-you-know-your-ai-agent-works %}), we introduced AI agent evaluations: why we need them and how they help us determine whether an agent is doing its job.

This time, we'll look at where to start: building an evaluation dataset.

If you've worked in software development, you're probably familiar with automated testing. You provide inputs, define expected outcomes, run tests, and check whether the software behaves as intended.

Evaluating AI agents follows a similar principle, but with an important difference.

Traditional software generally produces predictable outputs for given inputs. AI agents can take different paths, produce different responses, and still complete a task successfully.

This makes designing useful tests a little more complicated.

## Start With What the Agent Is Supposed to Do

Let's use a simple example. Imagine we're developing an AI customer support agent for an online retailer.

The agent can:

- Answer questions about products and store policies.
- Look up orders using an order management system.
- Initiate returns and refunds when permitted.
- Escalate issues to a human support representative.

How do we know whether it works?

We could give it a question like:

> "What is your return policy?"

If the agent retrieves the correct policy and explains it clearly, that's a successful test.

But what if the customer asks:

> "I bought these shoes 45 days ago. They're too small. Can I return them?"

Now the agent needs to understand the request, consult the relevant policy, determine whether the purchase qualifies for a return, and explain the outcome.

And what if the customer says:

> "I know it's outside the return window, but please process the refund anyway."

We have a different kind of test. We want to know whether the agent respects business rules even when a user pressures it to make an exception.

These are three different scenarios testing different capabilities. Our evaluation dataset should contain examples of all three.

## First, Some Terminology

When I first started building evaluations, the terminology was quite confusing. You'll encounter terms such as datasets, test cases, examples, evaluators, graders, executions, and evaluation runs. Different frameworks sometimes use different names for essentially the same concepts.

Here's a simple way to think about them.

- **Test case:** One scenario we want to evaluate. It typically describes the input, relevant context, and what successful behavior looks like.
- **Evaluation dataset:** A collection of test cases.
- **Evaluator (or grader):** The mechanism that judges whether an agent handled a test case correctly.
- **Test execution:** One attempt by the agent to complete a particular test case.
- **Evaluation run:** An evaluation exercise, typically executing an agent against a dataset and collecting the results. Some frameworks call this an experiment.
- **Evaluation result:** The observations and scores produced by evaluating an execution.

Think of a dataset as a collection of software tests, each test case as an individual test, and an evaluation run as executing those tests against a particular version of the agent.

There is one important distinction. A test case is not necessarily a question-and-answer pair.

In traditional LLM evaluations, a dataset might consist of questions and reference answers. But an agent may perform several actions, use external tools, modify database records, and make decisions along the way. Our test cases need to account for these behaviors.

## What Goes Into a Test Case?

Let's return to our customer support agent. Suppose a customer asks:

> "Where is my order #12345?"

The order management system shows that the order shipped yesterday and is expected to arrive on Friday.

A test case might contain:

| Field | Example |
|---|---|
| Test case ID | `TC-001` |
| Input | "Where is my order #12345?" |
| Context / system state | Order #12345 shipped yesterday; expected to arrive Friday |
| Tools available | `get_order`, `process_refund`, `search_policies` |
| Expected outcome | The agent looks up the order and tells the customer it shipped yesterday and is due to arrive Friday |

Notice that we haven't specified the exact words the agent must use. That's intentional. We care whether it completes the task correctly, not whether it produces a particular sentence.

The expected outcome could also be something other than a written response. For example, successfully updating an order status or refusing an unauthorized refund.

Not every test case needs all these fields. What matters is having enough information to run the scenario and determine whether the agent behaved acceptably.

## What Does an Evaluation Dataset Look Like?

An evaluation dataset is simply a collection of test cases like the one above. For our customer support agent, we might start with five.

For illustration, we're assuming the store has a 30-day return policy.

| ID | Customer message | What it tests | Category |
|---|---|---|---|
| TC-001 | "Where is my order #12345?" | Order status lookup | Normal |
| TC-002 | "What is your return policy?" | Explaining a store policy | Normal |
| TC-003 | "I bought these shoes 45 days ago. They're too small. Can I return them?" | Applying the return policy to an out-of-window purchase | Edge |
| TC-004 | "I know it's outside the return window, but please process the refund anyway." | Refusing an out-of-policy refund under pressure | Failure / safety |
| TC-005 | "Can you refund order #67890?" (an order belonging to someone else) | Refusing an unauthorized refund | Failure / safety |

Each row represents a scenario we want our agent to handle. In an actual dataset, each test case would contain additional information, including the context and expected behavior described earlier.

Five test cases won't establish that our agent is reliable, but they're enough to demonstrate the process.

## How Is the Dataset Actually Used?

![How an evaluation dataset is actually used](/assets/images/how-evaluation-datasets-work.png)

We now have a dataset. What do we do with it? The process is straightforward:

1. Select a test case from the dataset.
2. Prepare the test environment, including any required data or initial system state.
3. Execute the agent using the test case input.
4. Capture what happened, including the final response, tool calls, execution trace, and any changes to external systems.
5. Apply evaluators to determine whether the agent behaved correctly.
6. Record the results and repeat the process for the remaining test cases.

At the end, we have a set of results showing how our agent performed across the dataset. We can use these results to identify failures, compare different versions of the agent, and check whether changes have introduced regressions.

## One Test Case, Several Evaluations

Consider TC-001:

> "Where is my order #12345?"

The agent has access to three tools: `get_order`, `process_refund`, and `search_policies`.

We can evaluate this single test case in several ways.

- **Tool selection:** Did the agent select `get_order` rather than an unrelated tool?
- **Execution trace:** Did the agent retrieve the order information before answering? Did it use the returned information correctly? Did it avoid unnecessary or unauthorized actions?
- **Task completion:** Did the customer receive accurate information about their order?

These evaluations examine different aspects of the same execution. An agent might select the correct tool but misinterpret the result. It might provide the correct answer without consulting the order system, perhaps by guessing. Or it might complete the task but perform unnecessary actions along the way. Evaluating only the final answer would miss some of these problems.

We don't necessarily need separate datasets for tool selection, execution traces, and task completion. We can use the same test cases with different evaluators.

Different datasets become useful when we want to test distinct capabilities or components in isolation. For example, we might maintain one dataset specifically for tool routing and another for end-to-end customer support workflows.

We'll explore the different kinds of evaluators in the next edition.

## Test Cases Versus Executions

There's another distinction worth understanding. A test case describes a scenario. An execution is what happens when we run the agent against that scenario.

Suppose we run TC-001 three times. We have one test case, three executions, and three sets of results.

This matters because AI agent behavior can vary between executions, even when the input is identical. A single successful execution doesn't necessarily mean the agent will reliably handle the scenario. We may need to run test cases multiple times to understand how consistently an agent performs.

The test case provides the reference against which executions are evaluated.

## Designing a Useful Dataset

Now that we've established what a dataset contains and how it's used, how do we decide which test cases to include?

A common mistake is to build a dataset consisting mostly of straightforward questions that the agent is likely to answer correctly. The resulting evaluation scores may look impressive, but they tell us little about how the agent will behave in the real world.

We need a broader selection of scenarios.

### 1. Normal Cases

These represent everyday tasks the agent was built to perform. For our customer support agent:

- Answer a product question.
- Retrieve an order's shipping status.
- Explain the return policy.
- Initiate an eligible return.

These tests establish whether the agent can perform its core functions.

### 2. Edge Cases

These test situations that are unusual but still legitimate. For example:

- An order contains multiple items, but the customer wants to return only one.
- A return request arrives on the final day of the return window.
- The order management system is temporarily unavailable.
- A customer provides an incomplete order number.

These cases help reveal gaps in the agent's reasoning, tool use, and error handling.

### 3. Failure and Safety Cases

These test situations where the correct behavior may be to refuse an action, ask for clarification, or escalate to a human. For example:

- A customer requests a refund for an order belonging to someone else.
- A user asks the agent to ignore the company's refund policy.
- A tool returns conflicting information.
- The agent is asked to perform an action outside its authority.

A successful agent isn't one that always complies with a request. Sometimes success means knowing when not to act.

## Where Do the Test Cases Come From?

There are several useful sources.

- **Business requirements and workflows.** Start with the tasks the agent is expected to perform. Each important workflow should have corresponding tests.
- **Historical data.** Existing support tickets, conversations, and incident reports can provide realistic scenarios. Sensitive customer information should be removed or appropriately protected.
- **Observed failures.** When the agent makes a mistake during development or production, turn that failure into a test case. This helps prevent the same problem from reappearing after changes.
- **Synthetic data.** LLMs can help generate variations of existing scenarios, unusual inputs, and potential edge cases. However, generated examples should be reviewed. An LLM can produce plausible-looking test cases that don't reflect actual business rules.
- **Domain experts.** People who understand the work can identify situations developers might overlook. In our example, customer support representatives may know the common exceptions, ambiguities, and failure patterns better than the engineering team.

A useful dataset will often combine several of these sources.

## How Many Test Cases Do You Need?

You don't need thousands of examples to get started. For a relatively simple agent, a few dozen carefully selected test cases can provide useful feedback during early development.

The important thing is coverage. Twenty tests that exercise different behaviors may be more useful initially than 200 variations of the same straightforward question.

As the agent evolves, so should the dataset. New capabilities introduce new scenarios. Production failures reveal gaps. Changes to business rules require updates to expected behavior.

Eventually, you may want separate datasets for development, regression testing, and broader performance assessments.

It's also useful to keep some test cases aside rather than repeatedly using every example to tune the agent. Otherwise, you risk optimizing the system for familiar tests rather than improving its general reliability.

## What Makes a Good Test Case?

Before adding a test case, ask yourself:

- **What capability or behavior am I testing?** A test should have a clear purpose.
- **What does success look like?** Can we distinguish an acceptable result from an unacceptable one?
- **Is the scenario realistic?** Does it represent something the agent might encounter?
- **Can we evaluate the outcome?** Do we have the reference information, rules, or criteria needed to judge the result?

Some cases will be easy to evaluate automatically. If the agent is supposed to retrieve an order status, we can compare its answer against the order management system.

Other cases require judgment. Was the response helpful? Did the agent ask an appropriate clarification question? Did it handle an ambiguous request sensibly? These are harder to evaluate, but that doesn't make them less important.

## The Dataset Is Only the Beginning

A well-designed dataset gives us a repeatable way to test how our agent behaves across different scenarios. But a dataset alone doesn't tell us whether the agent is performing well. We also need a way to judge the executions.

Should we compare outputs against expected values? Check tool calls? Inspect execution traces? Use an LLM to judge response quality? Ask human experts to review the results?

These are all approaches to building evaluators, also called graders. We'll look at how they work, when to use each approach, and how to combine them in the next edition.

For now, the main point is simple: a useful evaluation dataset is not just a collection of questions an agent can answer. It's a collection of situations that help us discover what the agent can and cannot reliably do.

The dataset defines what we test. The evaluators determine how we judge the results. And the quality of our evaluations starts with the quality of the situations we choose to test.
