---
title: "Agent Evals: How to Design Good Test Cases"
date: 2026-11-18 08:00:00 +0300
categories: ai-strategy
tags: agents ai evals
excerpt: "A useful agent test case does more than provide an example prompt and an expected response. It defines the conditions the agent operates under, what success looks like, and the behaviour you're willing to accept along the way."
---

In [the edition on designing evaluation datasets]({% post_url 2026-11-12-designing-evaluation-datasets-for-ai-agents %}), we looked at the components of an agent evaluation system: datasets, test cases, executions, evaluators, and results.

This article is part of a series on agent evaluations, but you don't need to have read the earlier editions to follow along. I'll explain the relevant concepts as we go. If you'd like a broader understanding of how the different parts of an evaluation system fit together, the earlier editions provide that context.

We established that an evaluation dataset is a collection of test cases, and that each test case describes a scenario we want to evaluate.

For now, we'll focus on one question: what exactly goes into a good agent test case?

Let's work through an example.

## Start With the Task, Not the Test

Imagine we're building a customer support agent for an online retailer. The agent can look up orders, check refund eligibility, issue refunds, and communicate with customers.

A customer sends the following message:

> "I ordered a pair of headphones last week. They arrived damaged. Can I get a refund?"

We want to establish whether our agent can handle this request correctly.

A first attempt at a test case might look like this:

- **Input:** Customer requests a refund for damaged headphones.
- **Expected outcome:** Agent issues a refund.

This looks reasonable, but it's missing important information.

What if the customer is outside the refund window? What if the order has already been refunded? What if the agent refunds the wrong order? And what if the agent issues a refund without checking whether the customer is eligible?

We need to define not just what the agent should accomplish, but also what counts as acceptable behaviour while accomplishing it.

## What Does Success Look Like?

For our refund agent, a successful execution might involve:

- Identifying the correct customer and order.
- Checking the order's refund eligibility.
- Applying the company's refund policy.
- Issuing the correct refund if eligible.
- Informing the customer of the outcome.

Notice that we're describing more than the agent's final response.

An agent could tell the customer that their refund has been processed without actually issuing one. It could also issue the correct refund but skip the mandatory eligibility check.

In the first case, the outcome is incorrect. In the second, the outcome may be correct, but the agent violated an important business rule.

A correct outcome doesn't necessarily mean the agent behaved correctly.

This gives us two broad dimensions to evaluate:

- **Outcome:** Did the agent achieve the intended result?
- **Process:** Did the agent take acceptable actions to achieve that result?

Depending on the application, one or both may matter.

## Anatomy of a Test Case

Let's make our original example more precise.

**Test case: Refund an eligible damaged order**

**1. Input**

Customer message: *"My headphones arrived damaged. I'd like a refund."*

**2. Initial environment state**

- Customer is authenticated.
- Order #1234 belongs to the customer.
- Order total is $120.
- Order was delivered seven days ago.
- Order has not previously been refunded.
- The refund policy permits refunds for damaged goods within 30 days.

**3. Expected outcome**

- A refund of $120 is issued against order #1234.
- The order is marked as refunded.
- The customer receives an accurate confirmation.

**4. Expected tool usage**

The agent is expected to use its tools to:

- Retrieve the customer's order.
- Verify refund eligibility.
- Issue the refund against the correct order.

We could be more specific and require calls such as `lookup_order`, `check_refund_eligibility`, and `issue_refund`, including expected arguments.

**5. Expected trajectory**

The agent should:

> Retrieve order → Verify eligibility → Issue refund → Confirm outcome.

This defines the important sequence of actions, rather than just the final result.

**6. Constraints**

- No other orders may be modified.
- The agent must not issue a refund before checking eligibility.
- No duplicate refund may be issued.
- The agent must not claim success if the refund operation fails.

We now have a much more useful test case. It describes what the agent receives, the conditions under which it operates, what success looks like, and the behaviour we expect.

Not every test case needs all these elements. A simple question-answering agent might require only an input and a reference answer. An agent that modifies external systems may need considerably more.

## Expected Outcomes, Tool Calls, and Trajectories

It's worth distinguishing these expectations because they answer different questions.

**Expected outcome** describes the result we want. For example, order #1234 should have a completed refund of $120. We might verify this by inspecting the order and payment records after execution.

**Expected tool calls** describe which tools the agent should use and, where necessary, the arguments it should provide. For example:

```
issue_refund(order_id=1234, amount=120)
```

We might verify this by inspecting the tool calls recorded during execution.

**Expected trajectory** describes how the agent should progress through the task. For example, eligibility must be verified before the refund is issued. We might verify this by examining the sequence of actions recorded in the agent's execution trace.

An execution trace is a record of what happened during a run. Depending on the system, it may contain model responses, tool calls, tool results, intermediate decisions, timestamps, and errors. It provides the evidence we need to evaluate not just the final result, but also the agent's behaviour.

## How Strictly Should We Specify the Trajectory?

Suppose our agent can verify refund eligibility in two ways:

- Call a dedicated `check_refund_eligibility` tool.
- Retrieve the order details and refund policy separately, then determine eligibility.

Both approaches may be valid. Should we fail the agent because it didn't use our preferred tool?

Not necessarily.

For some tasks, we need an exact sequence of tool calls. For others, several different approaches are acceptable. We can therefore specify expectations at different levels:

- **Exact actions:** The agent must call a particular tool with particular arguments.
- **Required behaviour:** The agent must verify eligibility, regardless of how it does so.
- **Ordering constraints:** Eligibility verification must happen before issuing the refund.
- **Reference trajectory:** The agent's sequence of actions can be compared against a known successful execution.
- **Outcome only:** Any permitted sequence is acceptable provided the correct result is achieved.

The appropriate level depends on what we're testing. Overly restrictive trajectory requirements can penalize an agent for finding a valid alternative solution.

In general, we should specify the behaviours that matter, rather than unnecessarily prescribing every step.

## Initial State Matters

There's another detail worth examining.

Suppose we execute our refund test case and the agent successfully issues the $120 refund. We run the same test again. This time, the agent discovers that the order has already been refunded and declines to issue another refund.

Has the agent become inconsistent?

No. We changed the conditions under which it was operating. The second execution was not equivalent to the first.

A test case must define not just the input, but also the relevant state of the environment. For an agent interacting with external systems, we may need to restore database records, reset mock services, or recreate other dependencies before each execution. Otherwise, differences in environment state can be mistaken for differences in agent behaviour.

Even with identical starting conditions, an agent may take different paths across executions. Repeated runs can help us understand how reliably it satisfies the same requirements. This is one reason an evaluation harness needs to do more than send prompts to a model and collect responses. We'll explore that in a later edition.

## Test Cases, Executions, and Evaluators

Let's return to the terminology from the previous edition.

A test case specifies a scenario, its initial conditions, and the expected outcomes or behaviours. A test execution is an individual run of the agent against that test case.

During execution, we capture evidence such as:

- The final response and resulting state.
- Actual tool calls and arguments.
- The sequence of agent actions.
- Errors, timing, and other execution details.

The evaluators then assess that evidence against the relevant expectations. For our refund example, we might use:

- A database check to verify that the correct order was refunded.
- A tool-call check to verify that the correct refund amount was submitted.
- A trajectory check to confirm eligibility was verified before issuing the refund.
- An LLM-based evaluator to assess whether the customer's confirmation was accurate and understandable.

Different aspects of success may require different evaluation methods. We don't need an LLM to judge whether a $120 refund was issued; a deterministic assertion against the transaction record would be more appropriate. On the other hand, assessing whether the agent communicated clearly may require a more flexible evaluator.

An evaluator may be configured for one test case or shared across many. The important distinction is that the test case describes the expected behaviour, while the evaluator determines whether the observed behaviour satisfies the relevant criteria. We'll examine evaluator design in a subsequent edition.

## Bringing It Together

A useful agent test case does more than provide an example prompt and an expected response. It establishes:

- **Input and initial state:** What the agent receives and the conditions under which it operates.
- **Expected outcomes:** What the agent should accomplish.
- **Expected behaviour:** What tools, actions, or trajectories are required or acceptable.
- **Constraints:** What the agent must not do.

Not every evaluation needs to inspect the agent's trajectory. Sometimes the final outcome is sufficient. But when an agent takes actions, interacts with external systems, or operates under business rules, evaluating the process can be just as important as evaluating the result.

Good test cases help us distinguish between an agent that produces convincing responses and one that actually completes tasks correctly and within the required constraints.

In an upcoming edition, we'll look at how to execute these test cases using an evaluation harness: setting up the environment, capturing execution traces, and making repeated evaluations comparable.
