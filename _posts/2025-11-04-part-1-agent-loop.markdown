---
title:  "From Prompts to Agents: Understanding the AI Agent Loop"
date:   2025-11-03 17:19:51 +0300
categories: agents ai
---
## Introduction
You have probably heard the term **AI Agent**, often mentioned, but not always clearly explained. In this tutorial series, we will demystify what AI agents really are. 

We will start with the concept of the **AI agent loop** - the core idea behind how intelligent systems *observe, think, and act.*  
We will also take a look at the basic building blocks of AI agents, which we will build step by step throughout the series.

In this tutorial, you will:
* Understand the observation-thought-action cycle
* Implement a basic **agent loop** in Python
* See how this looop can evolve into a more advanced agent with goals, tools goals and memory

### To run the Jupyter notebooks
* Make sure you have [Python](https://www.python.org/) installed
* Install and setup [OpenAI Python library](https://platform.openai.com/docs/libraries). In this tutorial we will be using OpenAI LLM models but you can use any LLM model or provider that you are familiar with since the concepts are not provider specific.
* Install and setup [Jupyter Notebook](https://jupyter.org/install)

## What is an AI Agent
An agent is a system that:
1. **Observes** its environment
2. **Thinks** based on goals and context
3. **Acts** in that environment

In other words:
> **Agent = Observe -> Think -> Act**

*Agent loop:*

![Agent Loop](/assets/images/agent-loop.png)

## Agent Building Blocks
Now that we have seen the **Agent Loop** - *Observe -> Think -> Act* let's take a look at the core building blocks that make this work in practice.
We'll explore each of these in more details later in the series, but here is a simple overview to get started:
1. **Prompt:** - This is how we communicate the goal and situation to the agent. It includes goals, instruction and context to help the agent reason effectively.
2. **LLM:** The *brain* of the agent. It processes the prompt and context, reasons through the situation, and decides what to do next by producing natural language outputs or strucutred actions.
3. **Tools:** Tools allow the agent to go beyond text; to fetch data, run code, search the web, send emails, or intereact with APIs. Tools use connects LLM reasoning and real world action.
4. **Memory:** Memory lets the agent recall instructions, past experiences, conversations or actions so that it can act more consistently and intelligently over time.

These four building blocks work together inside the agent loop. The prompt tells the LLM what to do, the LLM decides an action, tools let it perform that action, and memory helps it learn and adapt from experience.

*Agent building blocks:*

![Agent building blocks](/assets/images/agent-building-blocks.png)


### The Code Reviewer Assistant
Imagine you have hired an assistant to help review code
* You show them a piece of code and an instruction(*observation*)
* They think about it and and form their opinion of the code (*thought*)
* They reply with a comment (*action*)

So how does an AI Agent Observe, then Think and Act.

### Implementing the Agent
Let's see how this works by creating a simple code review agent class.
The agent:
* Receives a code snippet *(observe)* 
* Uses an LLM to review the code snippet and return its review *(think)* 
* Prints out its review *(act)*


```python
import openai

class CodeReviewAgent:
    def __init__(self, llm_model="gpt-4o-mini"):
        self.model = llm_model
    
    def observe(self, code_snippet: str):
        return code_snippet.strip()
    
    def think(self, observation: str):
        """Agent uses an LLM to reason about what to do"""
        prompt = f"""
        You are a helpful code review assistant.
        Analyze the following Python code and suggest one improvement or hughlight on potential issue

        Code:
        {observation}
        """

        response = openai.responses.create(model=self.model,
                                           input=[{"role": "user","content": prompt}])
        
        thought = response.output_text

        return thought
    
    def act(self, thought: str):
        """Agent takes an action - here, it's returning a suggesting"""
        print (f"Action:  {thought}")
    
```

Below we run the review agent


```python
agent = CodeReviewAgent()

code_snippet = """
def divide(a,b):
    return a/b
"""

goal_complete = False

while not goal_complete:
    # Step 1: Observe
    observation = agent.observe(code_snippet=code_snippet)
    print(f"\nOBSERVATION: {observation}")

    # Step 2: Think
    thought = agent.think(observation=observation)
    print(f"\nTHOUGHT: {thought}")

    #Step 3: Act
    action = agent.act(thought)
    goal_complete = True

```

In the introduction we mentioned that an agent executes tools in a loop until a task is done. In our first example above we can think of our agent as having just one loop because it has a simple task and one tool action `print()` after which the task is complete.  
In later parts of the series we will see how to implement multiple loops with multiple tools

**Full Source Code Here:**  [Agent Loop Jupyter Notebook](https://github.com/asanyaga/ai-agents-tutorial/blob/main/part-1-agent-loop.ipynb)

### What's next
In the next part of the series shall see how to define tools and configure the code review agent to make decisions about which tool to use.