---
title:  "Agents Tool Use - Adding Actions to your AI agent"
date:   2025-11-03 17:19:51 +0300
categories: agents ai
---

## Introduction
 So far, our **AI Code Assistant** has learned how to think - it can read code and make intelligent suggestions.

 But it still can't do anything.  
 It just talks about code instead of working with code.

 In this part, we will change that by introducing **tools** ; external actions your agent can perform beyond text generation.

 By the end of this tutorial, your assistant will:
 * Choose between **analyzing** code or **reading** from a file,
 * Dynamically decide what to do,
 * And execute the right tool to complete the job

## Concept: Tools are Actions
 In the real worlds, a human code reviewer might:
 * Open files
 * Edit code
 * Run tests


 LLMs cannot directly do that.
 With tool calling, we can connect them to real Python functions that perform these tasks.
 Tools are just code functions that the LLM can request via a formatted response to be executed. LLMs cannot directly execute this code

## The Tool Registry
As we have seen above, tools are really just functions that we call after giving it a task and asking it to determine the best function to use.  
The tools registry is simply a list of callable functions that the LLM has access to. We will give our agent multiple tools, like:


 | Tool Name      | Description                      | Function                  |
 | -------------- | -------------------------------- | ------------------------- |
| `read_file`    | Read code from a file            | `def read_file(path)`      |
| `analyze_code` | Analyze a string of code         | LLM reasoning via API call |

The agent decides; "Do I need to read a file, test or analyze the given snippet?"  
We are keeping our tools simple here so that we focus on the concepts.

Let's go ahead and build some tools, and give them to the code assistant that decides which tool to use based on user instruction, and run the tool.


### Step 1 Define the tools


```python
import openai
import os
from typing import Callable, Dict

def read_file(filepath: str) -> str:
    """Read contents of a Python file"""
    if not os.path.exists(filepath):
        return f"File not found: {filepath}"
    
    with open(filepath, "r") as f:
        return f.read()

def analyze_code(code: str) -> str:
    """Ask an LLM to analyze the provided code."""
    prompt = f"""
    You are a helpful code review assistant.
    Analyze the following Python code and suggest one improvement.

    Code:
    {code}
    """

    response = openai.responses.create(model="gpt-4.1-mini",input=[{"role":"user","content":prompt}])

    return response.output_text
```

### Step 2: Build a simple tool registry
The tools registry is a class that holds a list of Python functions, and allows us to execute a given function


```python
class ToolRegistry:
    """Holds available tools and dispatches them by name."""
    def __init__(self):
        self.tools: Dict[str,Callable] = {}
    
    def register(self, name:str, func: Callable):
        self.tools[name] = func

    def call(self, name:str, *args, **kwargs):
        if name not in self.tools:
            return f"Unknown tool: {name}"
        return self.tools[name](*args, **kwargs)
    
```

### Step 3: Agent with tool use
We will build on the CodeReview agent from the agent loop tutorial [CodeReviewAgent](https://github.com/asanyaga/ai-agents-tutorial/blob/main/code_review_agent.ipynb)
Most recent LLM models have support for *function calling*. Function calling is a capability of language models that enables them to output text in a structured format suitable for executing a function or code.

**NOTE:**  Not all LLM models have function calling and structured output capabilities. For models that dont support function calling, this capability is achieved with well structured prompts and instructions.
1. Initialize the agent with a `ToolRegistry`
```python
class CodeReviewAgentWithTools:
    def __init__(self, tool_registry: ToolRegistry, model= "gpt-4o-mini"):
        self.tools = tool_registry
        self.model = model
```
2. Update the prompt in `think()` method to instruct the model to output its response in a specicic format, JSON. Here we are choosing JSON because it is a popular well supported format. LLMs will have seen a lot of JSON examples in their training and should be able to output the tools calls in the specified JSON format.

```python
    def think(self, user_input: str):
        """LLM decides which tool to use"""

        prompt = f"""
        You are a code assistant with access to the tools below.

        Available tools:
        - read_file(filepath)
        - analyze_code(code)
        - write_tests(test_code)

        Decide which tool is most appropriate based on the user input below.
        Reply ONLY with the tool call to make in JSON format {{"tool": "tool_name", "args": ["arg1", "arg2"]}} (e.g., {{"tool":"patch_file", "args":["file_path","content"]}}

        Examples:
        read_file("main.py")
        analyze_code("def foo():pass")
        write_tests("def foo():pass")

        user input: {user_input}
        """

        response = openai.responses.create(model=self.model, input=[{"role":"user","content":prompt}])

        return response.output_text
```
3. Update the `act()` method to call a tool based on the LLM selected tool. 
```python
    def act(self, decision:str):
        """Execute the chosen tool and output the result."""
        try:
            parsed = json.loads(decision)
            tool_name = parsed["tool"]
            args = parsed.get("args",[])

            result = self.tools.call(tool_name,*args)
            return result
        except Exception as e:
            error_msg = f"Error executing tool: {e}"
            return error_msg
```


```python
import json
import openai
class CodeReviewAgentWithTools:
    def __init__(self, tool_registry: ToolRegistry, model= "gpt-4o-mini"):
        self.tools = tool_registry
        self.model = model
    
    def think(self, user_input: str):
        """LLM decides which tool to use"""

        prompt = f"""
        You are a code assistant with access to the tools below.

        Available tools:
        - read_file(filepath)
        - analyze_code(code)

        Decide which tool is most appropriate based on the user input below.
        Reply ONLY with the tool call to make in JSON format {{"tool": "tool_name", "args": ["arg1", "arg2"]}} (e.g., {{"tool":"read_file", "args":["sample.py"]}}

        Examples:
        read_file("main.py")
        analyze_code("def foo():pass")

        user input: {user_input}
        """

        response = openai.responses.create(model=self.model, input=[{"role":"user","content":prompt}])

        return response.output_text
    
    def act(self, decision:str):
        """Execute the chosen tool and record the result."""
        try:
            parsed = json.loads(decision)
            tool_name = parsed["tool"]
            args = parsed.get("args",[])

            result = self.tools.call(tool_name,*args)
            return result
        except Exception as e:
            error_msg = f"Error executing tool: {e}"
            return error_msg
```

### Register tools and run a demo
1. Add tools to the registry
2. Create the agent with tools
3. Ask the agent what to do given a user request and tools
4. Ask the agent to act on the decision

**NOTE:** Usually the above sequence of actions will run consecutively and you might have seen agent frameworks that have a single `run` or similar command. We are keeping things simple here to demonstrate concepts


```python
registry = ToolRegistry()

# Register the tools we defined above
registry.register("read_file", read_file)
registry.register("analyze_code",analyze_code)

agent = CodeReviewAgentWithTools(registry)

user_request = "Please review the code in sample.py"

# We give the agent an observation (user request) to think about and give us a decision
decision = agent.think(user_input=user_request)
print(f"Agent Decision:\n{decision}")

# We give the agent the decision to act on
result = agent.act(decision)
print(f"Tool Call Result:\n{result}")

```

## Human in the loop (HIL)
Human in the loop is the concept of requiring that the agent requests and receives human consent before performing actions. Agents are meant to be autonomous human assistants but they are not infallible and more importantly cannot be held accountable.  
In situations where actions might be risky, it is good practice to require that a human review an agent's actions and approve or reject the agents proposed action.

Below we shall implement a simple human in the loop by having the agent ask for human input before performing an action. If the response is "YES" the agent performs the action, otherwise it just exits

We will modify our CodeAssistent agent to request input before acting

```python
    def act(self, decision: str):
        """Execute the chosen tool command with humn consent."""
        response = input(f"I want to act on {decision}. Reply with YES or NO")

        #....existing act code...
```


```python
import json
import openai
class CodeReviewAgentWithToolsHIL:
    def __init__(self, tool_registry: ToolRegistry, model= "gpt-4o-mini"):
        self.tools = tool_registry
        self.model = model
    
    def think(self, user_input: str):
        """LLM decides which tool to use"""

        prompt = f"""
        You are a code assistant with access to the tools below.

        Available tools:
        - read_file(filepath)
        - analyze_code(code)

        Decide which tool is most appropriate based on the user input below.
        Reply ONLY with the tool call to make in JSON format {{"tool": "tool_name", "args": ["arg1", "arg2"]}}  

        Example:
        {{"tool":"read_file", "args":["sample.py"]}}

        user input: {user_input}
        """

        response = openai.responses.create(model=self.model, input=[{"role":"user","content":prompt}])

        return response.output_text
    
    def act(self, decision: str):
        """Execute the chosen tool command."""
        response = input(f"I want to act on {decision}. Reply with YES or NO")
        if response.lower()=="yes":
            try:
                parsed = json.loads(decision)
                tool_name = parsed["tool"]
                args = parsed.get("args",[])

                result = self.tools.call(tool_name,*args)
                return result
            except Exception as e:
                error_msg = f"Error executing tool: {e}"
                return error_msg
```


```python
code_snippet = """
def divide(a,b):
    return a/b
"""

agent_hil = CodeReviewAgentWithToolsHIL(tool_registry=registry)
user_request = f"Please review this code {code_snippet}"

# We give the agent an observation (user request) to think about and give us a decision
decision = agent_hil.think(user_input=user_request)
print(f"Agent Decision: \n{decision}")

# We give the agent the decision to act on
result=agent_hil.act(decision=decision)
print(f"Tool Call Result:\n{result}")
```

**Full Source Code Here:**  [Agent Tasks Jupyter Notebook](https://github.com/asanyaga/ai-agents-tutorial/blob/main/part-2-agent-tools.ipynb)

### What's next
In the next part of this series we will give our agent memory.

We will see why memory is an important part of agents and how to use and manage the different types of memory.
