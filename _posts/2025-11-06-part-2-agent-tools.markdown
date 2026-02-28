---
title:  "Agents Tool Use: Adding Actions to your AI agent"
date:   2025-11-03 17:19:51 +0300
categories: tutorials
tags: agents ai
series: "From Prompts to Agents"
series_order: 2
---

So far, our **AI Code Assistant** has learned how to think - it can read code and make intelligent suggestions.

 But it still can't do anything. 
 It just talks about code instead of working with code.

 In this tutorial, we will change that by introducing **tools** ; external actions your agent can perform beyond text generation.

 By the end of this tutorial, your assistant will;
 * Choose from a list of actions provided by tools
 * Dynamically decide what to do
 * And execute the right tool to complete the job

 ## Concept: Tools are Actions
 In the real worlds, a human code reviewer might:
 * Open files
 * Edit code
 * Run tests


 LLMs cannot directly do that.
 With tool calling, we can connect them to code functions that perform these tasks.
 Tools are just code functions that the LLM can request via a formatted response to be executed. LLMs cannot directly execute this code.

## The Tool Registry
As we have seen above, tools are functions that we call after giving it a task and asking it to determine the best function to use.  
The tools registry is simply a list of callable functions that the LLM has access to. We will give our agent multiple tools, like:


 | Tool Name      | Description                      | Function                  |
 | -------------- | -------------------------------- | ------------------------- |
| `read_file`    | Read code from a file            | `def read_file(filepath)`      |
| `patch_file` | Edit file         | `patch_file(filepath: str, content: str)` |
| `print_review` | Print review         | `print_review(review:str)` |

The agent decides; "Do I need to read a file, edit a file or print?"  
We are keeping our tools simple here so that we focus on the concepts.

Let's go ahead and build some tools, and give them to the code assistant that decides which tool to use based on user instruction, and run the tool.

### Step 1 Define the tools


```python
import os
from typing import Callable, Dict

def read_file(filepath: str) -> str:
    """Read contents of a Python file"""
    if not os.path.exists(filepath):
        return f"File not found: {filepath}"
    
    with open(filepath, "r") as f:
        return f.read()
    
def patch_file(filepath: str, content: str) -> str:
    """Writes the given content to a file, completely replacing its current content."""
    try:
        with open(filepath, "w") as f:
            f.write(content)
        return f"File successfully updated: {filepath}. New content written."
    except Exception as e:
        return f"Error writing to file {filepath}: {e}"

def print_review(review: str):
    print(f"Review: {review}")
    return f"Printed review: {review}"
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
We will build on the CodeReview agent from the agent loop tutorial [CodeReviewAgent](/code_review_agent.ipynb)  
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
        - patch_file(filepath: str, content: str)
        - print_review(review: str)

        Decide which tool is most appropriate based on the user input below.
        Reply ONLY with the tool call to make in JSON format {{"tool": "tool_name", "args": ["arg1", "arg2"]}} (e.g., {{"tool":"patch_file", "args":["file_path","content"]}}
    

        Examples:
        read_file("main.py")
        patch_file("main.py","def foo():pass")
        print_review(review: str)
        
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

4. Add `run()` method to manage the agent loop.  
```python
    def run(self, user_input: str, max_steps:int=3):

        original_input = user_input
        for step in range(max_steps):
            print(f"Step: {step+1} of {max_steps}")
            decision = self.think(user_input)
            
            try:
                decision_parsed = json.loads(decision)
            except json.JSONDecodeError as e:
                print(f"Could not parse decision: {e}")
                user_input = f"Your response was not valid JSON.\nOriginal user request: {original_input}"
            
            if decision_parsed.get("done"):
                print(f"Task complete\nAssistant Repose:{decision}")
                return decision_parsed.get("summary")
            
            result = self.act(decision)
            user_input = f"Original user request: {original_input}\nLast assistant response{decision}\nLast tool result: {result}. continue with original user request"

        print("Loop complete. (max steps reached)")
        
        return result 
```
#### NOTE:
- We have a `max_steps` argument to prevent ending up in an infinite loop. In a later tutorial we will see how to use the language model to determine if a task is complete.
- After the first loop we update the user input to include the last tool call `user_input = f"Original user request: {original_input}. Last Tool result: {result}`. This is because llms calls are stateless. The llm does not "remember" previous requests made to it, so in a multiturn interation we have to provide details of previous interations.  
In a later tutorial we will explore memory management strategies to handle this issue.


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
        - patch_file(filepath, content)
        - print_review(review: str)

        Decide which tool is most appropriate based on the user input below if a tool is needed.
        If a tool call is needed Reply ONLY with the tool call to make in JSON format {{"tool": "tool_name", "args": ["arg1", "arg2"]}} (e.g., {{"tool":"read_file", "args":["sample.py"]}}
        Examples:
        - read_file("main.py")
        - patch_file(filepath, content)
        - print_review(review: str)

        If the task is complete respond with JSON {{"done: true, "summary:"The task is complete because"}} where the summary is the reason why the task is complete

        user input: {user_input}
        """
        print(f"Calling LLM with:\n{"-"*10}\n{prompt}\n{"-"*20}")
        response = openai.responses.create(model=self.model, input=[{"role":"user","content":prompt}])

        return response.output_text
    
    def act(self, decision:str):
        """Execute the chosen tool and record the result."""
        print(f"Acting on decision: {decision}")
        try:
            parsed = json.loads(decision)
            tool_name = parsed["tool"]
            args = parsed.get("args",[])

            result = self.tools.call(tool_name,*args)
            print(f"Tools Result: {result}")
            return result
        except Exception as e:
            error_msg = f"Error executing tool: {e}"
            return error_msg
    
    def run(self, user_input: str, max_steps:int=3):
        original_input = user_input
        for step in range(max_steps):
            print(f"Step: {step+1} of {max_steps}")
            decision = self.think(user_input)
            
            try:
                decision_parsed = json.loads(decision)
            except json.JSONDecodeError as e:
                print(f"Could not parse decision: {e}")
                user_input = f"Your response was not valid JSON.\nOriginal user request: {original_input}"
            
            if decision_parsed.get("done"):
                print(f"Task complete\nAssistant Repose:{decision}")
                return decision_parsed.get("summary")
            
            result = self.act(decision)
            user_input = f"Original user request: {original_input}\nLast assistant response{decision}\nLast tool result: {result}. continue with original user request"

        print("Loop complete. (max steps reached)")
        return result
```

### Register tools and run a demo
1. Add tools to the registry
2. Create the agent with tools
3. Ask the agent what to do given a user request and tools
4. Ask the agent to act on the decision



```python
registry = ToolRegistry()

# Register the tools we defined above
registry.register("read_file", read_file)
registry.register("patch_file",patch_file)
registry.register("print_review",print_review)

agent = CodeReviewAgentWithTools(registry)

user_request = "Review the code in sample.py and print the review"

result = agent.run(user_input=user_request,max_steps=5)
# We give the agent an observation (user request) to think about and give us a decision

print(f"Result: {result}")

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
        - patch_file(filepath, content)
        - print_review(review: str)

        Decide which tool is most appropriate based on the user input below if a tool is needed.
        If a tool call is needed Reply ONLY with the tool call to make in JSON format {{"tool": "tool_name", "args": ["arg1", "arg2"]}} (e.g., {{"tool":"read_file", "args":["sample.py"]}}
        Examples:
        - read_file("main.py")
        - patch_file(filepath, content)
        - print_review(review: str)

        If the task is complete respond with JSON {{"done: true, "summary:"The task is complete because"}} where the summary is the reason why the task is complete

        user input: {user_input}
        """
        print(f"Calling LLM with:\n{"-"*10}\n{prompt}\n{"-"*20}")
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
        else:
            return "Action cancelled by user"
    
    def run(self, user_input: str, max_steps:int=3):
        original_input = user_input
        for step in range(max_steps):
            print(f"Step: {step+1} of {max_steps}")
            decision = self.think(user_input)
            
            try:
                decision_parsed = json.loads(decision)
            except json.JSONDecodeError as e:
                print(f"Could not parse decision: {e}")
                user_input = f"Your response was not valid JSON.\nOriginal user request: {original_input}"
            
            if decision_parsed.get("done"):
                print(f"Task complete\nAssistant Repose:{decision}")
                return decision_parsed.get("summary")
            
            result = self.act(decision)
            user_input = f"Original user request: {original_input}\nLast assistant response{decision}\nLast tool result: {result}. continue with original user request"

        print("Loop complete. (max steps reached)")
        return result
```


```python
code_snippet = """
def divide(a,b):
    return a/b
"""

agent_hil = CodeReviewAgentWithToolsHIL(tool_registry=registry)
user_request = f"Please review this code {code_snippet}"

result = agent_hil.run(user_request,max_steps=5)

print(f"Result: {result}")
```

**Full Source Code Here:**  [Agent Tools Jupyter Notebook](https://github.com/asanyaga/ai-agents-tutorial/blob/main/part-2-agent-tools.ipynb)

### What's next
In the next part of this series we will give our agent memory.

We will see why memory is an important part of agents and how to use and manage the different types of memory.