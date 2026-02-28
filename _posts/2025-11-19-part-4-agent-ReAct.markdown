---
title:  "Agents That Think: Introducing the ReAct Pattern"
date:   2025-11-19 10:35:51 +0300
categories: tutorials
tags: agents ai
series: "From Prompts to Agents"
series_order: 4
---

In our previous tutorials, we established the basic building blocks of an LLM agent. We implemented the **observe**,**think**,**act** loop, we added **tool use** and **memory**.  
Our agent can now read code, suggest improvements and remember past interactions.

However, what happens when the agent makes a mistake, or its suggested action doesn't achieve the desired outcome? This is where **ReAct** comes in. In this tutorial, we'll introduce the **ReAct pattern** and show how to enable our agent to integrate reasoning with tool use, intepret the results of it's actions and adapt it's next steps accordingly, making it a more reliable, flexible and effective assistant.

## The ReAct Pattern: Think, Act, Observe
The **ReAct**(Reasoning and Acting) pattern is a powerful pattern that combines **Reasoning** steps with **Action** steps performed via tools.
* **Thought(Reasoning)**: The LLM internally reasons about the current situation and determines the next **Action** to take.
* **Action**: The agent executes the chosen action (e.g. calling a tool like `read_file(sample.py)`)
* **Observation** The result of the action (tool output) is returned to the agent, serving as the observation for the next turn.

### Why Reasoning is Essential for Agent Success
* **Self-correction**: Reflection allows the agent to recognize when a tool call failed or when the output didn't align with the goal.
* **Plan Adjustment**: The agent can assess the progress of the plan and modify its approach dynamically. For instance if it suggests a fix, the reasoning step can verify that the suggested fix actually addresses the original problem.
* **Increased Robustness**: By incorporating a dedicated step to reason about its outputs, the agent becomes less prone to "hallucination" and its responses are more grounded in real world tool outputs.

### Implementing the ReaAct agent

### The Agent code so far
We will make changes to this agent code [CodeReviewAgentWithContext](https://github.com/asanyaga/ai-agents-tutorial/blob/main/code_review_agent_with_context.ipynb) to demonstrate a simple implementation of the ReAct pattern


### Set up the tools

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

### Implement the ReAct Agent
* Update the `run()` method to manage the observe, reason, act loop
```python
    def run(self, user_query:str, max_iterations=3):
        """
        Main execution loop with reflection.
        Args:
            user_query: The user's request
            max_iterations: Maxumum number of think-act-reflect cycles. this is to avoid the agent getting stuck in a loop.
        
        Returns:
            Final response string
        """
        step = 0

        current_input = user_query

        for step in range(max_iterations):.
            
            print(f"\n{'-'*60}")
            print(f"\nStep {step+1} of {max_iterations}")
            print(f"\n{'-'*60}")

            llm_response = self.think(current_input)

            print(f"Agent's LLM Response:\n{llm_response}")

            try:
                parsed_reponse = json.loads(llm_response)
            except json.JSONDecodeError as e:
                current_input = (
                    f"Your response was not valid Json. Error: {e}\n"
                    f"Respond with ONLY valid JSON matching the required format."
                )
            if "thought" in parsed_reponse:
                print(f"\nThought: {parsed_reponse["thought"]}")

            if "answer" in parsed_reponse:
                print(f"\n Answer: {parsed_reponse["answer"]}")
                return parsed_reponse["answer"]
            
            if "action" in parsed_reponse:
                action = parsed_reponse["action"]
                tool_name = action.get("tool","unknown")
                args = action.get("args", [])

                observation = self.act(action)
                print(f"Action: {tool_name}({','.join(repr(a) for a in args)})")
                current_input = f"Observation: {observation}"
            else:
                # Neither action nor answer
                print("\nResponse missing both 'action' and 'answer'")
                current_input = (
                    "Your response must include either 'action' (to use a tool) "
                    "or 'answer' (if the task is complete). Please try again."
                )

        return "Max steps reached without a final answer"
```
* Add `build_system_prompt()` to generate a ReAct system prompt.
```python
    def build_system_prompt(self) -> str:
        """Construct the ReAct system prompt with current context."""
        return f"""You are a code review assistant using the ReAct pattern.

        ## Available Tools
        - read_file(filepath): Read contents of a file
        - analyze_code(code): Get LLM analysis of code  
        - patch_file(filepath, content): Replace file contents entirely

        ## Context
        {self.get_relevant_memories()}

        Conversation summary: {self.conversation_summary or 'This is the start of the conversation.'}

        ## Response Format

        You MUST respond with valid JSON in one of these two formats:

        ### Format 1: When you need to use a tool
        {{
        "thought": "Your reasoning about what to do and why",
        "action": {{
            "tool": "tool_name",
            "args": ["arg1", "arg2"]
        }}
        }}

        ### Format 2: When the task is complete
        {{
            "thought": "Your reasoning about why the task is complete",
            "answer": "Your final response to the user"
        }}

        ## Rules
        1. Always include "thought" explaining your reasoning
        2. Include "action" when you need to call a tool
        3. Include "answer" only when the task is fully complete
        4. Never include both "action" and "answer"
        5. Respond with ONLY valid JSON—no markdown, no extra text

        ## Example

        User: Review auth.py and fix any bugs

        Response 1:
        {{"thought": "I need to read the file first to see its contents.", "action": {{"tool": "read_file", "args": ["auth.py"]}}}}

        Observation: def check(u): return db.user = u

        Response 2:
        {{"thought": "There's a bug: using = (assignment) instead of == (comparison). I'll fix it.", "action": {{"tool": "patch_file", "args": ["auth.py", "def check(u): return db.user == u"]}}}}

        Observation: File successfully updated: auth.py

        Response 3:
        {{"thought": "The bug is fixed. The comparison operator is now correct.", "answer": "Fixed auth.py: changed assignment operator (=) to comparison operator (==) in the return statement."}}
        """

```
* Update the `system_message_context` prompt in `think()` to implement the ReAct pattern

* The ReAct pattern is implemented by a prompt engineering technique where we give the LLM a crafted promnpt that directs it to reason about past actions and results and respond with the next action.  
Note that we give the LLM a specific output format. The output format is specified to be JSON so that we can have better response handling control.

**NOTE:** As noted earlier in the tools tutorial, most modern LLM have specific tool calling and structured output conventions that would give more predictable structured output.  
In this example, we keep things simple by telling the LLM how to format its response so it can work with most LLMs.


```python
import tiktoken
import json
import openai

class CodeReviewAgentReAct:
    def __init__(self,tools_registry: ToolRegistry, model="gpt-4.1",memory_file="agent_memory.json",summarize_after=10,max_context_tokens=6000):
        self.tools = tools_registry
        self.model = model
        self.conversation_history = [] # Short-term memory
        self.memory_file = memory_file
        self.load_long_term_memory() # Long-term memory (key-value store)
        self.conversation_summary = "" # Summarized conversation history
        self.summarize_after = summarize_after
        self.turns_since_summary = 0
        self.max_context_tokens = max_context_tokens
        

        # Initialize tokenizer for the model
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text:str) -> int:
        """Count tokens in a string"""
        return len(self.tokenizer.encode(text))
    
    def trim_history_to_fit(self, system_message:str):
        """Remove old messages until we fit within the token budget"""

        # Count tokens in system message
        fixed_tokens = self.count_tokens(system_message)

        # Count tokens in conversation history
        history_tokens = sum([self.count_tokens(msg["content"]) for msg in self.conversation_history])

        total_tokens = fixed_tokens + history_tokens

        while total_tokens > self.max_context_tokens and len(self.conversation_history) > 2:
            removed_msg = self.conversation_history.pop(0)
            total_tokens -= self.count_tokens(removed_msg["content"])

        return total_tokens

    def summarize_history(self):
        """Use LLM to summarize the conversation so far."""
        if len(self.conversation_history) < 3:
            return
        
        history_text = "\n".join([f"{msg["role"]}:{msg["content"]}" for msg in self.conversation_history])

        summary_prompt = f"""Summarize this conversation in 3-4 sentences,
        preserving key fact, decisions, and actions taken:
        {history_text}

        Previous Summary: {self.conversation_summary or 'None'}
        """

        response = openai.responses.create(model=self.model, input=[{"role":"user","content":summary_prompt}])

        self.conversation_summary = response.output_text

        # Keep only the last few turns + the summary
        recent_turns = self.conversation_history[-4:] # Keep the last 4 messages (2 user/assistant exchanges)

        self.conversation_history = recent_turns
        self.turns_since_summary = 0


    def remember(self, key:str, value: str):
        """Retrieve information from long term memory."""
        self.long_term_memory[key] = value
        self.save_long_term_memory()
    
    def recall(self,key:str) -> str:
        """Retrieve information from long term memory"""
        return self.long_term_memory.get(key,"No memory found for this key.")
    
    def get_relevant_memories(self) -> str:
        """Format long term memories for inclusion in prompts."""
        if not self.long_term_memory:
            return "No stored memories"
        
        memories = "\n".join([f"- {k}:{v}" for k, v in self.long_term_memory.items()])
        return f"Relevant memories:\n{memories}"
    
    def save_long_term_memory(self):
        """Persist long term memory to JSON file"""
        try:
            with open(self.memory_file,"w") as f:
                json.dump(self.long_term_memory,f,indent=2)
        except Exception as e:
            print(f"Warning: Could not save memory to {self.memory_file}:  {e}")

    def load_long_term_memory(self):
        """Load long term memory from JSON file"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    self.long_term_memory = json.load(f)
                print(f"Loaded {len(self.long_term_memory)} memories from {self.memory_file}")
            except Exception as e:
                print(f"Warning: Could not load memory from {self.memory_file}: {e}")
        else:
            self.long_term_memory = {}

    
    def build_system_prompt(self) -> str:
        """Construct the ReAct system prompt with current context."""
        return f"""You are a code review assistant using the ReAct pattern.

        ## Available Tools
        - read_file(filepath): Read contents of a file
        - analyze_code(code): Get LLM analysis of code  
        - patch_file(filepath, content): Replace file contents entirely

        ## Context
        {self.get_relevant_memories()}

        Conversation summary: {self.conversation_summary or 'This is the start of the conversation.'}

        ## Response Format

        You MUST respond with valid JSON in one of these two formats:

        ### Format 1: When you need to use a tool
        {{
        "thought": "Your reasoning about what to do and why",
        "action": {{
            "tool": "tool_name",
            "args": ["arg1", "arg2"]
        }}
        }}

        ### Format 2: When the task is complete
        {{
            "thought": "Your reasoning about why the task is complete",
            "answer": "Your final response to the user"
        }}

        ## Rules
        1. Always include "thought" explaining your reasoning
        2. Include "action" when you need to call a tool
        3. Include "answer" only when the task is fully complete
        4. Never include both "action" and "answer"
        5. Respond with ONLY valid JSON—no markdown, no extra text

        ## Example

        User: Review auth.py and fix any bugs

        Response 1:
        {{"thought": "I need to read the file first to see its contents.", "action": {{"tool": "read_file", "args": ["auth.py"]}}}}

        Observation: def check(u): return db.user = u

        Response 2:
        {{"thought": "There's a bug: using = (assignment) instead of == (comparison). I'll fix it.", "action": {{"tool": "patch_file", "args": ["auth.py", "def check(u): return db.user == u"]}}}}

        Observation: File successfully updated: auth.py

        Response 3:
        {{"thought": "The bug is fixed. The comparison operator is now correct.", "answer": "Fixed auth.py: changed assignment operator (=) to comparison operator (==) in the return statement."}}
        """

    def think(self, user_input:str):
        """LLM decides which tool to use with both short term and long term context."""
        # Add user message to history
        self.conversation_history.append({"role":"user","content":user_input})

        self.turns_since_summary += 1

        # Check if we should summarize
        if self.turns_since_summary >= self.summarize_after:
            self.summarize_history()

        #Include long term memory & summary in system context
        system_message_context = self.build_system_prompt()

        self.trim_history_to_fit(system_message_context)
        
        # Build prompt with system instructions
        messages = [
            {
                "role":"system",
                "content":system_message_context
            }
        ] + self.conversation_history

        response = openai.responses.create(model=self.model, input=messages)

        decision = response.output_text

        # Add assistant's decision to conversation history
        self.conversation_history.append({
            "role":"assistant",
            "content": decision
        })

        return decision
    
    def act(self, action:str):
        """Execute the chosen tool and return the result."""
        try:
            
            tool_name = action.get("tool")
            args = action.get("args",[])

            result = self.tools.call(tool_name,*args)
            self.conversation_history.append({"role":"system","content":result})
            return result
        except Exception as e:
            error_msg = f"Error executing tool: {e}"
            self.conversation_history.append({
                "role":"system",
                "content": error_msg
            })
            
            return error_msg

    def run(self, user_query:str, max_iterations=3):
        """
        Main execution loop with reflection.
        Args:
            user_query: The user's request
            max_iterations: Maxumum number of think-act-reflect cycles. this is to avoid the agent getting stuck in a loop.
        
        Returns:
            Final response string
        """
        step = 0

        current_input = user_query

        for step in range(max_iterations):
            
            print(f"\n{'-'*60}")
            print(f"\nStep {step+1} of {max_iterations}")
            print(f"\n{'-'*60}")

            llm_response = self.think(current_input)

            print(f"Agent's LLM Response:\n{llm_response}")

            try:
                parsed_reponse = json.loads(llm_response)
            except json.JSONDecodeError as e:
                current_input = (
                    f"Your response was not valid Json. Error: {e}\n"
                    f"Respond with ONLY valid JSON matching the required format."
                )
            if "thought" in parsed_reponse:
                print(f"\nThought: {parsed_reponse["thought"]}")

            if "answer" in parsed_reponse:
                print(f"\n Answer: {parsed_reponse["answer"]}")
                return parsed_reponse["answer"]
            
            if "action" in parsed_reponse:
                action = parsed_reponse["action"]
                tool_name = action.get("tool","unknown")
                args = action.get("args", [])

                observation = self.act(action)
                print(f"Action: {tool_name}({','.join(repr(a) for a in args)})")
                current_input = f"Observation: {observation}"
            else:
                # Neither action nor answer
                print("\nResponse missing both 'action' and 'answer'")
                current_input = (
                    "Your response must include either 'action' (to use a tool) "
                    "or 'answer' (if the task is complete). Please try again."
                )

        return "Max steps reached without a final answer"
```

### Run the agent
Let's run the agent and give it a more complex task to complete.


```python
tool_registry = ToolRegistry()

# Register the tools we defined above
tool_registry.register("read_file", read_file)
tool_registry.register("print_review",print_review)
tool_registry.register("patch_file",patch_file)

agent = CodeReviewAgentReAct(tools_registry=tool_registry)

agent.run(user_query="Review the code in sample.py and fix any issues you find")
```

## What's next
In this tutorial we have implemented a simple ReAct agent that 'thinks' about what actions to take based on the input that it's been given.

In the next part of the series we will look at more advanced patterns such as routing, planning, and multi agent workflows. 

**Full Source Code Here:**  [ReAct Agent Jupyter Notebook](https://github.com/asanyaga/ai-agents-tutorial/blob/main/part-4-agent-ReAct.ipynb)
