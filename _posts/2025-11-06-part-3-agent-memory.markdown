---
title:  "Adding Memory to our Agent: Short-Term and Long-Term memory in practice"
date:   2025-11-03 17:19:51 +0300
categories: agents ai
---
## Introduction
In our previous tutorials, we built a code review agent that uses tools to read files and analyze code. But there's a critical limitation:**our agent has no memory between interactions**. Every time we call `think()`, the agent starts fresh, with no knowledge of previous conversations or actions.

Imagine asking the agent to "review the last file I mentioned" or "compare this code to what you saw earlier". Without memory it cant do either. In this article, we'll transform our stateless agent into one that remembers conversations, learns from interactions, and manages its memory efficiently.

We will cover:
* Why memory matters for agents
* Short-term memory
* Long-term memory
* Memory summarization techniques
* Context window management strategies

## Why memory matters
**Memory enables continuity**. Without it agents can't:
* Reference previous questions or answers
* Build on past interactions
* Learn user preferences
* Handle multi-turn workflows (e.g. "read the file, then analyze it, then write tests")
Real world conversations have context. Our agent needs memory to maintain that context and provide intelligent, contextual responses.

## Short term memory: Conversation History
**Short-term memory** stores the recent conversation between user and agent. This is the foundation of a multi-turn dialogue.

### Implementation: Adding a Message Buffer
Let's add a simple conversation history to our previous agent [CodeReveiwAgentWithTools](https://github.com/asanyaga/ai-agents-tutorial/blob/main/code_review_agent_with_tools.ipynb)
1. Inititalize a list for conversation history
```python
class CodeReviewAgentWithSTMemory:
    def __init__(self,tools_registry: ToolRegistry, model="gpt-4o-mini"):
        self.tools = tools_registry
        self.model = model
        self.conversation_history = [] # Short-term memory
```
2. Update `think()` to add user input and LLM responses to `conversation_history` and include the conversation history in the prompt
```python
    def think(self, user_input:str):
        """LLM decides which tool to use with conversation context."""
        # Add user message to history
        self.conversation_history.append({"role":"user","content":user_input})

        # Build prompt with system instructions
        messages = [
            {
                "role":"system",
                "content":"""You are a code assistant with access to these tools:
                - read_file(filepath)
                - analyze_code(code)

                Decide which tool to use based on the conversation.
                Reply ONLY with the tool call to make in JSON format {{"tool": "tool_name", "args": ["arg1", "arg2"]}} (e.g., {{"tool":"read_file", "args":["sample.py"]}}

                Examples:
                read_file("main.py")
                analyze_code("def foo():pass")

                """
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
```
3. Update `act()` to add tool call results to conversation history
```python
    def act(self, decision:str):
        """Execute the chosen tool and record the result."""
        try:
            parsed = json.loads(decision)
            tool_name = parsed["tool"]
            args = parsed.get("args",[])

            result = self.tools.call(tool_name,*args)

            #Store tool call result in conversation history
            self.conversation_history.append({
                "role":"system",
                "content":f"Tool result: {result}"
            })
            return result
        except Exception as e:
            error_msg = f"Error executing tool: {e}"
            self.conversation_history.append({
                "role":"system",
                "content": error_msg
            })
            return error_msg
```
### What changed
* **`conversation_history` list**: Stores all messages as dictionaries with `role` and `content`
* **Messages passed to LLM**: Instead of a single prompt string, we send the entire conversation
* **Tool call result stored**: After each action we append the result to history so the agent can reference it


```python
# Set up the tools and tool registry
import os
import openai
from typing import Dict, Callable

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


```python
import json
class CodeReviewAgentWithSTMemory:
    def __init__(self,tools_registry: ToolRegistry, model="gpt-4o-mini"):
        self.tools = tools_registry
        self.model = model
        self.conversation_history = [] # Short-term memory

    def think(self, user_input:str):
        """LLM decides which tool to use with conversation context."""
        # Add user message to history
        self.conversation_history.append({"role":"user","content":user_input})

        # Build prompt with system instructions
        messages = [
            {
                "role":"system",
                "content":"""You are a code assistant with access to these tools:
                - read_file(filepath)
                - analyze_code(code)

                Decide which tool to use based on the conversation.
                Reply ONLY with the tool call to make in JSON format {"tool": "tool_name", "args": ["arg1", "arg2"]} (e.g., {"tool":"read_file", "args":["sample.py"]}
                
                Examples:
                {"tool":"read_file", "args":["sample.py"]}
                {"tool":"analyze_code", "args":["def foo():pass"]}
                """
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
    
    def act(self, decision:str):
        """Execute the chosen tool and record the result."""
        try:
            parsed = json.loads(decision)
            tool_name = parsed["tool"]
            args = parsed.get("args",[])

            result = self.tools.call(tool_name,*args)

            #Store tool call result in conversation history
            self.conversation_history.append({
                "role":"system",
                "content":f"Tool result: {result}"
            })
            return result
        except Exception as e:
            error_msg = f"Error executing tool: {e}"
            self.conversation_history.append({
                "role":"system",
                "content": error_msg
            })
            return error_msg

```

### Let's give it a try by simulating a multi-turn conversation


```python
registry = ToolRegistry()
registry.register("read_file",read_file)
registry.register("analyze_code",analyze_code)

agent_with_st_memory = CodeReviewAgentWithSTMemory(registry)

# Multi-turn conversation
decision1 = agent_with_st_memory.think("read the file sample.py")
print(f"Agent first decision: {decision1}")

result1 = agent_with_st_memory.act(decision1)
print(f"Agent first action result: {result1}")

#The agent remembers what it read. We don't need to provide it the code
decision2 = agent_with_st_memory.think("Analyze that code")
print(f"Agent second decision: {decision2}")

result2 = agent_with_st_memory.act(decision2)
print(f"Second action result: {result2}")

print(f"Chat history : {agent_with_st_memory.conversation_history}")
```

**Key insight:** The LLM sees the full conversation each time, allowing it to understand context and references like "that code" or "the last file"

## Long-term Memory: Persistent Knowledge
Short-term memory is exists only during a session. Long term memory persists across sessions and stores important information the agent should remember 
every time it is performing a task.

### Use cases for long term memory
* **User preferences**: "I prefer tests with pytest, not unittest"
* **Project context**: "this is a fastapi web api with sqlalchemy models"
* **Learned patterns**: "user often asks for sql injection vulnerabilities"
* **Important facts**: File paths, project structure, common issues

### Implementation: Adding a long term knowledge store
Let's add long term memory to our agent
1. **Add** `long_term_memory` and `memory_file` which we will implement as a simple key value store persisted in a `.json` file
```python
class CodeReviewAgentWithLTMemory:
    def __init__(self,tools_registry: ToolRegistry, model="gpt-4o-mini",memory_file="agent_memory.json"):
        #...rest of init ..
```
2. **Add** `remember()` adds/updates key value in the long term memory
```python
    def remember(self, key:str, value: str):
        """Retrieve information from long term memory."""
        self.long_term_memory[key] = value
        self.save_long_term_memory()
```
3. **Add** `recall()` retrieves a particular item from the long term memory
```python
    def recall(self,key:str) -> str:
        """Retrieve information from long term memory"""
        return self.long_term_memory.get(key,"No memory found for this key.")
```
4. **Add** `get_relevant_memories()` gets and formats the long term memories to include in the system message
```python
    def get_relevant_memories(self) -> str:
        """Format long term memories for inclusion in prompts."""
        if not self.long_term_memory:
            return "No stored memories"
        
        memories = "\n".join([f"- {k}:{v}" for k, v in self.long_term_memory.items()])
        return f"Relevant memories:\n{memories}"
```
5. **Add** `save_long_term_memory()` to persist long term memory to disk. This makes sure it persists between agent sessions
```python
    def save_long_term_memory(self):
        """Persist long term memory to JSON file"""
        try:
            with open(self.memory_file,"w") as f:
                json.dump(self.long_term_memory,f,indent=2)
        except Exception as e:
            print(f"Warning: Could not save memory to {self.memory_file}:  {e}")
```
6. **Add** `load_long_term_memory()` load long term memory when the agent initializes
```python
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
```
7. Update the prompt's system message to include the long term memory as `relevant_memories`
```python
    def think(self, user_input:str):
        """LLM decides which tool to use with both short term and long term context."""
        #...existing code...

        #Include long term memory in system context
        system_message_context = f"""You are a code assistant with access to these tools:
                - read_file(filepath)
                - analyze_code(code)

                {self.get_relevant_memories()}

                Decide which tool to use based on the conversation and relevant memories.
                Reply ONLY with the tool call to make in JSON format {"tool": "tool_name", "args": ["arg1", "arg2"]} (e.g., {"tool":"read_file", "args":["sample.py"]}
                
                Examples:
                {"tool":"read_file", "args":["sample.py"]}
                {"tool":"analyze_code", "args":["def foo():pass"]}

                """
        #... existing code
```

```python
import json
import os
from typing import Dict, Callable
class CodeReviewAgentWithLTMemory:
    def __init__(self,tools_registry: ToolRegistry, model="gpt-4o-mini",memory_file="agent_memory.json"):
        self.tools = tools_registry
        self.model = model
        self.conversation_history = [] # Short-term memory
        self.memory_file = memory_file
        self.load_long_term_memory() # Long-term memory (key-value store)

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

    def think(self, user_input:str):
        """LLM decides which tool to use with both short term and long term context."""
        # Add user message to history
        self.conversation_history.append({"role":"user","content":user_input})

        #Include long term memory in system context
        system_message_context = f"""You are a code assistant with access to these tools:
                - read_file(filepath)
                - analyze_code(code)

                {self.get_relevant_memories()}

                Decide which tool to use based on the conversation and relevant memories.
                Reply ONLY with the tool call to make in JSON format {{"tool": "tool_name", "args": ["arg1", "arg2"]}} (e.g., {{"tool":"read_file", "args":["sample.py"]}}
                
                Examples:
                {{"tool":"read_file", "args":["sample.py"]}}
                {{"tool":"analyze_code", "args":["def foo():pass"]}}

                """

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
    
    def act(self, decision:str):
        """Execute the chosen tool and record the result."""
        try:
            if "(" in decision and ")" in decision:
                name, arg = decision.split("(",1)
                arg = arg.strip(")'\"")
                result = self.tools.call(name.strip(),arg)
            else:
                result = self.tools.call(decision)

            #Store tool call result in conversation history
            self.conversation_history.append({
                "role":"system",
                "content":f"Tool result: {result}"
            })

            return result
        except Exception as e:
            error_msg = f"Error executing tool: {e}"
            self.conversation_history.append({
                "role":"system",
                "content": error_msg
            })
            return error_msg
```

### Demo: Long term memory persists across agent sessions
**Key insight:** Long term memory provides persistent context that informs every interaction, enabling the agent to personalize its behaviour and remember important facts across sessions


```python
# Multi-turn conversation with long term memory
registry = ToolRegistry()
registry.register("read_file",read_file)
registry.register("analyze_code",analyze_code)

agent_with_lt_memory1 = CodeReviewAgentWithLTMemory(registry)
code_snippet = """
def divide(a,b):
    return a/b
"""

agent_with_lt_memory1.remember("documentation","add comprehensive documentation and doc string to ALL code you suggest")
decision1_with_ltm1 = agent_with_lt_memory1.think(f"Analyze this code:{code_snippet}")

print(f"First Agent Long Term Memory {agent_with_lt_memory1.long_term_memory}")
print((f"First Agent Conversion History: {agent_with_lt_memory1.conversation_history}"))


# New session has long term memory
agent_with_lt_memory2 = CodeReviewAgentWithLTMemory(registry)
print(f"Second Agent Long Term Memory {agent_with_lt_memory2.long_term_memory}")
print((f"Second Agent Conversion History: {agent_with_lt_memory2.conversation_history}"))
```

## Memory summarization: Keeping Context Compact
As conversations grow so does the memory footprint. A 50 turn conversation might contain thousands of tokens. Summarization compresses old conversation turns into consise summaries, preserving essential information while reducing token usgae.

### When to summarize
* After a number of conversation turns
* When conversation history exceeds a token threshold
* When moving to a new topic or task

Lte's implement a simple periodic summarization where we use an LLM to generate a summary from the conversation history and trim the conversation to the last few turns.  
1. Add `summarize_after` parameter to agent initialization to set after how many messages to summarize
```python
class CodeReviewAgentWithSTMemorySummarization:
    def __init__(self,tools_registry: ToolRegistry, model="gpt-4o-mini",memory_file="agent_memory.json",summarize_after=10):
        # ...exisiting init code...
        self.conversation_summary = "" # Summarized conversation history
        self.summarize_after = summarize_after # Number of conversation turns after which to summarize
        self.turns_since_summary = 0 # Track number of turns sinse last summary
```
2. Add `conversation_summary` to keep the conversation summary
3. Add `summarize_history()`: Periodically use LLM to summarize conversation history when the `summarize_after` message limit is reached
```python
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

```
4. Include the `conversation_summary` in the system prompt
```python
    def think(self, user_input:str):
        """LLM decides which tool to use with both short term and long term context."""
        # Add user message to history
        self.conversation_history.append({"role":"user","content":user_input})

        self.turns_since_summary += 1

        # Check if we should summarize
        if self.turns_since_summary >= self.summarize_after:
            self.summarize_history()

        #Include long term memory & summary in system context
        system_message_context = f"""You are a code assistant with access to these tools:
                - read_file(filepath)
                - analyze_code(code)

                {self.get_relevant_memories()}

                Conversation Summary: {self.conversation_summary or 'This is the start of the conversation'}

                Decide which tool to use based on the conversation, conversation summary and relevant memories.
                Reply ONLY with the tool name and argument.
                Examples: read_file("main.py") or analyze_code("def foo():pass")

                """
        # ...rest of think code...
```


```python
class CodeReviewAgentWithSTMemorySummarization:
    def __init__(self,tools_registry: ToolRegistry, model="gpt-4o-mini",memory_file="agent_memory.json",summarize_after=10):
        self.tools = tools_registry
        self.model = model
        self.conversation_history = [] # Short-term memory
        self.memory_file = memory_file
        self.load_long_term_memory() # Long-term memory (key-value store)
        self.conversation_summary = "" # Summarized conversation history
        self.summarize_after = summarize_after # Number of conversation turns after which to summarize
        self.turns_since_summary = 0 # Track number of turns sinse last summary

    
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

    def think(self, user_input:str):
        """LLM decides which tool to use with both short term and long term context."""
        # Add user message to history
        self.conversation_history.append({"role":"user","content":user_input})

        self.turns_since_summary += 1

        # Check if we should summarize
        if self.turns_since_summary >= self.summarize_after:
            self.summarize_history()

        #Include long term memory & summary in system context
        system_message_context = f"""You are a code assistant with access to these tools:
                - read_file(filepath)
                - analyze_code(code)

                {self.get_relevant_memories()}

                Conversation Summary: {self.conversation_summary or 'This is the start of the conversation'}

                Decide which tool to use based on the conversation, conversation summary and relevant memories.
                Reply ONLY with the tool name and argument.
                Examples: read_file("main.py") or analyze_code("def foo():pass")

                """

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
    
    def act(self, decision:str):
        """Execute the chosen tool and record the result."""
        try:
            if "(" in decision and ")" in decision:
                name, arg = decision.split("(",1)
                arg = arg.strip(")'\"")
                result = self.tools.call(name.strip(),arg)
            else:
                result = self.tools.call(decision)

            #Store tool call result in conversation history
            self.conversation_history.append({
                "role":"system",
                "content":f"Tool result: {result}"
            })

            return result
        except Exception as e:
            error_msg = f"Error executing tool: {e}"
            self.conversation_history.append({
                "role":"system",
                "content": error_msg
            })
            return error_msg
```

## Context Window Management
Every LLM has a **context window** - a maximum number of tokens it can process at once.  
When conversation history + long term memory + prompt + response exceed this limit, the LLM call my fail or return an incomplete response.  
For this reason we need to manage the context window limits.

### Strategies for Managing the Context Window
1. **Token Counting**: Estimate or count tokens before sending to the LLM
2. **Trimming**: Remove the oldest messages beyond a threshold
3. **Selctive forgetting**: Drop less important messages
4. **Hierarchical Summarization** Sumarize summaries for very long interactions

### Implement Token Aware Trimming
Below is a simple implementation of token aware trimming

1. Token counting. We use `tiktoken` to accurately count tokens
```python
import tiktoken # OpenAI token counting library
```
2. Add `trim_history_to_fit()`: Removes the oldest messages when over budget. This is called every time the agent calls `think()`
```python
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
```
3. Update `think()` to trim history
```python
    def think(self, user_input:str):
        """LLM decides which tool to use with both short term and long term context."""
        # Add user message to history
        self.conversation_history.append({"role":"user","content":user_input})

        self.turns_since_summary += 1

        # Check if we should summarize
        if self.turns_since_summary >= self.summarize_after:
            self.summarize_history()

        #Include long term memory & summary in system context
        system_message_context = f"""You are a code assistant with access to these tools:
                - read_file(filepath)
                - analyze_code(code)

                {self.get_relevant_memories()}

                Conversation Summary: {self.conversation_summary or 'This is the start of the conversation'}

                Decide which tool to use based on the conversation, conversation summary and relevant memories.
                Reply ONLY with the tool name and argument.
                Examples: read_file("main.py") or analyze_code("def foo():pass")

                """

        self.trim_history_to_fit(system_message_context)

        #...existing think code...
```
3. Add `max_context_tokens` to configure token limits
```python
class CodeReviewAgentWithTrimming:
    def __init__(self,tools_registry: ToolRegistry, model="gpt-4o-mini",memory_file="agent_memory.json",summarize_after=10,max_context_tokens=6000):
        # ...existing init code...
        self.max_context_tokens = max_context_tokens
```

## Code review agent with memory and context management


```python
import tiktoken # OpenAI token counting library

class CodeReviewAgentWithContext:
    def __init__(self,tools_registry: ToolRegistry, model="gpt-4o-mini",memory_file="agent_memory.json",summarize_after=10,max_context_tokens=6000):
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

    def think(self, user_input:str):
        """LLM decides which tool to use with both short term and long term context."""
        # Add user message to history
        self.conversation_history.append({"role":"user","content":user_input})

        self.turns_since_summary += 1

        # Check if we should summarize
        if self.turns_since_summary >= self.summarize_after:
            self.summarize_history()

        #Include long term memory & summary in system context
        system_message_context = f"""You are a code assistant with access to these tools:
                - read_file(filepath)
                - analyze_code(code)

                {self.get_relevant_memories()}

                Conversation Summary: {self.conversation_summary or 'This is the start of the conversation'}

                Decide which tool to use based on the conversation, conversation summary and relevant memories.
                Reply ONLY with the tool name and argument.
                Examples: read_file("main.py") or analyze_code("def foo():pass")

                """

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
    
    def act(self, decision:str):
        """Execute the chosen tool and record the result."""
        try:
            if "(" in decision and ")" in decision:
                name, arg = decision.split("(",1)
                arg = arg.strip(")'\"")
                result = self.tools.call(name.strip(),arg)
            else:
                result = self.tools.call(decision)

            #Store tool call result in conversation history
            self.conversation_history.append({
                "role":"system",
                "content":f"Tool result: {result}"
            })

            return result
        except Exception as e:
            error_msg = f"Error executing tool: {e}"
            self.conversation_history.append({
                "role":"system",
                "content": error_msg
            })
            return error_msg
```

### Notes on Memory
* We have shown storing long term memory and retrieving all of it. In practice, with large memory sizes, it may be more efficient to store in a e.g. a vector store or database and use retrieval based on user input to fetch long term memory that is relevant to the agent task.
* In our example we showed conversation history as lasting only for the session. It may be useful for later reference to also persist chat history. This stored conversation history would not be considered part of the agent's long term memory to be used during task sessions.
**Full Source Code Here:**  [Agent Tasks Jupyter Notebook](https://github.com/asanyaga/ai-agents-tutorial/blob/main/part-3-agent-memory.ipynb)
## What's next
That concludes the first part of the series where we implemented the simple building blocks of AI agents.

In the next part of the series we will look at more advanced patterns such as routing, planning and orchestration and multi agent workflows. 

We will also start to dive deeper into the practical considerations for deploying real world agents such as evaluating agents, observability, guardrails and security.
