---
title:  "Agents That Think: Introducing the ReAct Pattern"
date:   2025-11-19 10:35:51 +0300
categories: agents ai
---

In our previous tutorials, we established the basic building blocks of an LLM agent. We implemented the **observe**,**think**,**act** loop, we added **tool use** and **memory**.  
Our agent can now read code, suggest improvements and remember past interactions.

However, what happens when the agent makes a mistake, or its suggested action doesn't achieve the desired outcome? This is where **reflection** comes in. In this tutorial, we'll introduce the **ReAct pattern** and show how to enable our agent to evaluate its own actions, identify errors, and self correct making it a more reliable and autonomous assistant.

## The ReAct Pattern: Think, Act, Observe
The **ReAct**(Reasoning and Acting) pattern is a powerful paradigm that combines **Reasoning** steps with **Action** steps performed via tools.
* **Thought(Reasoning)**: The LLM internally reasons about the current situation and determines the next **Action** to take.
* **Action**: The agent executes the chosen action (e.g. calling a tool like `read_file(sample.py)`)
* **Observation** The result of the action (tool output) is returned to the agent, serving as the observation for the next turn.

Our current agent already follows the Observe, Think, Act sequence. The **reflection pattern** a layer to the **thought** step, allowing the agent to evaluate its past steps and observations before proceeding

### Why Reflection is Essential for Agent Success
* **Self-correction**: Reflection allows the agent to recognize when a tool call failed or when the output didn't align with the goal.
* **Plan Adjustment**: The agent can assess the progress of the plan and modify its approach dynamically. For instance if `analyze_code` suggests a fix, the reflection step can verify that the suggested fix actually addresses the original problem.
* **Increased Robustness**: By incorporating a dedicated step to evaluate its outputs, the agent becomes less prone to "hallucination" and its responses are more grounded in real world tool outputs.


### Implementing the ReaAct agent
We are going to add a new tool for the agent to be able to apply any fixes to code that it suggests

```python
import os
import json

def patch_file(file_path:str, content: str) -> str:
    """Writes the given content to a file, completely replacing its current content"""
    try:
        with open(file_path,"w") as f:
            f.write(content)
        return f"File succesfully updated: {file_path}"
    except Exception as e:
        return f"Error writing to file {file_path}: {e}"

```


### The Agent code so far
We will make changes to this agent code to demonstrate a simple implementation of the ReAct pattern

```python
import tiktoken # OpenAI token counting library

class CodeReviewAgent:
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

### Set up the tools


```python
from typing import Callable, Dict
import openai
import os


## Set up the tools and tools registry

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

def patch_file(filepath: str, content: str) -> str:
    """Writes the given content to a file, completely replacing its current content."""
    try:
        with open(filepath, "w") as f:
            f.write(content)
        return f"File successfully updated: {filepath}. New content written."
    except Exception as e:
        return f"Error writing to file {filepath}: {e}"
        
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
* Add a `run()` method to manage the think, reflect, act loop
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

        while step < max_iterations:
            print(f"\n--- Step {step+1} ---")

            llm_response = self.think(current_input)

            print(f"Agent's LLM Response:\n{llm_response}")

            if "Answer:" in llm_response:
                final_answer = llm_response.split("Answer:",1)[1].strip()
                print(f"\n Agent Finished: \n {final_answer}")
                return final_answer
            if "Action:" in llm_response:
                action_line = llm_response.split("Action:",1)[1].split("\n")[0].strip()
                print(f"Acting: {action_line}")

                tool_result = self.act(action_line)

                print(f"\nTool Result:\n{tool_result}")
                current_input = f"Observation:{tool_result}"
            else:
                error_msg = f"LLM did not provide valid Action or Answer: LLM Respose:: {llm_response}"
                print(f"\n Error: {error_msg}")
                return error_msg
            
            step +=1
        
        return "Max steps reached without a final answer"
```
* Update the `system_message_context` prompt in `think()` to implement the ReAct pattern
* The ReAct pattern is implemented by a prompt engineering technique where we give the LLM a crafted promnpt that directs it to reflect on past actions and respond with the next action.  
Note that we give the LLM a specific output format. Note that the output format for the **Action** is specified to be JSON so that we can have better tool calling control.


**NOTE:** As noted earlier in the tools tutorial, most modern LLM have specific tool calling and structured output conventions that would give more predictable structured output.  
In this example, we keep things simple by telling the LLM how to format its response so it can work with most LLMs.


```python
import tiktoken
import json

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
                - patch_file(filepath,content)

                {self.get_relevant_memories()}

                Conversation Summary: {self.conversation_summary or 'This is the start of the conversation'}

                Decide which tool to use based on the conversation, conversation summary and relevant memories.

                Follow the ReAct pattern: **Thought**, then **Action** or a final **Answer**
                **Format your response STRICTLY as follows:**

                1. Thought:Your internal reasoning and plan.
                2. Action:The tool call to make in JSON format {{"tool": "tool_name", "args": ["arg1", "arg2"]}} (e.g., {{"tool":"patch_file", "args":["file_path","content"]}}. **OR**
                3. Answer:Your final human-readable response.

                After each action you will receive an observation which is a result from the tool call
                DO NOT include any previous thoughts or observations from the conversation history
                DO NOT include any observation

                Reply only with 
                Thought:reasoning or plan
                Action:tool_call
             
                Only provide an Answer: if you have completed the task based on a tool call result
                If based on a tool call result you have completed the task respond with ONLY
                Thought:reasoning
                Answer:why based on the latest observation the task is complete

                Example exchange
                initial user query
                User: Review and fix the code in auth.py

                your response
                Thought: I need to use the tool read_file(auth.py) to review its code
                Action:{{"tool":"read_file","args":["auth.py"]}}

                response from tool call
                Observation: return user_store.username = user_name

                your response
                Thought:I have found the issue. the code is using an assignment instead of comparison so the code will not work. I will use the tool patch_file to correct the code.
                Action: {{"tool":"patch_file","args":["auth.py","return user_store.username == user_name"]}}

                response from tool call
                Observation: successfully applied patch to file auth.py

                your response
                Thought: The code in auth.py has been updated. I will now read the updated auth.py to confirm
                Action:{{"tool":"read_file","args":["auth.py"]}}

                response from tool call
                Observation: "return user_store.username == user_name"

                your response
                Thought: The auth.py file now has the updated code. This task is complete
                Answer: Task is complete because the auth.py file now has the correct code

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
            parsed = json.loads(decision)
            tool_name = parsed["tool"]
            args = parsed.get("args",[])

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

        while step < max_iterations:
            print(f"\n--- Step {step+1} ---")

            llm_response = self.think(current_input)

            print(f"Agent's LLM Response:\n{llm_response}")

            if "Answer:" in llm_response:
                final_answer = llm_response.split("Answer:",1)[1].strip()
                print(f"\n Agent Finished: \n {final_answer}")
                return final_answer
            if "Action:" in llm_response:
                action_line = llm_response.split("Action:",1)[1].split("\n")[0].strip()
                print(f"Acting: {action_line}")

                tool_result = self.act(action_line)

                print(f"\nTool Result:\n{tool_result}")
                current_input = f"Observation:{tool_result}"
            else:
                error_msg = f"LLM did not provide valid Action or Answer: LLM Respose:: {llm_response}"
                print(f"\n Error: {error_msg}")
                return error_msg
            
            step +=1
        
        return "Max steps reached without a final answer"
```

### Run the agent
Let's run the agent and give it a more complex task to complete.


```python
tool_registry = ToolRegistry()

# Register the tools we defined above
tool_registry.register("read_file", read_file)
tool_registry.register("analyze_code",analyze_code)
tool_registry.register("patch_file",patch_file)

agent = CodeReviewAgentReAct(tools_registry=tool_registry)

agent.run(user_query="Review the code in sample.py and fix any issues you find")
```

**Full Source Code Here:**  [ReAct Agent Jupyter Notebook](https://github.com/asanyaga/ai-agents-tutorial/blob/main/part-4-agent-ReAct.ipynb)

## What's next
In this tutorial we have implemented a simple ReAct agent that 'thinks' about what actions to take based on the input that it's been given.

In the next part of the series we will look at more advanced patterns such as routing, planning, orchestration and multi agent workflows. 
