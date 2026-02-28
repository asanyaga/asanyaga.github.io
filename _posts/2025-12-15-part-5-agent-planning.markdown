---
title:  "Planning and Task Decomposition in AI Agents"
date:   2025-12-15 10:19:51 +0300
categories: tutorials
tags: agents ai
series: "From Prompts to Agents"
series_order: 5
---

In the previous tutorial we built a code review agent that uses the ReAct pattern to reason about tasks, call tools and manage memory. However, our agent still handles tasks linearly; it takes one step at a time without a comprehensive plan for complex, multi-step workflows.

In this tutorial we will add **planning and task decomposition** capabilities to our agent.

We will teach it to;
1. Break down complex tasks into smaller sub tasks
2. Create and follow execution plans
3. Track progress through multi step workflows
4. Adapt plans based on intermediate results

We will demostrate these concepts by adding testing capabilities to our code review agent, which will require multi step coordination.

Consider this request *Review the code, write tests for it, write tests for it, run and verify the tests*  
This requires:
* Reading the file
* Analyzing and fixing bugs
* Writing appropriate test cases
* Running tests to verify the fix

Our current agent could handle this, but might lose track of what's been done or miss steps. With explicit planning, we can ensure systematic execution.


### Adding Testing Tools
We will be making changes to this code [CodeReviewAgentReAct](https://github.com/asanyaga/ai-agents-tutorial/blob/main/code-review-agent-ReAct.ipynb)

First, let's add two new tools that will enable our testing workflow
```python
def write_test(file_path:str, test_code: str) -> str:
    """Write test code to a test file"""
    try:
        test_dir = os.path.dirname(file_path) or "tests"
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        with open(file_path, "w") as f:
            f.write(test_code)
        return f"Test file created: {file_path}"
    except Exception as e:
        return f"Error writing test file {file_path: {e}}"

def run_test(file_path: str) -> str:
    """Run a Python test file and return results"""
    try:
        import subprocess
        result = subprocess.run(
            ["python","-m","pytest", file_path,"-v"],
            capture_output=True,
            text=True,
            timeout=30
        )
        return f"Exit code {result.returncode}\n\nOuput:\n{result.stdout}\n\nErrors:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "Test execution timed out after 30 seconds"
    except Exception as e:
        return f"Error running tests: {e}"
```


```python
from typing import Callable, Dict
import openai
import os

## Set up the tools and tools registry
def write_test(file_path:str, test_code: str) -> str:
    """Write test code to a test file"""
    try:
        test_dir = os.path.dirname(file_path) or "tests"
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        with open(file_path, "w") as f:
            f.write(test_code)
        return f"Test file created: {file_path}"
    except Exception as e:
        return f"Error writing test file {file_path: {e}}"

def run_test(file_path: str) -> str:
    """Run a Python test file and return results"""
    try:
        import subprocess
        result = subprocess.run(
            ["python","-m","pytest", file_path,"-v"],
            capture_output=True,
            text=True,
            timeout=30
        )
        return f"Exit code {result.returncode}\n\nOuput:\n{result.stdout}\n\nErrors:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "Test execution timed out after 30 seconds"
    except Exception as e:
        return f"Error running tests: {e}"

def read_file(file_path: str) -> str:
    """Read contents of a Python file"""
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"
    
    with open(file_path, "r") as f:
        return f.read()
    
def print_review(review: str):
    print(f"Review: {review}")
    return f"Printed review: {review}"

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

### Implement the planning agent

1. **Add create_plan**
This method asks the LLM to break down the user's request into discrete, actionable steps:
```python
    def create_plan(self, user_query:str) -> list:
        """Generate a step by step plan for the user's request"""
        planning_prompt = f"""
        Given this task:""{user_query}""
        Create a detailed execution plan with numbered steps. Each step should be a specific action

        Available tools:
        - read_file(file): Read a file's contents
        - analyze_code(code): Get code analysis and suggestions
        - patch_file(file_path, content): Update a file
        - write_test(file_path, text_code): Create a test file
        - run_test(file_path): Execute tests

        Format your response as a JSON list of steps
        [
        {{"step":1,"action":"description","tool":"tool_name"}},
        {{"step":1,"action":"description","tool":"tool_name"}}
        ]

        Only include necessary steps. Be specific about which files to work with.
        """

        resposnse = openai.responses.create(model=self.model,
                                            input=[{"role":"user","content":planning_prompt}])
        
        try:
            plan = json.loads(resposnse.output_text)
            self.current_plan = plan
            self.plan_created = True
            return plan
        except json.JSONDecodeError:
            self.current_plan = [{"step":1,"action":"Proceed step by step","tool":"read_file"}]
            self.plan_created= True
            return self.current_plan
```
2. The '_build_lan_context()' method
This method formats the current plan state for inclusion in the system prompt:
```python
    def _build_plan_context(self,next_step) -> str:
    """Format plan information for the prompt"""
    completed = "\n".join([f"Step {step["step"]}:{step["action"]}" for step in self.completed_steps])

    if next_step:
        current = f"\nCURRENT: Step {next_step["step"]}: {next_step["action"]}"
    else:
        current = "\n All steps completed"
    
    remaining = "\n".join([f" Step {step["step"]}: {step["action"]}" for step in self.current_plan[len(self.completed_steps)+1:]])

    execution_plan = f"""
    Completed:
    {completed if completed else "None"}
    {current}
    Remaining:
    {remaining if remaining else "None"}
    """

    return execution_plan
```
3. **Updated** ```build_system_prompt()```
The system prompt now includes plan context and the additional testing tools:
```python
def build_system_prompt(self, plan_context: str = "") -> str:
    """Construct the ReAct system prompt with current context and plan."""
    return f"""You are a code review assistant using the ReAct pattern with planning.

## Available Tools
- read_file(filepath): Read contents of a file
- patch_file(filepath, content): Replace file contents entirely
- write_test(file_path, test_code): Create a test file
- run_test(file_path): Execute tests and return results

## Context
{self.get_relevant_memories()}

Conversation summary: {self.conversation_summary or 'This is the start of the conversation.'}

{plan_context}

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
3. Include "answer" only when ALL plan steps are complete
4. Never include both "action" and "answer"
5. Respond with ONLY valid JSON—no markdown, no extra text
6. Follow the execution plan systematically
7. After each successful action, the system will mark that step complete

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
```

4. **Update** ```run`` to include the plan
```python
    def run(self, user_query: str, max_iterations=10):
        """
        Main execution loop with planning and ReAct pattern.
        
        Args:
            user_query: The user's request
            max_iterations: Maximum number of think-act cycles
        
        Returns:
            Final response string
        """
        current_input = user_query

        for step in range(max_iterations):
            print(f"\n{'-'*60}")
            print(f"Step {step+1} of {max_iterations}")
            print(f"{'-'*60}")

            llm_response = self.think(current_input)
            print(f"\nAgent's LLM Response:\n{llm_response}")

            # Parse the JSON response
            try:
                parsed_response = json.loads(llm_response)
            except json.JSONDecodeError as e:
                print(f"\nJSON Parse Error: {e}")
                current_input = (
                    f"Your response was not valid JSON. Error: {e}\n"
                    f"Respond with ONLY valid JSON matching the required format."
                )
                continue

            # Handle plan creation response
            if parsed_response.get("plan_created"):
                print(f"\nPlan Created:")
                print(parsed_response.get("plan", ""))
                current_input = "Proceed with step 1 of the plan."
                continue

            # Print thought if present
            if "thought" in parsed_response:
                print(f"\nThought: {parsed_response['thought']}")

            # Check for final answer
            if "answer" in parsed_response:
                print(f"\nAnswer: {parsed_response['answer']}")
                print(f"\nProgress: {len(self.completed_steps)}/{len(self.current_plan)} steps completed")
                return parsed_response["answer"]
            
            # Execute action if present
            if "action" in parsed_response:
                action = parsed_response["action"]
                tool_name = action.get("tool", "unknown")
                args = action.get("args", [])

                print(f"\nAction: {tool_name}({', '.join(repr(a) for a in args)})")
                
                observation = self.act(action)
                
                # Truncate long observations for display
                obs_display = observation[:500] + "..." if len(str(observation)) > 500 else observation
                print(f"\nObservation: {obs_display}")
                
                current_input = f"Observation: {observation}"
            else:
                # Neither action nor answer
                print("\nResponse missing both 'action' and 'answer'")
                current_input = (
                    "Your response must include either 'action' (to use a tool) "
                    "or 'answer' (if the task is complete). Please try again."
                )

        print(f"\nMaximum Iterations ({max_iterations}) reached")
        print(f"Progress: {len(self.completed_steps)}/{len(self.current_plan)} steps completed")
        return "Task Incomplete: Max steps reached"
```


```python
import tiktoken
import json

class CodeReviewAgentPlanning:
    def __init__(self, tools_registry: ToolRegistry, model="gpt-4.1", 
                 memory_file="agent_memory.json", summarize_after=10, 
                 max_context_tokens=6000):
        self.tools = tools_registry
        self.model = model
        self.conversation_history = []  # Short-term memory
        self.memory_file = memory_file
        self.load_long_term_memory()  # Long-term memory (key-value store)
        self.conversation_summary = ""  # Summarized conversation history
        self.summarize_after = summarize_after
        self.turns_since_summary = 0
        self.max_context_tokens = max_context_tokens
        
        # Planning-specific attributes
        self.current_plan = []  # List of planned steps
        self.completed_steps = []  # Track what has been done
        self.plan_created = False

        # Initialize tokenizer for the model
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in a string"""
        return len(self.tokenizer.encode(text))
    
    def trim_history_to_fit(self, system_message: str):
        """Remove old messages until we fit within the token budget"""
        fixed_tokens = self.count_tokens(system_message)
        history_tokens = sum([self.count_tokens(msg["content"]) 
                            for msg in self.conversation_history])
        total_tokens = fixed_tokens + history_tokens

        while total_tokens > self.max_context_tokens and len(self.conversation_history) > 2:
            removed_msg = self.conversation_history.pop(0)
            total_tokens -= self.count_tokens(removed_msg["content"])

        return total_tokens

    def summarize_history(self):
        """Use LLM to summarize the conversation so far."""
        if len(self.conversation_history) < 3:
            return
        
        history_text = "\n".join([
            f"{msg['role']}:{msg['content']}" 
            for msg in self.conversation_history
        ])

        summary_prompt = f"""Summarize this conversation in 3-4 sentences,
        preserving key facts, decisions, and actions taken:
        {history_text}

        Previous Summary: {self.conversation_summary or 'None'}
        """

        response = openai.responses.create(
            model=self.model, 
            input=[{"role": "user", "content": summary_prompt}]
        )
        self.conversation_summary = response.output_text

        # Keep only the last few turns + the summary
        recent_turns = self.conversation_history[-4:]
        self.conversation_history = recent_turns
        self.turns_since_summary = 0

    def remember(self, key: str, value: str):
        """Store information in long term memory."""
        self.long_term_memory[key] = value
        self.save_long_term_memory()
    
    def recall(self, key: str) -> str:
        """Retrieve information from long term memory"""
        return self.long_term_memory.get(key, "No memory found for this key.")
    
    def get_relevant_memories(self) -> str:
        """Format long term memories for inclusion in prompts."""
        if not self.long_term_memory:
            return "No stored memories"
        
        memories = "\n".join([f"- {k}: {v}" for k, v in self.long_term_memory.items()])
        return f"Relevant memories:\n{memories}"
    
    def save_long_term_memory(self):
        """Persist long term memory to JSON file"""
        try:
            with open(self.memory_file, "w") as f:
                json.dump(self.long_term_memory, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save memory to {self.memory_file}: {e}")

    def load_long_term_memory(self):
        """Load long term memory from JSON file"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    self.long_term_memory = json.load(f)
                print(f"Loaded {len(self.long_term_memory)} memories from {self.memory_file}")
            except Exception as e:
                print(f"Warning: Could not load memory from {self.memory_file}: {e}")
                self.long_term_memory = {}
        else:
            self.long_term_memory = {}
    
    def create_plan(self, user_query: str) -> list:
        """Generate a step by step plan for the user's request"""
        planning_prompt = f"""
        Given this task: "{user_query}"
        Create a detailed execution plan with numbered steps. 
        Each step should be a specific action.

        Available tools:
        - read_file(filepath): Read a file's contents
        - patch_file(filepath, content): Update a file
        - write_test(file_path, test_code): Create a test file
        - run_test(file_path): Execute tests

        Format your response as a JSON list of steps:
        [
            {{"step": 1, "action": "description", "tool": "tool_name"}},
            {{"step": 2, "action": "description", "tool": "tool_name"}}
        ]

        Only include necessary steps. Be specific about which files to work with.
        Respond with ONLY the JSON array—no markdown, no extra text.
        """

        response = openai.responses.create(
            model=self.model,
            input=[{"role": "user", "content": planning_prompt}]
        )
        
        try:
            plan = json.loads(response.output_text)
            self.current_plan = plan
            self.plan_created = True
            return plan
        except json.JSONDecodeError:
            self.current_plan = [
                {"step": 1, "action": "Proceed step by step", "tool": "read_file"}
            ]
            self.plan_created = True
            return self.current_plan
    
    def _build_plan_context(self, next_step) -> str:
        """Format plan information for the prompt"""
        completed = "\n".join([
            f"  ✓ Step {step['step']}: {step['action']}" 
            for step in self.completed_steps
        ])

        if next_step:
            current = f"\n→ CURRENT: Step {next_step['step']}: {next_step['action']}"
        else:
            current = "\n→ All steps completed"
        
        remaining = "\n".join([
            f"  Step {step['step']}: {step['action']}" 
            for step in self.current_plan[len(self.completed_steps)+1:]
        ])

        return f"""
## Execution Plan Progress

Completed:
{completed if completed else "  None yet"}
{current}

Remaining:
{remaining if remaining else "  None"}
"""

    def build_system_prompt(self, plan_context: str = "") -> str:
        """Construct the ReAct system prompt with current context and plan."""
        return f"""You are a code review assistant using the ReAct pattern with planning.

## Available Tools
- read_file(filepath): Read contents of a file
- patch_file(filepath, content): Replace file contents entirely
- write_test(file_path, test_code): Create a test file
- run_test(file_path): Execute tests and return results

## Context
{self.get_relevant_memories()}

Conversation summary: {self.conversation_summary or 'This is the start of the conversation.'}

{plan_context}

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
3. Include "answer" only when ALL plan steps are complete
4. Never include both "action" and "answer"
5. Respond with ONLY valid JSON—no markdown, no extra text
6. Follow the execution plan systematically
7. After each successful action, the system will mark that step complete

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

    def think(self, user_input: str):
        """LLM decides which tool to use with plan awareness."""
        
        # First request: create a plan
        if not self.plan_created:
            plan = self.create_plan(user_query=user_input)
            plan_summary = "\n".join([
                f"Step {step['step']}: {step['action']}" 
                for step in plan
            ])
            
            # Return a special response indicating the plan was created
            return json.dumps({
                "thought": "I've analyzed the task and created an execution plan.",
                "plan_created": True,
                "plan": plan_summary
            })

        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.turns_since_summary += 1

        # Check if we should summarize
        if self.turns_since_summary >= self.summarize_after:
            self.summarize_history()

        # Get current step from plan
        next_step = None
        if len(self.completed_steps) < len(self.current_plan):
            next_step = self.current_plan[len(self.completed_steps)]
        
        # Build context with plan information
        plan_context = self._build_plan_context(next_step)

        # Include long term memory & summary in system context
        system_message_context = self.build_system_prompt(plan_context)

        self.trim_history_to_fit(system_message_context)
        
        # Build prompt with system instructions
        messages = [
            {"role": "system", "content": system_message_context}
        ] + self.conversation_history

        response = openai.responses.create(model=self.model, input=messages)
        decision = response.output_text

        # Add assistant's decision to conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": decision
        })

        return decision
    
    def act(self, action: dict):
        """Execute the chosen tool and update plan progress."""
        try:
            tool_name = action.get("tool")
            args = action.get("args", [])

            result = self.tools.call(tool_name, *args)

            # Mark current step as complete
            if len(self.completed_steps) < len(self.current_plan):
                current_step = self.current_plan[len(self.completed_steps)]
                self.completed_steps.append(current_step)

            self.conversation_history.append({"role": "system", "content": result})
            return result
        except Exception as e:
            error_msg = f"Error executing tool: {e}"
            self.conversation_history.append({
                "role": "system",
                "content": error_msg
            })
            return error_msg

    def run(self, user_query: str, max_iterations=10):
        """
        Main execution loop with planning and ReAct pattern.
        
        Args:
            user_query: The user's request
            max_iterations: Maximum number of think-act cycles
        
        Returns:
            Final response string
        """
        current_input = user_query

        for step in range(max_iterations):
            print(f"\n{'-'*60}")
            print(f"Step {step+1} of {max_iterations}")
            print(f"{'-'*60}")

            llm_response = self.think(current_input)
            print(f"\nAgent's LLM Response:\n{llm_response}")

            # Parse the JSON response
            try:
                parsed_response = json.loads(llm_response)
            except json.JSONDecodeError as e:
                print(f"\nJSON Parse Error: {e}")
                current_input = (
                    f"Your response was not valid JSON. Error: {e}\n"
                    f"Respond with ONLY valid JSON matching the required format."
                )
                continue

            # Handle plan creation response
            if parsed_response.get("plan_created"):
                print(f"\nPlan Created:")
                print(parsed_response.get("plan", ""))
                current_input = "Proceed with step 1 of the plan."
                continue

            # Print thought if present
            if "thought" in parsed_response:
                print(f"\nThought: {parsed_response['thought']}")

            # Check for final answer
            if "answer" in parsed_response:
                print(f"\nAnswer: {parsed_response['answer']}")
                print(f"\nProgress: {len(self.completed_steps)}/{len(self.current_plan)} steps completed")
                return parsed_response["answer"]
            
            # Execute action if present
            if "action" in parsed_response:
                action = parsed_response["action"]
                tool_name = action.get("tool", "unknown")
                args = action.get("args", [])

                print(f"\nAction: {tool_name}({', '.join(repr(a) for a in args)})")
                
                observation = self.act(action)
                
                # Truncate long observations for display
                obs_display = observation[:500] + "..." if len(str(observation)) > 500 else observation
                print(f"\nObservation: {obs_display}")
                
                current_input = f"Observation: {observation}"
            else:
                # Neither action nor answer
                print("\nResponse missing both 'action' and 'answer'")
                current_input = (
                    "Your response must include either 'action' (to use a tool) "
                    "or 'answer' (if the task is complete). Please try again."
                )

        print(f"\nMaximum Iterations ({max_iterations}) reached")
        print(f"Progress: {len(self.completed_steps)}/{len(self.current_plan)} steps completed")
        return "Task Incomplete: Max steps reached"
```

### Usage
Let's see our planning agent in action


```python
registry = ToolRegistry()
registry.register("read_file",read_file)
registry.register("print_review",print_review)
registry.register("write_test",write_test)
registry.register("patch_file",patch_file)
registry.register("run_test",run_test)

agent = CodeReviewAgentPlanning(tools_registry=registry,model="gpt-4.1",max_context_tokens=8000)

user_query = "Review sample.py, fix any issues, write test and verify issues are fixed"

result = agent.run(user_query)
```

## What's next
In this tutorial we have implemented a simple planning agent that 'plans' about what actions to take based on task it has been given.

In the next part of the series we will look at more advanced patterns such as routing, and multi agent workflows.

We will also start to explore practical considerations for deploying real world agents such as observability, agent evaluation, guardrails and security


**Full Source Code Here** [Planning Agent Jupyter Notebook](https://github.com/asanyaga/ai-agents-tutorial/blob/main/part-5-agent-planning.ipynb)