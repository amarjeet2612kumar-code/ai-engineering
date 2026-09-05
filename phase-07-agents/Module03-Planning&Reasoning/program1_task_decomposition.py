# Import the Ollama chat function.
from ollama import chat

# Define the customer's large goal.
goal = "My payment failed. Please investigate the problem and tell me what I should do."

# Ask the LLM to break the large goal into smaller tasks.
prompt = f"""
Break this goal into small tasks:

{goal}

Return a numbered list.
"""

# Send the planning request to the local LLM.
response = chat(
    model="llama3.2:3b",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

# Get the tasks created by the LLM.
tasks = response["message"]["content"]

# Display the decomposed tasks.
print(tasks)