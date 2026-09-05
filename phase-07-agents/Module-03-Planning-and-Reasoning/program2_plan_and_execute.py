# Import the Ollama chat function.
from ollama import chat

# Define the local Ollama model.
MODEL = "llama3.2:3b"

# Define the customer's goal.
goal = "Investigate my failed payment and tell me what I should do."

# Define the standard payment investigation plan.
plan = [
    "Get payment transaction details",
    "Check payment status",
    "Check payment failure reason",
    "Check whether the payment can be retried",
    "Recommend the next action"
]

# Print the program title.
print("\nPLAN-AND-EXECUTE DEMO")

# Print the customer's goal.
print("\nGOAL:")
print(goal)

# Print the standard plan.
print("\nSTANDARD PLAN:")

# Loop through every task in the plan.
for number, task in enumerate(plan, start=1):

    # Print the current task number and task name.
    print(f"{number}. {task}")

# Print the execution section.
print("\nEXECUTION:")

# Loop through every task in the standard plan.
for number, task in enumerate(plan, start=1):

    # Create a prompt for the executor.
    prompt = f"""
You are executing a payment investigation task.

Customer goal:
{goal}

Current task:
{task}

Explain briefly what should be done for this task.
Do not create a new task.
Do not change the task.
"""

    # Send the current task to the Ollama model.
    response = chat(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the LLM's execution response.
    result = response["message"]["content"]

    # Print the current task number.
    print(f"\nTask {number}: {task}")

    # Print the execution result.
    print(f"Execution: {result}")

# Print the completion message.
print("\nPLAN EXECUTION COMPLETED")