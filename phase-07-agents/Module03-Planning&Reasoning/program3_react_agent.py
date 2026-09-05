# Import the Ollama chat function.
from ollama import chat

# Define the local Ollama model.
MODEL = "llama3.2:3b"

# Define fake payment data for our learning example.
payment = {
    "transaction_id": "1001",
    "status": "FAILED",
    "failure_reason": "INSUFFICIENT_FUNDS"
}

# Define a tool that gets the payment status.
def get_payment_status(transaction_id):

    # Return the payment status from our fake payment data.
    return payment["status"]


# Define a tool that gets the payment failure reason.
def get_failure_reason(transaction_id):

    # Return the payment failure reason from our fake payment data.
    return payment["failure_reason"]


# Define the customer's goal.
goal = "Investigate why payment transaction 1001 failed."

# Store the observations collected by the agent.
observations = []

# Print the program title.
print("\nREACT AGENT DEMO")

# Print the customer's goal.
print("\nGOAL:")
print(goal)

# Start the ReAct loop.
for iteration in range(1, 5):

    # Create the prompt containing the current goal and observations.
    prompt = f"""
You are a payment investigation agent.

Goal:
{goal}

Available tools:
1. GET_PAYMENT_STATUS
2. GET_FAILURE_REASON
3. FINISH

Previous observations:
{observations}

Choose exactly one action.

Return only one of these:
GET_PAYMENT_STATUS
GET_FAILURE_REASON
FINISH
"""

    # Ask the LLM to decide the next action.
    response = chat(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the LLM decision.
    action = response["message"]["content"].strip()

    # Print the current iteration.
    print(f"\nITERATION {iteration}")

    # Print the agent's selected action.
    print(f"Agent Action: {action}")

    # Check whether the agent wants to get the payment status.
    if "GET_PAYMENT_STATUS" in action:

        # Execute the payment status tool.
        result = get_payment_status(payment["transaction_id"])

        # Create an observation from the tool result.
        observation = f"Payment status is {result}."

        # Store the observation for the next reasoning step.
        observations.append(observation)

        # Print the observation.
        print(f"Observation: {observation}")

    # Check whether the agent wants to get the failure reason.
    elif "GET_FAILURE_REASON" in action:

        # Execute the failure reason tool.
        result = get_failure_reason(payment["transaction_id"])

        # Create an observation from the tool result.
        observation = f"Payment failure reason is {result}."

        # Store the observation for the next reasoning step.
        observations.append(observation)

        # Print the observation.
        print(f"Observation: {observation}")

    # Check whether the agent has enough information.
    elif "FINISH" in action:

        # Print that the agent has finished its investigation.
        print("Agent decided that the investigation is complete.")

        # Stop the ReAct loop.
        break

    # Handle an unexpected LLM response.
    else:

        # Print that the action was not recognized.
        print("Unknown action returned by the LLM.")

        # Stop the loop safely.
        break

# Create a prompt to generate the final answer.
final_prompt = f"""
You investigated this payment:

Goal:
{goal}

Observations:
{observations}

Give a short final answer explaining:
1. Whether the payment failed.
2. Why it failed.
3. What the customer should do next.
"""

# Ask the LLM to generate the final answer.
final_response = chat(
    model=MODEL,
    messages=[
        {"role": "user", "content": final_prompt}
    ]
)

# Extract the final answer.
final_answer = final_response["message"]["content"]

# Print the final answer.
print("\nFINAL ANSWER:")
print(final_answer)