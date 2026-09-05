# Import the Ollama chat function.
from ollama import chat

# Define the local Ollama model.
MODEL = "llama3.2:3b"

# Define the maximum number of agent iterations.
MAX_ITERATIONS = 5

# Define fake payment data for this learning example.
payment = {
    "transaction_id": "1001",
    "status": "FAILED",
    "failure_reason": "INSUFFICIENT_FUNDS"
}

# Define the customer's goal.
goal = "Investigate why payment transaction 1001 failed."

# Store observations collected by the agent.
observations = []

# Store actions that have already been executed.
completed_actions = []

# Define a function to get the payment status.
def get_payment_status():

    # Return the payment status from the fake payment data.
    return payment["status"]


# Define a function to get the payment failure reason.
def get_failure_reason():

    # Return the failure reason from the fake payment data.
    return payment["failure_reason"]


# Print the program title.
print("\nMULTIPLE AGENT-LOOP ITERATIONS DEMO")

# Print the customer goal.
print("\nGOAL:")
print(goal)

# Start the agent loop.
for iteration in range(1, MAX_ITERATIONS + 1):

    # Print the current iteration number.
    print(f"\nITERATION {iteration}")

    # Create the prompt for the agent.
    prompt = f"""
You are a payment investigation agent.

Goal:
{goal}

Available actions:
GET_PAYMENT_STATUS
GET_FAILURE_REASON
FINISH

Previous observations:
{observations}

Actions already executed:
{completed_actions}

Rules:
- Do not select an action that has already been executed.
- Select GET_PAYMENT_STATUS if payment status is unknown.
- Select GET_FAILURE_REASON if payment status is known but failure reason is unknown.
- Select FINISH if the payment status and failure reason are both known.
- Return only the action name.
"""

    # Ask the LLM to select the next action.
    response = chat(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the selected action from the LLM response.
    action = response["message"]["content"].strip()

    # Print the selected action.
    print(f"Agent Action: {action}")

    # Check whether the agent selected the payment status action.
    if "GET_PAYMENT_STATUS" in action:

        # Check whether this action was already executed.
        if "GET_PAYMENT_STATUS" in completed_actions:

            # Print that the action is a duplicate.
            print("Duplicate action detected.")

            # Continue to the next iteration.
            continue

        # Execute the payment status tool.
        result = get_payment_status()

        # Create an observation from the tool result.
        observation = f"Payment status is {result}."

        # Store the observation.
        observations.append(observation)

        # Store the executed action.
        completed_actions.append("GET_PAYMENT_STATUS")

        # Print the observation.
        print(f"Observation: {observation}")

    # Check whether the agent selected the failure reason action.
    elif "GET_FAILURE_REASON" in action:

        # Check whether this action was already executed.
        if "GET_FAILURE_REASON" in completed_actions:

            # Print that the action is a duplicate.
            print("Duplicate action detected.")

            # Continue to the next iteration.
            continue

        # Execute the failure reason tool.
        result = get_failure_reason()

        # Create an observation from the tool result.
        observation = f"Payment failure reason is {result}."

        # Store the observation.
        observations.append(observation)

        # Store the executed action.
        completed_actions.append("GET_FAILURE_REASON")

        # Print the observation.
        print(f"Observation: {observation}")

    # Check whether the agent selected FINISH.
    elif "FINISH" in action:

        # Print that the agent has enough information.
        print("Agent has enough information.")

        # Stop the agent loop.
        break

    # Handle an unexpected action.
    else:

        # Print that the action is unknown.
        print("Unknown action returned by the LLM.")

        # Stop the agent safely.
        break

# Check whether the required information was collected.
if len(observations) >= 2:

    # Create the final answer prompt.
    final_prompt = f"""
The payment investigation is complete.

Goal:
{goal}

Observations:
{observations}

Give a short final answer explaining why the payment failed.
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

# Handle the situation where the goal was not completed.
else:

    # Print that the investigation could not be completed.
    print("\nINVESTIGATION NOT COMPLETED.")

    # Print the collected observations.
    print(f"Observations collected: {observations}")