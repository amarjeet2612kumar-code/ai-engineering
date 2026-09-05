# Import the Ollama chat function.
from ollama import chat

# Define the local Ollama model.
MODEL = "llama3.2:3b"

# Define the maximum number of agent iterations.
MAX_ITERATIONS = 6

# Define fake payment data for this learning example.
payment = {
    "transaction_id": "1001",
    "status": "FAILED",
    "failure_reason": "INSUFFICIENT_FUNDS"
}

# Define the customer's goal.
goal = "Investigate why payment transaction 1001 failed and tell me what should be done."

# Store the tasks created during task decomposition.
tasks = []

# Store information discovered by the agent.
observations = []

# Store the payment status.
payment_status = None

# Store the payment failure reason.
failure_reason = None

# Store whether the Payment API failed.
api_failed = False

# Store whether the agent completed the investigation.
completed = False


# Define the tool for getting payment status.
def get_payment_status():

    # Return the payment status.
    return payment["status"]


# Define the tool for getting the failure reason from the API.
def get_failure_reason_from_api():

    # Simulate an unavailable Payment API.
    return "ERROR: Payment API is unavailable."


# Define the tool for getting the failure reason from the database.
def get_failure_reason_from_database():

    # Return the failure reason from the database.
    return payment["failure_reason"]


# Print the program title.
print("\nCOMPLETE PLANNING AND REASONING AGENT")

# Print the customer's goal.
print("\nGOAL:")
print(goal)


# Create the task decomposition prompt.
decomposition_prompt = f"""
You are a task planner.

Break this goal into exactly 3 simple tasks:

{goal}

The investigation should:
1. Check the payment status.
2. Find the payment failure reason.
3. Determine the next action.

Return only a numbered list.
"""

# Ask the LLM to decompose the goal.
decomposition_response = chat(
    model=MODEL,
    messages=[
        {"role": "user", "content": decomposition_prompt}
    ]
)

# Extract the tasks created by the LLM.
task_text = decomposition_response["message"]["content"]

# Split the LLM response into separate lines.
tasks = task_text.splitlines()

# Print the task decomposition section.
print("\nTASK DECOMPOSITION:")

# Print each task created by the planner.
for task in tasks:

    # Print the task.
    print(task)


# Start the agent loop.
for iteration in range(1, MAX_ITERATIONS + 1):

    # Print the current iteration.
    print(f"\nITERATION {iteration}")

    # Check whether the payment status is unknown.
    if payment_status is None:

        # Execute the payment status tool.
        payment_status = get_payment_status()

        # Create the observation.
        observation = f"Payment status is {payment_status}."

        # Store the observation.
        observations.append(observation)

        # Print the selected action.
        print("Action: GET_PAYMENT_STATUS")

        # Print the observation.
        print(f"Observation: {observation}")

        # Continue the agent loop.
        continue

    # Check whether the failure reason is unknown and the API has not failed.
    if failure_reason is None and not api_failed:

        # Execute the Payment API tool.
        api_result = get_failure_reason_from_api()

        # Print the selected action.
        print("Action: GET_FAILURE_REASON_FROM_API")

        # Print the observation.
        print(f"Observation: {api_result}")

        # Store the API observation.
        observations.append(api_result)

        # Check whether the Payment API failed.
        if "ERROR" in api_result:

            # Record that the API failed.
            api_failed = True

            # Print the replanning section.
            print("\nREPLANNING")

            # Create the replanning prompt.
            replan_prompt = """
The Payment API is unavailable.

The investigation still needs the payment failure reason.

Available alternative action:
GET_FAILURE_REASON_FROM_DATABASE

Choose the alternative action.

Return only the action name.
"""

            # Ask the LLM to choose the alternative.
            replan_response = chat(
                model=MODEL,
                messages=[
                    {"role": "user", "content": replan_prompt}
                ]
            )

            # Extract the replanned action.
            new_action = replan_response["message"]["content"].strip()

            # Print the replanned action.
            print(f"New Plan Action: {new_action}")

        # Continue the agent loop.
        continue

    # Check whether the database should now be used.
    if failure_reason is None and api_failed:

        # Execute the database tool.
        failure_reason = get_failure_reason_from_database()

        # Create the database observation.
        observation = f"Payment failure reason is {failure_reason}."

        # Store the observation.
        observations.append(observation)

        # Print the selected action.
        print("Action: GET_FAILURE_REASON_FROM_DATABASE")

        # Print the observation.
        print(f"Observation: {observation}")

        # Mark the investigation as completed.
        completed = True

        # Stop the agent loop.
        break


# Print the reflection section.
print("\nREFLECTION")

# Check whether enough information was collected.
if payment_status is not None and failure_reason is not None:

    # Print that the result is sufficient.
    print("Reflection: The investigation has enough information.")

# Handle incomplete information.
else:

    # Print that the result is incomplete.
    print("Reflection: More information is required.")


# Generate the final answer when the investigation is complete.
if completed:

    # Create the final answer prompt.
    final_prompt = f"""
Answer the customer's question using only the investigation results.

Customer goal:
{goal}

Investigation results:
{observations}

Give a short answer explaining:
1. Why the payment failed.
2. What the customer should do next.
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

# Handle an incomplete investigation.
else:

    # Print that the investigation could not be completed.
    print("\nINVESTIGATION NOT COMPLETED.")