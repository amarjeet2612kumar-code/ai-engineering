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

# Store the payment status.
payment_status = None

# Store the payment failure reason.
failure_reason = None

# Store whether the Payment API failed.
api_failed = False

# Store whether the investigation is complete.
completed = False


# Define a tool to get payment status.
def get_payment_status():

    # Return the payment status.
    return payment["status"]


# Define a tool that simulates a failed Payment API.
def get_failure_reason_from_api():

    # Return an error to simulate API failure.
    return "ERROR: Payment API is unavailable."


# Define a database tool to get the failure reason.
def get_failure_reason_from_database():

    # Return the failure reason from the database.
    return payment["failure_reason"]


# Print the program title.
print("\nREPLANNING DEMO")

# Print the customer's goal.
print("\nGOAL:")
print(goal)

# Define the original plan.
original_plan = [
    "Get payment status",
    "Get failure reason from Payment API",
    "Recommend the next action"
]

# Print the original plan.
print("\nINITIAL PLAN:")

# Print each task in the original plan.
for number, task in enumerate(original_plan, start=1):

    # Display the task number and task.
    print(f"{number}. {task}")


# Start the agent loop.
for iteration in range(1, MAX_ITERATIONS + 1):

    # Print the current iteration.
    print(f"\nITERATION {iteration}")

    # Get payment status when it is not known.
    if payment_status is None:

        # Execute the payment status tool.
        payment_status = get_payment_status()

        # Print the tool result.
        print(f"Action: GET_PAYMENT_STATUS")

        # Print the observation.
        print(f"Observation: Payment status is {payment_status}.")

        # Continue to the next iteration.
        continue

    # Try the original Payment API when it has not failed yet.
    if failure_reason is None and not api_failed:

        # Execute the Payment API tool.
        api_result = get_failure_reason_from_api()

        # Print the tool action.
        print("Action: GET_FAILURE_REASON_FROM_API")

        # Print the API observation.
        print(f"Observation: {api_result}")

        # Check whether the Payment API failed.
        if "ERROR" in api_result:

            # Record that the API failed.
            api_failed = True

            # Print the replanning message.
            print("\nREPLANNING:")

            # Create a prompt for choosing an alternative action.
            replan_prompt = """
The original Payment API is unavailable.

The goal is to find the payment failure reason.

Available alternative:
GET_FAILURE_REASON_FROM_DATABASE

Choose the alternative action.

Return only:
GET_FAILURE_REASON_FROM_DATABASE
"""

            # Ask the LLM to select the alternative.
            replan_response = chat(
                model=MODEL,
                messages=[
                    {"role": "user", "content": replan_prompt}
                ]
            )

            # Extract the new action.
            new_action = replan_response["message"]["content"].strip()

            # Print the new action.
            print(f"New Plan Action: {new_action}")

        # Continue to the next iteration.
        continue

    # Use the database after the Payment API failed.
    if failure_reason is None and api_failed:

        # Execute the database tool.
        failure_reason = get_failure_reason_from_database()

        # Print the tool action.
        print("Action: GET_FAILURE_REASON_FROM_DATABASE")

        # Print the observation.
        print(f"Observation: Payment failure reason is {failure_reason}.")

        # Mark the investigation as complete.
        completed = True

        # Stop the agent loop.
        break


# Check whether the investigation completed.
if completed:

    # Print that the investigation was completed.
    print("\nINVESTIGATION COMPLETED.")

    # Create the final answer.
    final_answer = (
        f"Payment transaction {payment['transaction_id']} "
        f"failed because of {failure_reason}."
    )

    # Print the final answer.
    print("\nFINAL ANSWER:")
    print(final_answer)

# Handle an incomplete investigation.
else:

    # Print that the investigation could not be completed.
    print("\nINVESTIGATION NOT COMPLETED.")