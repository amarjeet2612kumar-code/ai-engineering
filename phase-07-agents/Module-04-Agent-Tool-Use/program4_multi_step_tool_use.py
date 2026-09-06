import ollama  # Import the Ollama library to communicate with the local LLM.


def get_payment_status(transaction_id: str) -> str:  # Define a tool that checks payment status.
    return f"Payment {transaction_id} is FAILED."  # Return the payment status.


def get_failure_reason(transaction_id: str) -> str:  # Define a tool that checks the payment failure reason.
    return f"Payment {transaction_id} failed because of INSUFFICIENT_FUNDS."  # Return the actual failure reason.


def get_customer_details(transaction_id: str) -> str:  # Define a tool that finds the customer linked to the payment.
    return f"Payment {transaction_id} belongs to customer 2001."  # Return the customer information.


tools = [
    {
        "type": "function",  # Tell Ollama that this is a callable function.
        "function": {
            "name": "get_payment_status",  # Give the tool a unique name.
            "description": "Check the status of a payment transaction.",  # Explain when this tool should be used.
            "parameters": {
                "type": "object",  # Define the tool input as an object.
                "properties": {
                    "transaction_id": {
                        "type": "string",  # Define the transaction ID as a string.
                        "description": "The payment transaction ID.",  # Explain the transaction ID argument.
                    }
                },
                "required": ["transaction_id"],  # Require the transaction ID.
            },
        },
    },
    {
        "type": "function",  # Tell Ollama that this is a callable function.
        "function": {
            "name": "get_failure_reason",  # Give the tool a unique name.
            "description": "Find the reason why a payment failed.",  # Explain when this tool should be used.
            "parameters": {
                "type": "object",  # Define the tool input as an object.
                "properties": {
                    "transaction_id": {
                        "type": "string",  # Define the transaction ID as a string.
                        "description": "The payment transaction ID.",  # Explain the transaction ID argument.
                    }
                },
                "required": ["transaction_id"],  # Require the transaction ID.
            },
        },
    },
    {
        "type": "function",  # Tell Ollama that this is a callable function.
        "function": {
            "name": "get_customer_details",  # Give the tool a unique name.
            "description": "Find the customer associated with a payment transaction.",  # Explain when this tool should be used.
            "parameters": {
                "type": "object",  # Define the tool input as an object.
                "properties": {
                    "transaction_id": {
                        "type": "string",  # Define the transaction ID as a string.
                        "description": "The payment transaction ID.",  # Explain the transaction ID argument.
                    }
                },
                "required": ["transaction_id"],  # Require the transaction ID.
            },
        },
    },
]


tool_functions = {
    "get_payment_status": get_payment_status,  # Map the payment status tool name to its Python function.
    "get_failure_reason": get_failure_reason,  # Map the failure reason tool name to its Python function.
    "get_customer_details": get_customer_details,  # Map the customer details tool name to its Python function.
}


user_request = "Investigate payment transaction 1001 and give me a complete report."  # Define the user's request.


messages = [
    {
        "role": "user",  # Identify this message as the user's request.
        "content": user_request,  # Store the user's request.
    }
]


state = {
    "payment_status": None,  # Store the payment status once it is retrieved.
    "failure_reason": None,  # Store the failure reason once it is retrieved.
    "customer_details": None,  # Store the customer information once it is retrieved.
}


MAX_ITERATIONS = 5  # Limit the number of agent iterations for safety.


for iteration in range(MAX_ITERATIONS):  # Continue the agent loop until all required information is collected.

    print(f"\nIteration {iteration + 1}")  # Display the current iteration number.

    missing_information = []  # Create a list to track information that has not been collected yet.

    if state["payment_status"] is None:  # Check whether payment status is missing.
        missing_information.append("payment status")  # Add payment status to the missing information list.

    if state["failure_reason"] is None:  # Check whether failure reason is missing.
        missing_information.append("failure reason")  # Add failure reason to the missing information list.

    if state["customer_details"] is None:  # Check whether customer details are missing.
        missing_information.append("customer details")  # Add customer details to the missing information list.

    if not missing_information:  # Check whether all required information has been collected.
        print("All required information has been collected.")  # Display that the investigation is complete.
        break  # Stop the agent loop.

    state_summary = f"""  # Build a summary of the information currently known by the agent.
Payment Status: {state["payment_status"]}  # Include the current payment status.
Failure Reason: {state["failure_reason"]}  # Include the current failure reason.
Customer Details: {state["customer_details"]}  # Include the current customer information.
Missing Information: {", ".join(missing_information)}  # Tell the LLM what information is still required.
"""  # Finish the state summary.

    decision_prompt = f"""  # Create the prompt for the next tool decision.
You are investigating a payment transaction.  # Define the LLM's role.

The required information for a complete report is:  # Define the completion requirements.
1. Payment status  # Require payment status.
2. Failure reason  # Require failure reason.
3. Customer details  # Require customer information.

Current state:  # Introduce the current agent state.
{state_summary}  # Provide the current state to the LLM.

Choose one tool that can provide missing information.  # Ask the LLM to select the next useful tool.
Do not choose a tool whose information is already available.  # Prevent unnecessary repeated tool calls.
"""  # Finish the decision prompt.

    messages.append(
        {
            "role": "user",  # Add the current state as an instruction to the LLM.
            "content": decision_prompt,  # Send the current state and decision request.
        }
    )

    response = ollama.chat(
        model="llama3.2:3b",  # Use the local llama3.2:3b model.
        messages=messages,  # Send the complete conversation to the model.
        tools=tools,  # Give the model access to all investigation tools.
    )

    assistant_message = response["message"]  # Extract the model's response message.

    messages.append(assistant_message)  # Save the model response in the conversation.

    tool_calls = assistant_message.get("tool_calls", [])  # Extract any tool call selected by the model.

    if not tool_calls:  # Check whether the model failed to select a tool.
        print("Agent did not select a tool.")  # Display the problem.
        break  # Stop the loop because we cannot continue the investigation.

    tool_call = tool_calls[0]  # Get the first selected tool call.

    tool_name = tool_call["function"]["name"]  # Extract the selected tool name.

    arguments = tool_call["function"]["arguments"]  # Extract the arguments selected by the model.

    print("Selected Tool:", tool_name)  # Display the selected tool.
    print("Arguments:", arguments)  # Display the selected tool arguments.

    if tool_name not in tool_functions:  # Check whether the selected tool is supported by the application.
        print("Unknown tool:", tool_name)  # Display the unsupported tool.
        break  # Stop the agent because the requested capability is unavailable.

    result = tool_functions[tool_name](**arguments)  # Execute the selected Python tool function.

    print("Tool Result:", result)  # Display the result returned by the tool.

    if tool_name == "get_payment_status":  # Check whether payment status was retrieved.
        state["payment_status"] = result  # Save the payment status in agent state.

    elif tool_name == "get_failure_reason":  # Check whether failure reason was retrieved.
        state["failure_reason"] = result  # Save the failure reason in agent state.

    elif tool_name == "get_customer_details":  # Check whether customer information was retrieved.
        state["customer_details"] = result  # Save the customer information in agent state.

    messages.append(
        {
            "role": "tool",  # Identify this message as a tool result.
            "content": result,  # Send the tool result back to the model.
        }
    )


if all(state.values()):  # Check whether every required state value has been collected.

    final_prompt = f"""  # Create the final answer prompt.
Create a final payment investigation report.  # Ask the LLM to create the final report.

Use only the information retrieved from the tools.  # Prevent unsupported information from being added.

Payment Status: {state["payment_status"]}  # Provide the retrieved payment status.
Failure Reason: {state["failure_reason"]}  # Provide the retrieved failure reason.
Customer Details: {state["customer_details"]}  # Provide the retrieved customer information.

Do not invent dates, times, amounts, or other information that is not present above.  # Add a grounding rule.
"""  # Finish the final answer prompt.

    response = ollama.chat(
        model="llama3.2:3b",  # Use the local llama3.2:3b model.
        messages=[{"role": "user", "content": final_prompt}],  # Send only the verified information to the model.
    )

    print("\nFinal Answer:", response["message"]["content"])  # Display the final investigation report.

else:  # Handle the case where the investigation could not collect all required information.
    print("\nInvestigation could not be completed.")  # Display the incomplete investigation message.