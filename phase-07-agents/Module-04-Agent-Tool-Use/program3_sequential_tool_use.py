import ollama  # Import the Ollama library to communicate with the local LLM.


def get_payment_status(transaction_id: str) -> str:  # Define a tool that checks the payment status.
    return f"Payment {transaction_id} is FAILED."  # Return the payment status.


def get_failure_reason(transaction_id: str) -> str:  # Define a tool that checks why the payment failed.
    return f"Payment {transaction_id} failed because of INSUFFICIENT_FUNDS."  # Return the failure reason.


status_tool = [
    {
        "type": "function",  # Tell Ollama that this is a callable function.
        "function": {
            "name": "get_payment_status",  # Give the tool a unique name.
            "description": "Check the current status of a payment transaction.",  # Explain what the tool does.
            "parameters": {
                "type": "object",  # Define the tool input as an object.
                "properties": {
                    "transaction_id": {
                        "type": "string",  # Define transaction ID as a string.
                        "description": "The payment transaction ID.",  # Explain the transaction ID argument.
                    }
                },
                "required": ["transaction_id"],  # Require the transaction ID.
            },
        },
    }
]


failure_tool = [
    {
        "type": "function",  # Tell Ollama that this is a callable function.
        "function": {
            "name": "get_failure_reason",  # Give the tool a unique name.
            "description": "Find the reason why a payment failed.",  # Explain what the tool does.
            "parameters": {
                "type": "object",  # Define the tool input as an object.
                "properties": {
                    "transaction_id": {
                        "type": "string",  # Define transaction ID as a string.
                        "description": "The payment transaction ID.",  # Explain the transaction ID argument.
                    }
                },
                "required": ["transaction_id"],  # Require the transaction ID.
            },
        },
    }
]


user_request = "Investigate payment transaction 1001."  # Define the user's investigation request.


messages = [
    {"role": "user", "content": user_request}  # Start the conversation with the user's request.
]


print("User Request:", user_request)  # Display the user's request.


response = ollama.chat(
    model="llama3.2:3b",  # Use the local llama3.2:3b model.
    messages=messages,  # Send the current conversation to the model.
    tools=status_tool,  # Initially expose only the payment status tool.
)


messages.append(response["message"])  # Save the model response for the next step.


tool_calls = response["message"].get("tool_calls", [])  # Extract the tool calls selected by the model.


if tool_calls:  # Check whether the model selected a tool.

    tool_call = tool_calls[0]  # Get the first tool call.

    tool_name = tool_call["function"]["name"]  # Extract the selected tool name.

    arguments = tool_call["function"]["arguments"]  # Extract the arguments selected by the model.

    print("\nStep 1")  # Display the first sequential step.
    print("Selected Tool:", tool_name)  # Display the selected tool.
    print("Arguments:", arguments)  # Display the tool arguments.

    if tool_name == "get_payment_status":  # Verify that the expected payment status tool was selected.
        result = get_payment_status(**arguments)  # Execute the payment status tool.
    else:  # Handle an unexpected tool.
        result = "Unexpected tool selected."  # Store an error message.

    print("Tool Result:", result)  # Display the result returned by the first tool.

    messages.append(
        {
            "role": "tool",  # Identify this message as a tool result.
            "content": result,  # Store the result returned by the tool.
        }
    )

    if "FAILED" in result:  # Check whether the payment failed before moving to the next step.

        messages.append(
            {
                "role": "user",  # Add a clear instruction for the next investigation step.
                "content": "The payment status is FAILED. Now call get_failure_reason for transaction 1001.",  # Tell the model exactly which tool is needed next.
            }
        )

        print("\nStep 2")  # Display the second sequential step.

        response = ollama.chat(
            model="llama3.2:3b",  # Use the local llama3.2:3b model again.
            messages=messages,  # Send the conversation and the first tool result.
            tools=failure_tool,  # Expose only the failure reason tool.
        )

        messages.append(response["message"])  # Save the second model response.

        tool_calls = response["message"].get("tool_calls", [])  # Extract the second tool call.

        if tool_calls:  # Check whether the model selected the failure reason tool.

            tool_call = tool_calls[0]  # Get the second tool call.

            tool_name = tool_call["function"]["name"]  # Extract the second tool name.

            arguments = tool_call["function"]["arguments"]  # Extract the second tool arguments.

            print("Selected Tool:", tool_name)  # Display the second selected tool.
            print("Arguments:", arguments)  # Display the second tool arguments.

            if tool_name == "get_failure_reason":  # Verify that the expected failure reason tool was selected.
                result = get_failure_reason(**arguments)  # Execute the failure reason tool.
            else:  # Handle an unexpected tool.
                result = "Unexpected tool selected."  # Store an error message.

            print("Tool Result:", result)  # Display the result returned by the second tool.

            messages.append(
                {
                    "role": "tool",  # Identify this message as a tool result.
                    "content": result,  # Store the second tool result.
                }
            )

            response = ollama.chat(
                model="llama3.2:3b",  # Use the local llama3.2:3b model for the final answer.
                messages=messages,  # Send the complete investigation to the model.
            )

            print("\nFinal Answer:", response["message"]["content"])  # Display the final answer.

        else:  # Handle the case where the second tool was not selected.
            print("No failure reason tool was selected.")  # Display the problem.

    else:  # Handle a payment that did not fail.
        print("\nPayment did not fail, so no failure reason lookup is required.")  # Explain why the second tool was skipped.

else:  # Handle the case where the first tool was not selected.
    print("\nNo payment status tool was selected.")  # Display the problem.