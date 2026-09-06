import ollama  # Import the Ollama library to communicate with the local LLM.


def get_payment_status(transaction_id: str) -> str:  # Define a tool that checks payment status.
    return f"Payment {transaction_id} is FAILED."  # Return the payment status for the given transaction.


def get_account_balance(customer_id: str) -> str:  # Define a tool that checks customer account balance.
    return f"Customer {customer_id} has an account balance of ₹5,000."  # Return the balance for the given customer.


tools = [
    {
        "type": "function",  # Tell Ollama that this tool is a callable function.
        "function": {
            "name": "get_payment_status",  # Give the tool a unique name.
            "description": "Check the status of a payment transaction.",  # Tell the LLM what this tool does.
            "parameters": {
                "type": "object",  # Define the tool arguments as an object.
                "properties": {
                    "transaction_id": {
                        "type": "string",  # Tell the LLM that transaction ID is a string.
                        "description": "The payment transaction ID.",  # Explain the transaction ID argument.
                    }
                },
                "required": ["transaction_id"],  # Tell the LLM that transaction ID must be provided.
            },
        },
    },
    {
        "type": "function",  # Tell Ollama that this tool is a callable function.
        "function": {
            "name": "get_account_balance",  # Give the tool a unique name.
            "description": "Check the account balance for a customer.",  # Tell the LLM what this tool does.
            "parameters": {
                "type": "object",  # Define the tool arguments as an object.
                "properties": {
                    "customer_id": {
                        "type": "string",  # Tell the LLM that customer ID is a string.
                        "description": "The customer ID.",  # Explain the customer ID argument.
                    }
                },
                "required": ["customer_id"],  # Tell the LLM that customer ID must be provided.
            },
        },
    },
]


user_requests = [
    "What is 10 + 20?",  # This request should be answered without using a tool.
    "Check the status of payment transaction 1001.",  # This request should use the payment tool.
    "Get the account balance for customer 2001.",  # This request should use the balance tool.
]


for user_request in user_requests:  # Process each user request one by one.

    print("\nUser Request:", user_request)  # Display the current user request.

    response = ollama.chat(
        model="llama3.2:3b",  # Use the local llama3.2:3b model.
        messages=[{"role": "user", "content": user_request}],  # Send the user's request to the model.
        tools=tools,  # Give the model the tools it is allowed to use.
    )

    message = response["message"]  # Extract the model's response message.

    tool_calls = message.get("tool_calls", [])  # Extract the tool calls selected by the model.

    if not tool_calls:  # Check whether the model decided that no tool is required.
        print("Decision: No tool required.")  # Display that no tool was selected.
        print("LLM Response:", message.get("content", ""))  # Display the model's normal response.
        continue  # Skip tool execution and process the next request.

    tool_call = tool_calls[0]  # Get the first tool call selected by the model.

    tool_name = tool_call["function"]["name"]  # Extract the name of the selected tool.

    arguments = tool_call["function"]["arguments"]  # Extract the arguments selected by the model.

    print("Decision: Tool required.")  # Display that a tool is required.
    print("Selected Tool:", tool_name)  # Display the selected tool name.
    print("Arguments:", arguments)  # Display the arguments selected by the model.

    if tool_name == "get_payment_status":  # Check whether the payment status tool was selected.
        result = get_payment_status(**arguments)  # Execute the payment tool using the model's arguments.

    elif tool_name == "get_account_balance":  # Check whether the account balance tool was selected.
        result = get_account_balance(**arguments)  # Execute the balance tool using the model's arguments.

    else:  # Handle a tool name that the application does not recognize.
        result = "Unknown tool."  # Return an error instead of executing an unknown function.

    print("Tool Result:", result)  # Display the result returned by the selected tool.