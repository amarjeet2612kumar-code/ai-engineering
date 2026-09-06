import requests  # Import requests for making HTTP API calls.
import ollama  # Import Ollama for communicating with the local LLM.


def get_exchange_rate(base_currency: str, target_currency: str) -> str:  # Define the API tool.

    url = "https://api.frankfurter.app/latest"  # Define the exchange-rate API endpoint.

    params = {
        "from": base_currency.upper(),  # Set the base currency.
        "to": target_currency.upper(),  # Set the target currency.
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10,
        )  # Send the request to the external API.

        response.raise_for_status()  # Raise an error if the API request failed.

        data = response.json()  # Convert the API response into a Python dictionary.

        rate = data["rates"][target_currency.upper()]  # Extract the requested exchange rate.

        return (
            f"1 {base_currency.upper()} = "
            f"{rate} {target_currency.upper()}"
        )  # Return the exchange rate as readable text.

    except requests.RequestException as error:  # Handle network or HTTP errors.
        return f"API request failed: {error}"  # Return the API error.

    except KeyError:  # Handle unexpected API response structures.
        return "API response did not contain the requested exchange rate."  # Return a clear error.


tools = [
    {
        "type": "function",  # Define the capability as an Ollama function tool.
        "function": {
            "name": "get_exchange_rate",  # Give the API tool a unique name.
            "description": "Get the latest exchange rate between two currencies using an external API.",  # Explain when the tool should be used.
            "parameters": {
                "type": "object",  # Define the tool arguments as an object.
                "properties": {
                    "base_currency": {
                        "type": "string",  # Define the base currency.
                        "description": "The currency to convert from, such as USD.",  # Explain the base currency.
                    },
                    "target_currency": {
                        "type": "string",  # Define the target currency.
                        "description": "The currency to convert to, such as INR.",  # Explain the target currency.
                    },
                },
                "required": ["base_currency", "target_currency"],  # Require both currencies.
            },
        },
    }
]


tool_functions = {
    "get_exchange_rate": get_exchange_rate,  # Map the tool name to the Python function.
}


user_request = "What is the current exchange rate between USD and INR?"  # Define the user's request.


messages = [
    {
        "role": "system",  # Define the agent's behavior.
        "content": (
            "You are an assistant that can use external API tools. "
            "When the user asks for current exchange-rate information, "
            "use the exchange-rate API tool. "
            "Treat API results as authoritative. "
            "Do not invent exchange rates."
        ),
    },
    {
        "role": "user",  # Identify this as the user's message.
        "content": user_request,  # Store the user's request.
    },
]


response = ollama.chat(
    model="llama3.2:3b",  # Use the local LLM.
    messages=messages,  # Send the conversation.
    tools=tools,  # Give the LLM access to the API tool.
)


assistant_message = response["message"]  # Extract the LLM response.

messages.append(assistant_message)  # Add the LLM response to the conversation.

tool_calls = assistant_message.get("tool_calls", [])  # Extract the selected tool calls.


if not tool_calls:  # Check whether the LLM selected the API tool.

    print("No API tool was selected.")  # Display that no tool was selected.

    print(
        "LLM Response:",
        assistant_message.get("content", ""),
    )  # Display the direct LLM response.

else:  # Execute the selected API tool.

    tool_call = tool_calls[0]  # Get the first tool call.

    tool_name = tool_call["function"]["name"]  # Extract the tool name.

    arguments = tool_call["function"]["arguments"]  # Extract the API arguments.

    print("Selected Tool:", tool_name)  # Display the selected API tool.

    print("Arguments:", arguments)  # Display the arguments selected by the LLM.

    if tool_name not in tool_functions:  # Check whether the requested tool is allowed.

        print("Error: Unknown tool.")  # Reject an unknown tool.

    else:  # Execute the approved API tool.

        tool_function = tool_functions[tool_name]  # Get the actual Python function.

        result = tool_function(**arguments)  # Call the external API.

        print("API Result:", result)  # Display the API result.

        messages.append(
            {
                "role": "tool",  # Identify this message as a tool result.
                "content": result,  # Send the API result to the LLM.
            }
        )

        messages.append(
            {
                "role": "user",  # Provide instructions for the final response.
                "content": (
                    "Answer the original question using the API result above. "
                    "Treat the API result as the source of truth. "
                    "Do not invent or modify the exchange rate."
                ),
            }
        )

        final_response = ollama.chat(
            model="llama3.2:3b",  # Use the local LLM.
            messages=messages,  # Send the complete conversation.
        )

        print(
            "\nFinal Answer:",
            final_response["message"]["content"],
        )  # Display the final answer.