import os  # Import os to read environment variables.
import time  # Import time for retry delays.
import requests  # Import requests for HTTP API calls.
import ollama  # Import Ollama for the local LLM.
from dotenv import load_dotenv  # Import dotenv to load .env configuration.


load_dotenv()  # Load environment variables from the .env file.


API_KEY = os.getenv("EXCHANGE_RATE_API_KEY")  # Read the API key from the environment.

BASE_URL = "https://v6.exchangerate-api.com/v6"  # Define the ExchangeRate-API base URL.


def get_exchange_rate(
    base_currency: str,
    target_currency: str,
) -> str:  # Define the authenticated API tool.

    if not API_KEY:  # Check whether the API key exists.
        return "API configuration error: EXCHANGE_RATE_API_KEY is not configured."  # Return a configuration error.

    base_currency = base_currency.upper()  # Normalize the base currency.

    target_currency = target_currency.upper()  # Normalize the target currency.

    url = f"{BASE_URL}/{API_KEY}/latest/{base_currency}"  # Build the authenticated API endpoint.

    for attempt in range(3):  # Allow up to three API attempts.

        try:
            response = requests.get(
                url,
                timeout=10,
            )  # Send the authenticated API request.

            if response.status_code == 401:  # Check for authentication failure.
                return "API authentication failed. Check the API key."  # Return an authentication error.

            if response.status_code == 403:  # Check for authorization failure.
                return "API access was forbidden."  # Return an authorization error.

            if response.status_code == 429:  # Check whether the API rate limit was exceeded.
                if attempt < 2:  # Check whether another retry is available.
                    time.sleep(2 ** attempt)  # Wait before retrying.
                    continue  # Retry the API request.

                return "API rate limit exceeded after multiple attempts."  # Stop after the retry limit.

            response.raise_for_status()  # Raise an exception for other HTTP errors.

            data = response.json()  # Convert the API response into a Python dictionary.

            if data.get("result") != "success":  # Check whether the API reported success.
                return f"API returned an error: {data.get('error-type', 'unknown error')}"  # Return the API error.

            rates = data.get("conversion_rates", {})  # Extract the currency conversion rates.

            if target_currency not in rates:  # Check whether the requested currency exists.
                return f"Currency {target_currency} was not found in the API response."  # Return a validation error.

            rate = rates[target_currency]  # Extract the requested exchange rate.

            return (
                f"1 {base_currency} = "
                f"{rate} {target_currency}"
            )  # Return the verified exchange rate.

        except requests.Timeout:  # Handle request timeout errors.
            if attempt < 2:  # Check whether another retry is available.
                time.sleep(2 ** attempt)  # Wait before retrying.
                continue  # Retry the request.

            return "API request timed out after multiple attempts."  # Return the timeout error.

        except requests.RequestException as error:  # Handle other HTTP/network errors.
            return f"API request failed: {error}"  # Return the request error.

    return "API request failed after multiple attempts."  # Return a final failure message.


tools = [
    {
        "type": "function",  # Define the capability as an Ollama function tool.
        "function": {
            "name": "get_exchange_rate",  # Give the API tool a unique name.
            "description": "Get the latest exchange rate between two currencies using an authenticated external API.",  # Explain when to use the tool.
            "parameters": {
                "type": "object",  # Define the function arguments as an object.
                "properties": {
                    "base_currency": {
                        "type": "string",  # Define the base currency.
                        "description": "Currency to convert from, such as USD.",  # Explain the parameter.
                    },
                    "target_currency": {
                        "type": "string",  # Define the target currency.
                        "description": "Currency to convert to, such as INR.",  # Explain the parameter.
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
            "You are an API assistant. "
            "Use the exchange-rate API when the user asks for current exchange-rate information. "
            "Treat the API response as authoritative. "
            "Do not invent or modify exchange rates."
        ),
    },
    {
        "role": "user",  # Identify the user's request.
        "content": user_request,  # Store the user's question.
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

    if tool_name not in tool_functions:  # Validate the requested tool.

        print("Error: Unknown tool.")  # Reject an unknown tool.

    else:  # Execute the approved API tool.

        tool_function = tool_functions[tool_name]  # Get the actual Python function.

        result = tool_function(**arguments)  # Execute the authenticated API call.

        print("API Result:", result)  # Display the API result.

        messages.append(
            {
                "role": "tool",  # Identify this message as a tool result.
                "content": result,  # Send the API result back to the LLM.
            }
        )

        messages.append(
            {
                "role": "user",  # Provide final-answer instructions.
                "content": (
                    "Answer the original question using the API result above. "
                    "Treat the API result as the source of truth. "
                    "Do not invent or modify the exchange rate."
                ),
            }
        )

        final_response = ollama.chat(
            model="llama3.2:3b",  # Use the local LLM again.
            messages=messages,  # Send the complete conversation.
        )

        print(
            "\nFinal Answer:",
            final_response["message"]["content"],
        )  # Display the final answer.