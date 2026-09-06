import ollama  # Import Ollama for communicating with the local LLM.
from playwright.sync_api import sync_playwright  # Import Playwright's synchronous browser API.


def browse_web(url: str) -> str:  # Define the browser automation tool.

    try:
        with sync_playwright() as playwright:  # Start the Playwright engine.

            browser = playwright.chromium.launch(
                headless=True
            )  # Launch Chromium without opening a visible browser window.

            page = browser.new_page()  # Create a new browser page.

            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=15000,
            )  # Navigate to the requested webpage.

            title = page.title()  # Extract the webpage title.

            content = page.locator("body").inner_text()  # Extract visible text from the webpage.

            browser.close()  # Close the browser.

            return (
                f"Page Title: {title}\n\n"
                f"Page Content:\n{content[:5000]}"
            )  # Return the webpage information to the agent.

    except Exception as error:  # Handle browser or navigation errors.
        return f"Browser automation failed: {error}"  # Return the error to the agent.


tools = [
    {
        "type": "function",  # Define the browser capability as an Ollama function tool.
        "function": {
            "name": "browse_web",  # Give the browser tool a unique name.
            "description": "Open a webpage using a browser and extract its visible text.",  # Explain what the tool does.
            "parameters": {
                "type": "object",  # Define the tool arguments as an object.
                "properties": {
                    "url": {
                        "type": "string",  # Define the URL as a string.
                        "description": "The webpage URL to open.",  # Explain the URL parameter.
                    }
                },
                "required": ["url"],  # Require the URL.
            },
        },
    }
]


tool_functions = {
    "browse_web": browse_web,  # Map the tool name to the Python function.
}


user_request = "Open https://www.tpointtech.com/python-tutorial and tell me the page title."  # Define the user's request.


messages = [
    {
        "role": "system",  # Define the agent's behavior.
        "content": (
            "You are a browser automation assistant. "
            "Use the browser tool when the user asks you to open or inspect a webpage. "
            "Treat information returned by the browser as the source of truth. "
            "Do not invent webpage information."
        ),
    },
    {
        "role": "user",  # Identify the user's message.
        "content": user_request,  # Store the user's request.
    },
]


response = ollama.chat(
    model="llama3.2:3b",  # Use the local LLM.
    messages=messages,  # Send the conversation to the LLM.
    tools=tools,  # Give the LLM access to the browser tool.
)


assistant_message = response["message"]  # Extract the LLM response.

messages.append(assistant_message)  # Add the LLM response to the conversation.

tool_calls = assistant_message.get("tool_calls", [])  # Extract the selected tool calls.


if not tool_calls:  # Check whether the LLM selected the browser tool.

    print("No browser tool was selected.")  # Display that no tool was selected.

    print(
        "LLM Response:",
        assistant_message.get("content", ""),
    )  # Display the direct LLM response.

else:  # Execute the selected browser tool.

    tool_call = tool_calls[0]  # Get the first tool call.

    tool_name = tool_call["function"]["name"]  # Extract the tool name.

    arguments = tool_call["function"]["arguments"]  # Extract the browser arguments.

    print("Selected Tool:", tool_name)  # Display the selected browser tool.

    print("Arguments:", arguments)  # Display the selected URL.

    if tool_name not in tool_functions:  # Validate the requested tool.

        print("Error: Unknown tool.")  # Reject an unknown tool.

    else:  # Execute the approved browser tool.

        tool_function = tool_functions[tool_name]  # Get the actual Python function.

        result = tool_function(**arguments)  # Open the webpage using Playwright.

        print("\nBrowser Result:\n")  # Display a heading for the browser result.

        print(result)  # Display the webpage information.

        messages.append(
            {
                "role": "tool",  # Identify this message as a tool result.
                "content": result,  # Send the browser result to the LLM.
            }
        )

        messages.append(
            {
                "role": "user",  # Provide final-answer instructions.
                "content": (
                    "Answer the original question using only the browser result above. "
                    "Do not invent information."
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