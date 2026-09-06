from pathlib import Path  # Import Path for safe filesystem path handling.
import ollama  # Import Ollama for communicating with the local LLM.


BASE_DIR = Path(__file__).resolve().parent  # Get the directory containing this Python program.

ALLOWED_DIRECTORY = BASE_DIR / "sample_logs"  # Define the directory the agent is allowed to access.


def read_file(file_name: str) -> str:  # Define the file-system tool.

    requested_path = (ALLOWED_DIRECTORY / file_name).resolve()  # Build and normalize the requested file path.

    try:
        requested_path.relative_to(ALLOWED_DIRECTORY.resolve())  # Verify the path stays inside the allowed directory.
    except ValueError:
        return "Access denied: file is outside the allowed directory."  # Reject paths outside the allowed directory.

    if not requested_path.exists():  # Check whether the requested file exists.
        return f"File not found: {file_name}"  # Return a clear not-found message.

    if not requested_path.is_file():  # Check whether the path points to a file.
        return f"Not a file: {file_name}"  # Reject directories.

    try:
        content = requested_path.read_text(
            encoding="utf-8"
        )  # Read the file using UTF-8 encoding.

        return content  # Return the file contents to the agent.

    except OSError as error:  # Handle filesystem errors.
        return f"File read failed: {error}"  # Return the filesystem error.


tools = [
    {
        "type": "function",  # Define the capability as an Ollama function tool.
        "function": {
            "name": "read_file",  # Give the filesystem tool a unique name.
            "description": "Read a text file from the approved application log directory.",  # Explain when the tool should be used.
            "parameters": {
                "type": "object",  # Define the tool arguments as an object.
                "properties": {
                    "file_name": {
                        "type": "string",  # Define the filename as a string.
                        "description": "The name of the log file to read, such as application.log.",  # Explain the parameter.
                    }
                },
                "required": ["file_name"],  # Require the filename.
            },
        },
    }
]


tool_functions = {
    "read_file": read_file,  # Map the tool name to the Python function.
}


user_request = "Read application.log and tell me what error occurred."  # Define the user's request.


messages = [
    {
        "role": "system",  # Define the agent's behavior.
        "content": (
            "You are a file analysis assistant. "
            "Use the file system tool when the user asks about file contents. "
            "Treat the file content as the source of truth. "
            "Do not invent information. "
            "Only access files through the provided tool."
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
    tools=tools,  # Give the LLM access to the filesystem tool.
)


assistant_message = response["message"]  # Extract the LLM response.

messages.append(assistant_message)  # Add the LLM response to the conversation.

tool_calls = assistant_message.get("tool_calls", [])  # Extract the selected tool calls.


if not tool_calls:  # Check whether the LLM selected the filesystem tool.

    print("No file system tool was selected.")  # Display that no tool was selected.

    print(
        "LLM Response:",
        assistant_message.get("content", ""),
    )  # Display the direct LLM response.

else:  # Execute the selected filesystem tool.

    tool_call = tool_calls[0]  # Get the first tool call.

    tool_name = tool_call["function"]["name"]  # Extract the tool name.

    arguments = tool_call["function"]["arguments"]  # Extract the tool arguments.

    print("Selected Tool:", tool_name)  # Display the selected tool.

    print("Arguments:", arguments)  # Display the selected filename.

    if tool_name not in tool_functions:  # Validate that the requested tool exists.

        print("Error: Unknown tool.")  # Reject an unknown tool.

    else:  # Execute the approved filesystem tool.

        tool_function = tool_functions[tool_name]  # Get the actual Python function.

        result = tool_function(**arguments)  # Read the file.

        print("\nFile Result:\n")  # Display a heading for the file contents.

        print(result)  # Display the file contents.

        messages.append(
            {
                "role": "tool",  # Identify this message as a tool result.
                "content": result,  # Send the file content back to the LLM.
            }
        )

        messages.append(
            {
                "role": "user",  # Provide instructions for the final answer.
                "content": (
                    "Answer the original question using only the file content above. "
                    "Identify the actual error from the log. "
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