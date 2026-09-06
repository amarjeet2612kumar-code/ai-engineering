import ast  # Import AST to safely inspect the Python expression.
import operator  # Import operators that we explicitly allow.
import ollama  # Import Ollama for communicating with the local LLM.


allowed_operators = {
    ast.Add: operator.add,  # Allow addition.
    ast.Sub: operator.sub,  # Allow subtraction.
    ast.Mult: operator.mul,  # Allow multiplication.
    ast.Div: operator.truediv,  # Allow division.
    ast.Pow: operator.pow,  # Allow exponentiation.
    ast.Mod: operator.mod,  # Allow modulo.
}


def execute_code(expression: str) -> str:  # Define the controlled code execution tool.
    try:
        tree = ast.parse(expression, mode="eval")  # Parse the expression without executing it directly.

        def evaluate(node):  # Recursively evaluate only approved AST nodes.
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):  # Allow numeric values.
                return node.value  # Return the numeric value.

            if isinstance(node, ast.BinOp):  # Check whether the expression contains an arithmetic operation.
                left = evaluate(node.left)  # Evaluate the left side.
                right = evaluate(node.right)  # Evaluate the right side.

                operation = allowed_operators.get(type(node.op))  # Find the approved operator.

                if operation is None:  # Reject operators that are not explicitly allowed.
                    raise ValueError("Operator is not allowed.")  # Stop unsafe operations.

                return operation(left, right)  # Execute the approved arithmetic operation.

            raise ValueError("Only numeric arithmetic expressions are allowed.")  # Reject everything else.

        result = evaluate(tree.body)  # Evaluate the validated expression.

        return str(result)  # Convert the result to text for the LLM.

    except Exception as error:  # Catch execution or validation errors.
        return f"Code execution failed: {error}"  # Return the error to the agent.


tools = [
    {
        "type": "function",  # Define the capability as an Ollama function tool.
        "function": {
            "name": "execute_code",  # Give the tool a unique name.
            "description": "Execute a safe arithmetic expression when calculation is required.",  # Explain when the tool should be used.
            "parameters": {
                "type": "object",  # Define the function arguments as an object.
                "properties": {
                    "expression": {
                        "type": "string",  # Define the expression as a string.
                        "description": "A mathematical expression such as 10 + 20 * 3.",  # Explain the expected input.
                    }
                },
                "required": ["expression"],  # Require the expression.
            },
        },
    }
]


user_request = "Calculate (125 * 8) + 450."  # Define a request that requires computation.


messages = [
    {
        "role": "user",  # Identify this message as the user's request.
        "content": user_request,  # Store the user's request.
    }
]


response = ollama.chat(
    model="llama3.2:3b",  # Use the local LLM.
    messages=messages,  # Send the user's request.
    tools=tools,  # Give the LLM access to the code execution tool.
)


assistant_message = response["message"]  # Extract the assistant's response.

messages.append(assistant_message)  # Add the assistant response to the conversation.

tool_calls = assistant_message.get("tool_calls", [])  # Extract any tool calls selected by the LLM.


if tool_calls:  # Check whether the LLM decided to execute code.

    tool_call = tool_calls[0]  # Get the first tool call.

    tool_name = tool_call["function"]["name"]  # Extract the selected tool name.

    arguments = tool_call["function"]["arguments"]  # Extract the tool arguments.

    print("Selected Tool:", tool_name)  # Display the selected tool.

    print("Expression:", arguments["expression"])  # Display the expression selected by the LLM.

    result = execute_code(**arguments)  # Execute the validated expression.

    print("Execution Result:", result)  # Display the execution result.

    messages.append(
        {
            "role": "tool",  # Identify this message as a tool result.
            "content": result,  # Send the execution result back to the LLM.
        }
    )

    response = ollama.chat(
        model="llama3.2:3b",  # Use the local LLM again.
        messages=messages,  # Send the original request and execution result.
    )

    print("\nFinal Answer:", response["message"]["content"])  # Display the final answer.

else:  # Handle the case where no tool was selected.
    print("No code execution tool was selected.")  # Display that no tool was used.
    print("LLM Response:", assistant_message.get("content", ""))  # Display the direct LLM response.