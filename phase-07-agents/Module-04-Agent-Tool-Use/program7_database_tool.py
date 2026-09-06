import os  # Import os to read environment variables.
import mysql.connector  # Import the MySQL connector.
import ollama  # Import Ollama for communicating with the local LLM.
from dotenv import load_dotenv  # Import dotenv to load .env configuration.


load_dotenv()  # Load environment variables from the .env file.


DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),  # Read the MySQL host.
    "port": int(os.getenv("MYSQL_PORT", "3306")),  # Read the MySQL port.
    "user": os.getenv("MYSQL_USER", "root"),  # Read the MySQL username.
    "password": os.getenv("MYSQL_PASSWORD", ""),  # Read the MySQL password.
    "database": os.getenv("MYSQL_DATABASE", "agent_demo"),  # Read the database name.
}


def setup_database():  # Create the demo database and sample data.

    connection = mysql.connector.connect(
        host=DB_CONFIG["host"],  # Connect to the MySQL host.
        port=DB_CONFIG["port"],  # Use the MySQL port.
        user=DB_CONFIG["user"],  # Use the MySQL username.
        password=DB_CONFIG["password"],  # Use the MySQL password.
    )

    cursor = connection.cursor()  # Create a database cursor.

    cursor.execute(
        f"CREATE DATABASE IF NOT EXISTS `{DB_CONFIG['database']}`"
    )  # Create the demo database if it does not exist.

    cursor.close()  # Close the cursor.

    connection.close()  # Close the connection.

    connection = mysql.connector.connect(
        **DB_CONFIG  # Connect directly to the demo database.
    )

    cursor = connection.cursor()  # Create a cursor for the database.

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS payments (
            transaction_id INT PRIMARY KEY,
            customer_id INT NOT NULL,
            amount DECIMAL(12,2) NOT NULL,
            status VARCHAR(20) NOT NULL
        )
        """
    )  # Create the payments table.

    sample_payments = [
        (1001, 2001, 5000.00, "FAILED"),  # Define payment 1001.
        (1002, 2002, 2500.00, "SUCCESS"),  # Define payment 1002.
        (1003, 2003, 1500.00, "PENDING"),  # Define payment 1003.
    ]

    cursor.executemany(
        """
        INSERT INTO payments
        (transaction_id, customer_id, amount, status)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            customer_id = VALUES(customer_id),
            amount = VALUES(amount),
            status = VALUES(status)
        """,
        sample_payments,
    )  # Insert or update the sample payments.

    connection.commit()  # Save the changes.

    cursor.close()  # Close the cursor.

    connection.close()  # Close the connection.


def get_payment_status(transaction_id) -> str:  # Define the database tool.

    try:
        transaction_id = int(transaction_id)  # Convert the LLM argument to an integer.
    except (TypeError, ValueError):
        return "Invalid transaction ID."  # Return an error for invalid input.

    connection = mysql.connector.connect(
        **DB_CONFIG  # Connect to MySQL.
    )

    cursor = connection.cursor(dictionary=True)  # Return database rows as dictionaries.

    cursor.execute(
        """
        SELECT transaction_id, customer_id, amount, status
        FROM payments
        WHERE transaction_id = %s
        """,
        (transaction_id,),
    )  # Execute a parameterized SQL query.

    row = cursor.fetchone()  # Get the matching payment.

    cursor.close()  # Close the cursor.

    connection.close()  # Close the connection.

    if row is None:  # Check whether the transaction exists.
        return f"Payment transaction {transaction_id} was not found."  # Return a not-found message.

    return (
        f"Transaction ID: {row['transaction_id']}, "
        f"Customer ID: {row['customer_id']}, "
        f"Amount: ₹{row['amount']}, "
        f"Status: {row['status']}"
    )  # Return verified information from MySQL.


tools = [
    {
        "type": "function",  # Define a callable function tool.
        "function": {
            "name": "get_payment_status",  # Give the tool a name.
            "description": "Retrieve payment transaction information from MySQL.",  # Explain the tool.
            "parameters": {
                "type": "object",  # Define the arguments as an object.
                "properties": {
                    "transaction_id": {
                        "type": "integer",  # Tell the LLM that the ID should be an integer.
                        "description": "The payment transaction ID to look up.",  # Explain the argument.
                    }
                },
                "required": ["transaction_id"],  # Require the transaction ID.
            },
        },
    }
]


tool_functions = {
    "get_payment_status": get_payment_status,  # Map the tool name to the Python function.
}


setup_database()  # Prepare the MySQL database and sample records.


user_request = "What is the status of payment transaction 1001?"  # Define the user's request.


messages = [
    {
        "role": "system",  # Define the agent's behavior.
        "content": (
            "You are a database assistant. "
            "Use the database tool when database information is required. "
            "Treat database results as authoritative. "
            "Answer using only information returned by the database tool. "
            "Do not invent information. "
            "Do not ask for additional information if the tool already provides the answer."
        ),
    },
    {
        "role": "user",  # Identify the user's message.
        "content": user_request,  # Store the user's question.
    },
]


response = ollama.chat(
    model="llama3.2:3b",  # Use the local LLM.
    messages=messages,  # Send the conversation.
    tools=tools,  # Give the LLM access to the database tool.
)


assistant_message = response["message"]  # Extract the LLM response.

messages.append(assistant_message)  # Add the LLM response to the conversation.

tool_calls = assistant_message.get("tool_calls", [])  # Get the selected tool calls.


if not tool_calls:  # Check whether the LLM selected a tool.

    print("No database tool was selected.")  # Display that no tool was selected.

    print(
        "LLM Response:",
        assistant_message.get("content", ""),
    )  # Display the LLM's response.

else:  # Execute the selected database tool.

    tool_call = tool_calls[0]  # Get the first tool call.

    tool_name = tool_call["function"]["name"]  # Get the selected tool name.

    arguments = tool_call["function"]["arguments"]  # Get the tool arguments.

    print("Selected Tool:", tool_name)  # Display the selected tool.

    print("Arguments:", arguments)  # Display the arguments.

    if tool_name not in tool_functions:  # Check whether the tool is allowed.

        print("Error: Unknown tool.")  # Reject an unknown tool.

    else:  # Execute the approved tool.

        tool_function = tool_functions[tool_name]  # Get the actual Python function.

        result = tool_function(**arguments)  # Execute the database function.

        print("Database Result:", result)  # Display the MySQL result.

        messages.append(
            {
                "role": "tool",  # Identify this message as a tool result.
                "content": result,  # Send the MySQL result to the LLM.
            }
        )

        messages.append(
            {
                "role": "user",  # Provide instructions for the final response.
                "content": (
                    "Answer the original question now. "
                    "Use the database result above as the source of truth. "
                    "Do not ask for additional information. "
                    "Do not invent or add any information."
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