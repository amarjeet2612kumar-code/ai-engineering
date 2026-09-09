# Import the MySQL connector for database communication.
import mysql.connector

# Import regular expressions for SQL safety checks.
import re

# Import the local Ollama chat model.
from langchain_ollama import ChatOllama

# Import message types used to communicate with the LLM.
from langchain_core.messages import HumanMessage, SystemMessage


# Create a connection to the MySQL database.
connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="dataops_demo",
)


# Create a cursor for executing SQL statements.
cursor = connection.cursor()


# Create the local Ollama model.
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)


# Define the function that validates a proposed SQL operation.
def validate_sql_safety(sql: str) -> tuple[bool, str]:

    # Remove unnecessary whitespace from the SQL.
    normalized_sql = sql.strip().upper()

    # Block DROP operations because they can remove database objects.
    if re.search(r"\bDROP\b", normalized_sql):

        # Return a safety failure.
        return False, "DROP operations are not allowed."

    # Block TRUNCATE operations because they remove all table data.
    if re.search(r"\bTRUNCATE\b", normalized_sql):

        # Return a safety failure.
        return False, "TRUNCATE operations are not allowed."

    # Block DELETE statements that do not contain a WHERE clause.
    if re.match(r"^\s*DELETE\b", normalized_sql) and " WHERE " not in normalized_sql:

        # Return a safety failure.
        return False, "DELETE without WHERE is not allowed."

    # Block UPDATE statements that do not contain a WHERE clause.
    if re.match(r"^\s*UPDATE\b", normalized_sql) and " WHERE " not in normalized_sql:

        # Return a safety failure.
        return False, "UPDATE without WHERE is not allowed."

    # Allow the SQL when none of the safety rules are violated.
    return True, "SQL passed safety validation."


# Define the agent's instructions.
system_prompt = """
You are a DataOps assistant.

Analyze the user's request and propose a SQL operation.

You may propose SQL, but you must NOT execute it.

The application will validate the SQL for safety before execution.
"""


# Define the user request.
user_request = "Delete all failed jobs from the jobs table."


# Build the messages sent to the LLM.
messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content=user_request),
]


# Ask the LLM to propose an operation.
response = llm.invoke(messages)


# Display the agent's recommendation.
print("Agent Recommendation:")
print(response.content)


# For this learning example, define the SQL proposed by the agent.
proposed_sql = "DELETE FROM jobs WHERE status = 'FAILED';"


# Display the SQL that the application is considering.
print("\nProposed SQL:")
print(proposed_sql)


# Validate the proposed SQL before sending it to MySQL.
is_safe, safety_message = validate_sql_safety(proposed_sql)


# Display the safety validation result.
print("\nSafety Check:")
print(safety_message)


# Execute the SQL only when the safety check passes.
if is_safe:

    # Execute the validated SQL.
    cursor.execute(proposed_sql)

    # Commit the database change.
    connection.commit()

    # Confirm that the operation was executed.
    print("SQL executed successfully.")

else:

    # Block the SQL when it fails the safety check.
    print("SQL execution BLOCKED.")


# Close the database cursor.
cursor.close()

# Close the database connection.
connection.close()