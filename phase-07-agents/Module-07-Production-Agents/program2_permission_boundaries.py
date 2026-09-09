# Import the MySQL connector for database communication.
import mysql.connector

# Import the local Ollama chat model.
from langchain_ollama import ChatOllama

# Import message types used to communicate with the LLM.
from langchain_core.messages import HumanMessage, SystemMessage


# Define the permissions available to each application role.
PERMISSIONS = {
    "viewer": ["READ"],
    "operator": ["READ", "RESTART"],
    "admin": ["READ", "RESTART", "DELETE"],
}


# Define the role of the current user.
user_role = "viewer"


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


# Define a function to retrieve a job from MySQL.
def get_job(job_id: int):

    # Execute a parameterized SELECT query.
    cursor.execute(
        "SELECT job_id, job_name, status FROM jobs WHERE job_id = %s",
        (job_id,),
    )

    # Return the first matching job.
    return cursor.fetchone()


# Define the function that checks whether a role has permission.
def has_permission(role: str, action: str) -> bool:

    # Get the permissions assigned to the role.
    allowed_actions = PERMISSIONS.get(role, [])

    # Return True only when the requested action is allowed.
    return action in allowed_actions


# Define the agent's instructions.
system_prompt = """
You are a DataOps assistant.

Analyze the job information and recommend an appropriate action.

If the job is FAILED, recommend RESTART.

Do not execute any database modification.

The application controls permissions and executes the action.
"""


# Define the job that the user wants to investigate.
job_id = 101


# Retrieve the job from MySQL.
job = get_job(job_id)


# Check whether the job exists.
if not job:

    # Display an error when the job does not exist.
    print("Job not found.")

else:

    # Extract the job information.
    job_id, job_name, status = job

    # Build the request for the LLM.
    user_prompt = f"""
Job ID: {job_id}
Job Name: {job_name}
Current Status: {status}

What action do you recommend?
"""

    # Build the messages sent to the LLM.
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    # Ask the LLM for its recommendation.
    response = llm.invoke(messages)

    # Display the LLM recommendation.
    print("\nAgent Recommendation:")
    print(response.content)

    # Define the action the application expects from the recommendation.
    requested_action = "RESTART"

    # Display the current user role.
    print("\nUser Role:", user_role)

    # Check whether the user's role allows the requested action.
    if has_permission(user_role, requested_action):

        # Execute the restart only when permission is granted.
        cursor.execute(
            "UPDATE jobs SET status = %s WHERE job_id = %s",
            ("RESTARTED", job_id),
        )

        # Commit the database change.
        connection.commit()

        # Confirm successful execution.
        print("Permission granted. Job restarted.")

    else:

        # Block the action when the role does not have permission.
        print(
            f"Permission denied. Role '{user_role}' "
            f"cannot perform '{requested_action}'."
        )


# Close the database cursor.
cursor.close()

# Close the database connection.
connection.close()