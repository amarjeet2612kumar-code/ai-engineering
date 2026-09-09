# Import the MySQL connector for database communication.
import mysql.connector

# Import the local Ollama chat model.
from langchain_ollama import ChatOllama

# Import message types for communicating with the LLM.
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


# Define a function to retrieve a job from MySQL.
def get_job(job_id: int):

    # Execute a parameterized SELECT query.
    cursor.execute(
        "SELECT job_id, job_name, status FROM jobs WHERE job_id = %s",
        (job_id,),
    )

    # Return the first matching job.
    return cursor.fetchone()


# Define the agent's instructions.
system_prompt = """
You are a DataOps assistant.

Your job is to analyze failed jobs.

If a job is FAILED, recommend restarting it.

Do not execute any database modification yourself.

The application will ask the human for approval before
performing the restart.
"""


# Define the job that the user wants to investigate.
job_id = 101


# Retrieve the job information from MySQL.
job = get_job(job_id)


# Check whether the requested job exists.
if not job:

    # Display an error when the job does not exist.
    print("Job not found.")

else:

    # Extract the job information from the database result.
    job_id, job_name, status = job

    # Build the user request for the LLM.
    user_prompt = f"""
Job ID: {job_id}
Job Name: {job_name}
Current Status: {status}

Should this job be restarted?
"""

    # Create the messages sent to the LLM.
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    # Ask the LLM to analyze the job.
    response = llm.invoke(messages)

    # Display the agent's recommendation.
    print("\nAgent Recommendation:")
    print(response.content)

    # Ask the human for approval before modifying the database.
    approval = input("\nApprove restart? (yes/no): ").strip().lower()

    # Check whether the human approved the action.
    if approval == "yes":

        # Update the job status only after human approval.
        cursor.execute(
            "UPDATE jobs SET status = %s WHERE job_id = %s",
            ("RESTARTED", job_id),
        )

        # Commit the database modification.
        connection.commit()

        # Confirm that the approved action was executed.
        print("\nAction approved and executed.")

    else:

        # Do not modify the database when approval is denied.
        print("\nAction rejected. No database change was made.")


# Close the database cursor.
cursor.close()

# Close the MySQL connection.
connection.close()