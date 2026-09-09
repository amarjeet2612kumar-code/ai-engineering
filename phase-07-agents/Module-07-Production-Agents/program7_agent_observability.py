import os
import mysql.connector

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from opik import track, configure


# Load environment variables from the .env file
load_dotenv()


# Read the Opik API key from the .env file
OPIK_API_KEY = os.getenv("OPIK_API_KEY")

# Read the Opik workspace from the .env file
OPIK_WORKSPACE = os.getenv("OPIK_WORKSPACE")

# Read the Opik API URL from the .env file
OPIK_URL = os.getenv("OPIK_URL_OVERRIDE")

# Read the Opik project name
OPIK_PROJECT_NAME = os.getenv("OPIK_PROJECT_NAME")

# Stop the program if the Opik API key is missing
if not OPIK_API_KEY:
    raise ValueError("OPIK_API_KEY is missing from .env")


# Stop the program if the Opik workspace is missing
if not OPIK_WORKSPACE:
    raise ValueError("OPIK_WORKSPACE is missing from .env")


# Configure the Opik SDK using the credentials from .env
configure(
    api_key=OPIK_API_KEY,
    workspace=OPIK_WORKSPACE,
    url_override=OPIK_URL,
    project_name=OPIK_PROJECT_NAME,
    force=True
)


# Create a connection to the DataOps MySQL database
def get_db_connection():
    # Return a connection to the local MySQL database
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="dataops_demo"
    )


# Track the database tool execution in Opik
@track
def get_job_status(job_id: int) -> dict:

    # Connect to MySQL
    connection = get_db_connection()

    # Create a cursor that returns database rows as dictionaries
    cursor = connection.cursor(dictionary=True)

    # Execute a parameterized query to retrieve the job
    cursor.execute(
        """
        SELECT job_id, job_name, status
        FROM jobs
        WHERE job_id = %s
        """,
        (job_id,)
    )

    # Read the first matching job
    job = cursor.fetchone()

    # Close the database cursor
    cursor.close()

    # Close the database connection
    connection.close()

    # Return the job or an error message
    return job or {"error": "Job not found"}


# Create the local Ollama model
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# Track the complete agent execution in Opik
@track
def investigate_job(job_id: int):

    # Retrieve job information from MySQL
    job = get_job_status(job_id)

    # Send the database result to the LLM for analysis
    response = llm.invoke(
        f"""
        You are a DataOps assistant.

        Analyze this job information:

        {job}

        Explain the current job status and what should be done next.

        Do not invent information.
        """
    )

    # Return the LLM's final response
    return response.content


# Define the job that we want to investigate
job_id = 101


# Execute the agent
result = investigate_job(job_id)


# Display the final agent response
print("\nAgent Result:")
print(result)