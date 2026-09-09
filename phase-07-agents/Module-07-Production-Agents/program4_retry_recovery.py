# Import time to create delays between retry attempts.
import time

# Import requests for HTTP API communication.
import requests

# Import the local Ollama chat model.
from langchain_ollama import ChatOllama

# Import message types for communicating with the LLM.
from langchain_core.messages import HumanMessage, SystemMessage


# Define the maximum number of API attempts.
MAX_RETRIES = 3


# Define the initial delay before retrying.
INITIAL_DELAY = 1


# Create the local Ollama model.
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)


# Define the API tool with retry and recovery logic.
def get_job_data():

    # Use a public API to simulate an external service.
    url = "https://jsonplaceholder.typicode.com/todos/1"

    # Start the retry loop.
    for attempt in range(1, MAX_RETRIES + 1):

        # Display the current attempt.
        print(f"\nAPI Attempt {attempt}/{MAX_RETRIES}")

        try:

            # Call the external API with a short timeout.
            response = requests.get(
                url,
                timeout=5,
            )

            # Raise an exception for HTTP error responses.
            response.raise_for_status()

            # Return the API response when successful.
            return response.json()

        except requests.RequestException as error:

            # Display the temporary API failure.
            print(f"API request failed: {error}")

            # Check whether more retries are available.
            if attempt < MAX_RETRIES:

                # Calculate exponential backoff delay.
                delay = INITIAL_DELAY * (2 ** (attempt - 1))

                # Display the retry delay.
                print(f"Retrying in {delay} seconds...")

                # Wait before the next attempt.
                time.sleep(delay)

            else:

                # Display the final recovery message.
                print("All retries exhausted.")

                # Return None to indicate failure.
                return None


# Define the agent instructions.
system_prompt = """
You are a DataOps assistant.

The application has an API tool that retrieves job information.

If the API succeeds, analyze the returned information.

If the API fails after all retries, report that the external service
could not be reached.

Do not invent API results.
"""


# Create the messages sent to the LLM.
messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content="Retrieve the job information from the external API."),
]


# Ask the LLM to understand the task.
response = llm.invoke(messages)


# Display the agent's response.
print("\nAgent:")
print(response.content)


# Execute the API tool with retry handling.
job_data = get_job_data()


# Check whether the API call eventually succeeded.
if job_data:

    # Display the successful API result.
    print("\nAPI Result:")
    print(job_data)

else:

    # Display the recovery result after all retries failed.
    print("\nRecovery:")
    print("Unable to retrieve job information after retries.")