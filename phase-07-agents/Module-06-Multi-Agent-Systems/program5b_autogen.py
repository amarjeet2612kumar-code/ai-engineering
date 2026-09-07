# Import asyncio to run asynchronous Python code.
import asyncio

# Import the AutoGen agent class.
from autogen_agentchat.agents import AssistantAgent

# Import the Ollama model client.
from autogen_ext.models.ollama import OllamaChatCompletionClient


# Define the asynchronous main function.
async def main():

    # Create the local Ollama model client.
    model_client = OllamaChatCompletionClient(
        model="llama3.2:3b",
    )

    # Create the Spark Agent.
    spark_agent = AssistantAgent(
        # Give the agent a unique name.
        name="spark_agent",

        # Define the Spark Agent's specialization.
        system_message=(
            "You are a Spark troubleshooting specialist. "
            "Analyze Spark job failures and provide concise findings."
        ),

        # Give the agent access to the Ollama model.
        model_client=model_client,
    )

    # Create the Kafka Agent.
    kafka_agent = AssistantAgent(
        # Give the agent a unique name.
        name="kafka_agent",

        # Define the Kafka Agent's specialization.
        system_message=(
            "You are a Kafka troubleshooting specialist. "
            "Analyze Kafka problems and provide concise findings."
        ),

        # Give the agent access to the Ollama model.
        model_client=model_client,
    )

    # Create a message representing information from the Spark Agent.
    spark_message = (
        "Spark job failed because Kafka input was unavailable. "
        "As a Kafka specialist, explain what should be investigated."
    )

    # Send the message to the Kafka Agent asynchronously.
    response = await kafka_agent.run(
        task=spark_message
    )

    # Display the Kafka Agent's response.
    print("Kafka Agent Response:")
    print(response)

    # Close the model client after execution.
    await model_client.close()


# Start the asynchronous program.
asyncio.run(main())