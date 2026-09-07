# Import TypedDict-related functionality if we later want structured state.
from typing import TypedDict

# Import the local Ollama chat model.
from langchain_ollama import ChatOllama

# Import message types used to send instructions and user requests.
from langchain_core.messages import HumanMessage, SystemMessage

# Import the decorator used to create LangChain tools.
from langchain_core.tools import tool


# Define the Spark tool that retrieves Spark job information.
@tool
def get_spark_status(job_id: int) -> str:
    """Get the status of a Spark job."""

    # Simulate Spark job data for this learning example.
    jobs = {
        101: "FAILED - Executor OutOfMemory",
        102: "SUCCESS",
    }

    # Return the status for the requested job.
    return jobs.get(job_id, "JOB_NOT_FOUND")


# Define the Kafka tool that retrieves Kafka topic information.
@tool
def get_kafka_status(topic: str) -> str:
    """Get the status of a Kafka topic."""

    # Simulate Kafka topic data for this learning example.
    topics = {
        "customer-events": "HEALTHY",
        "payment-events": "CONSUMER_LAG_HIGH",
    }

    # Return the status for the requested topic.
    return topics.get(topic, "TOPIC_NOT_FOUND")


# Create the local Ollama LLM used by both specialized agents.
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)


# Define instructions that specialize the Spark Agent.
spark_system_prompt = """
You are a Spark troubleshooting specialist.
You only handle Spark-related problems.
Use the available Spark tool when job information is required.
Give a short and clear answer.
"""


# Define instructions that specialize the Kafka Agent.
kafka_system_prompt = """
You are a Kafka troubleshooting specialist.
You only handle Kafka-related problems.
Use the available Kafka tool when topic information is required.
Give a short and clear answer.
"""


# Create the Spark Agent by giving the LLM access to only the Spark tool.
spark_agent = llm.bind_tools([get_spark_status])


# Create the Kafka Agent by giving the LLM access to only the Kafka tool.
kafka_agent = llm.bind_tools([get_kafka_status])


# Create the request containing Spark-specific instructions and the user question.
spark_messages = [
    SystemMessage(content=spark_system_prompt),
    HumanMessage(content="Why did Spark job 101 fail?"),
]


# Send the Spark request to the Spark Agent.
spark_response = spark_agent.invoke(spark_messages)


# Check whether the Spark Agent requested a tool.
if spark_response.tool_calls:

    # Get the first tool call requested by the Spark Agent.
    tool_call = spark_response.tool_calls[0]

    # Execute the requested Spark tool with the arguments selected by the LLM.
    spark_result = get_spark_status.invoke(tool_call["args"])

    # Display the result returned by the Spark tool.
    print("Spark Agent Tool Result:", spark_result)


# Create the request containing Kafka-specific instructions and the user question.
kafka_messages = [
    SystemMessage(content=kafka_system_prompt),
    HumanMessage(content="What is the status of payment-events?"),
]


# Send the Kafka request to the Kafka Agent.
kafka_response = kafka_agent.invoke(kafka_messages)


# Check whether the Kafka Agent requested a tool.
if kafka_response.tool_calls:

    # Get the first tool call requested by the Kafka Agent.
    tool_call = kafka_response.tool_calls[0]

    # Execute the requested Kafka tool with the arguments selected by the LLM.
    kafka_result = get_kafka_status.invoke(tool_call["args"])

    # Display the result returned by the Kafka tool.
    print("Kafka Agent Tool Result:", kafka_result)