# Import the local Ollama chat model.
from langchain_ollama import ChatOllama

# Import message types used to send instructions to the LLM.
from langchain_core.messages import HumanMessage, SystemMessage


# Create the local LLM used by the Router.
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)


# Define instructions for the Router.
router_system_prompt = """
You are a request router for a DataOps system.

Choose one destination:
- SPARK
- KAFKA

Return the destination that best matches the user's request.
"""


# Create the function that routes the user request.
def route_request(user_request: str) -> str:

    # Build the messages sent to the Router.
    messages = [
        SystemMessage(content=router_system_prompt),
        HumanMessage(content=user_request),
    ]

    # Ask the LLM to determine the destination.
    response = llm.invoke(messages)

    # Convert the LLM response into uppercase text.
    decision = response.content.strip().upper()

    # Check whether the response indicates Spark.
    if "SPARK" in decision:

        # Return the validated Spark destination.
        return "SPARK"

    # Check whether the response indicates Kafka.
    if "KAFKA" in decision:

        # Return the validated Kafka destination.
        return "KAFKA"

    # Return UNKNOWN when the response cannot be identified.
    return "UNKNOWN"


# Define the Spark Agent.
def spark_agent(request: str) -> str:

    # Simulate sending the request to the Spark Agent.
    return f"Spark Agent received: {request}"


# Define the Kafka Agent.
def kafka_agent(request: str) -> str:

    # Simulate sending the request to the Kafka Agent.
    return f"Kafka Agent received: {request}"


# Create a sample user request.
user_request = "Why is my Spark job failing with executor memory errors?"


# Ask the Router to select the appropriate agent.
destination = route_request(user_request)


# Display the Router's decision.
print("Router Decision:", destination)


# Send the request to the selected agent.
if destination == "SPARK":

    # Send the request to the Spark Agent.
    result = spark_agent(user_request)

elif destination == "KAFKA":

    # Send the request to the Kafka Agent.
    result = kafka_agent(user_request)

else:

    # Handle an invalid Router decision safely.
    result = "Unknown destination"


# Display the result from the selected agent.
print("Result:", result)