# Import BaseModel to define the structured Supervisor response.
from pydantic import BaseModel, Field

# Import TypedDict to define the shared workflow state.
from typing import TypedDict

# Import the local Ollama chat model.
from langchain_ollama import ChatOllama

# Import message types used to communicate with the LLM.
from langchain_core.messages import HumanMessage, SystemMessage


# Define the structure of the Supervisor's decision.
class SupervisorDecision(BaseModel):

    # Store the name of the next agent to execute.
    next_agent: str = Field(
        description="Next agent to execute: SPARK, KAFKA, or FINISH"
    )


# Define the structure of the shared workflow state.
class DataOpsState(TypedDict):

    # Store the original user request.
    request: str

    # Store the Spark investigation result.
    spark_result: str

    # Store the Kafka investigation result.
    kafka_result: str

    # Store the Supervisor's next decision.
    next_agent: str


# Create the local Ollama model.
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)


# Force the LLM to return the Supervisor decision in a structured format.
supervisor_llm = llm.with_structured_output(SupervisorDecision)


# Define the Spark Agent.
def spark_agent(state: DataOpsState) -> DataOpsState:

    # Simulate a Spark investigation.
    state["spark_result"] = (
        "Spark job failed because Kafka input was unavailable."
    )

    # Return the updated workflow state.
    return state


# Define the Kafka Agent.
def kafka_agent(state: DataOpsState) -> DataOpsState:

    # Simulate a Kafka investigation.
    state["kafka_result"] = (
        "Kafka topic is healthy, but consumer lag is high."
    )

    # Return the updated workflow state.
    return state


# Define the LLM-based Supervisor.
def supervisor(state: DataOpsState) -> DataOpsState:

    # Define the Supervisor's role and decision rules.
    system_prompt = """
You are a DataOps Supervisor coordinating specialized agents.

Available agents:
- SPARK
- KAFKA
- FINISH

Decision rules:

1. If Spark has not been investigated, choose SPARK.
2. If Spark has been investigated and its result indicates Kafka,
   and Kafka has not been investigated, choose KAFKA.
3. If Spark and Kafka have both been investigated, choose FINISH.

You must choose only one of:
SPARK
KAFKA
FINISH
"""

    # Provide the current workflow state to the Supervisor.
    user_prompt = f"""
User request:
{state["request"]}

Spark investigation:
{state["spark_result"]}

Kafka investigation:
{state["kafka_result"]}
"""

    # Build the messages sent to the Supervisor.
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    # Ask the LLM Supervisor to decide the next agent.
    decision = supervisor_llm.invoke(messages)

    # Store the structured decision in the workflow state.
    state["next_agent"] = decision.next_agent.upper()

    # Return the updated workflow state.
    return state


# Create the initial workflow state.
state: DataOpsState = {
    "request": "Why did my data pipeline fail?",
    "spark_result": "",
    "kafka_result": "",
    "next_agent": "",
}


# Ask the Supervisor which agent should work first.
state = supervisor(state)

# Display the first Supervisor decision.
print("Supervisor Decision 1:", state["next_agent"])


# Execute the Spark Agent when selected.
if state["next_agent"] == "SPARK":

    # Run the Spark investigation.
    state = spark_agent(state)


# Display the Spark investigation result.
print("Spark Result:", state["spark_result"])


# Ask the Supervisor what should happen next.
state = supervisor(state)

# Display the second Supervisor decision.
print("Supervisor Decision 2:", state["next_agent"])


# Execute the Kafka Agent when selected.
if state["next_agent"] == "KAFKA":

    # Run the Kafka investigation.
    state = kafka_agent(state)


# Display the Kafka investigation result.
print("Kafka Result:", state["kafka_result"])


# Ask the Supervisor for the final decision.
state = supervisor(state)

# Display the final Supervisor decision.
print("Supervisor Decision 3:", state["next_agent"])