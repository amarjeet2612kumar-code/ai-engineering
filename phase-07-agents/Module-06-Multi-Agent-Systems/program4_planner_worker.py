# Import BaseModel to define the structured Planner response.
from pydantic import BaseModel, Field

# Import the local Ollama chat model.
from langchain_ollama import ChatOllama

# Import message types used to communicate with the LLM.
from langchain_core.messages import HumanMessage, SystemMessage


# Define the structure of one planned task.
class PlannedTask(BaseModel):

    # Store the worker that should execute the task.
    worker: str = Field(
        description="Worker name: SPARK or KAFKA"
    )

    # Store the task assigned to the worker.
    task: str = Field(
        description="Task that the worker should perform"
    )


# Define the structure of the Planner response.
class Plan(BaseModel):

    # Store the list of tasks created by the Planner.
    tasks: list[PlannedTask]


# Create the local Ollama model.
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)


# Configure the LLM to return a structured plan.
planner_llm = llm.with_structured_output(Plan)


# Define the Planner.
def planner(user_request: str) -> Plan:

    # Define the Planner's instructions.
    system_prompt = """
You are a DataOps Planner.

Break the user's problem into investigation tasks.

Available workers:
- SPARK
- KAFKA

For a pipeline failure involving Spark and Kafka,
create tasks for the relevant workers.

Return only the planned tasks.
"""

    # Create the messages sent to the Planner.
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_request),
    ]

    # Ask the LLM to create the plan.
    return planner_llm.invoke(messages)


# Define the Spark Worker.
def spark_worker(task: str) -> str:

    # Simulate Spark investigation.
    return "Spark job failed because Kafka input was unavailable."


# Define the Kafka Worker.
def kafka_worker(task: str) -> str:

    # Simulate Kafka investigation.
    return "Kafka topic is healthy, but consumer lag is high."


# Create the user request.
user_request = "Investigate why my Spark data pipeline failed."


# Ask the Planner to create investigation tasks.
plan = planner(user_request)


# Display the generated plan.
print("PLAN:")

# Process every task created by the Planner.
for task in plan.tasks:

    # Display the worker selected by the Planner.
    print(f"Worker: {task.worker}")

    # Display the task assigned to that worker.
    print(f"Task: {task.task}")

    # Execute the Spark Worker when selected.
    if task.worker.upper() == "SPARK":

        # Run the Spark investigation.
        result = spark_worker(task.task)

    # Execute the Kafka Worker when selected.
    elif task.worker.upper() == "KAFKA":

        # Run the Kafka investigation.
        result = kafka_worker(task.task)

    # Handle an unknown worker safely.
    else:

        # Return an error instead of executing an unknown worker.
        result = "Unknown worker"

    # Display the worker's result.
    print(f"Result: {result}")

    # Print a separator between tasks.
    print("-" * 40)