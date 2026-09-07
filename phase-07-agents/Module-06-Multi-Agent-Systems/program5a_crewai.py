# Import Agent, Task, and Crew from CrewAI.
from crewai import Agent, Task, Crew


# Create the Spark specialist.
spark_agent = Agent(
    # Define the agent's role.
    role="Spark Troubleshooting Specialist",

    # Define what the agent should achieve.
    goal="Identify the cause of Spark job failures.",

    # Provide domain-specific background.
    backstory="You are an experienced Spark engineer who investigates Spark failures.",

    # Use verbose output so we can see what CrewAI is doing.
    verbose=True,
)


# Create the Kafka specialist.
kafka_agent = Agent(
    # Define the agent's role.
    role="Kafka Troubleshooting Specialist",

    # Define what the agent should achieve.
    goal="Identify Kafka-related problems affecting data pipelines.",

    # Provide domain-specific background.
    backstory="You are an experienced Kafka engineer who investigates Kafka issues.",

    # Use verbose output so we can see what CrewAI is doing.
    verbose=True,
)


# Create the task for the Spark Agent.
spark_task = Task(
    # Define the work that needs to be performed.
    description="Explain the common causes of a Spark Executor OutOfMemory error.",

    # Define the expected result.
    expected_output="A short explanation of the likely causes and troubleshooting steps.",

    # Assign the task to the Spark Agent.
    agent=spark_agent,
)


# Create the task for the Kafka Agent.
kafka_task = Task(
    # Define the work that needs to be performed.
    description="Explain the common causes of high Kafka consumer lag.",

    # Define the expected result.
    expected_output="A short explanation of the likely causes and troubleshooting steps.",

    # Assign the task to the Kafka Agent.
    agent=kafka_agent,
)


# Create a Crew containing both specialized agents and their tasks.
crew = Crew(
    # Register the agents participating in the Crew.
    agents=[spark_agent, kafka_agent],

    # Register the tasks that the Crew should execute.
    tasks=[spark_task, kafka_task],

    # Enable detailed execution information.
    verbose=True,
)


# Start execution of the Crew.
result = crew.kickoff()


# Display the final Crew result.
print("\nFINAL RESULT:")
print(result)