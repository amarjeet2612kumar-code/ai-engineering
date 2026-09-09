from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


# Define the expected structure of the agent's decision
class AgentDecision(BaseModel):
    category: str = Field(description="Failure category")
    action: str = Field(description="Recommended action")


# Create the local Ollama model
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# Force the model to return structured output
structured_llm = llm.with_structured_output(AgentDecision)


# Define evaluation test cases
test_cases = [
    {
        "input": "Spark job failed because executor memory was exhausted.",
        "expected_category": "SPARK",
        "expected_action": "INVESTIGATE_SPARK_MEMORY",
    },
    {
        "input": "Kafka consumer lag increased significantly.",
        "expected_category": "KAFKA",
        "expected_action": "INVESTIGATE_KAFKA",
    },
    {
        "input": "The API returned HTTP 503 Service Unavailable.",
        "expected_category": "API",
        "expected_action": "RETRY",
    },
]


# Define the agent's evaluation prompt
system_prompt = """
You are a DataOps failure classification agent.

Classify the failure and recommend the correct action.

Rules:

Spark executor memory failure:
category = SPARK
action = INVESTIGATE_SPARK_MEMORY

Kafka consumer lag:
category = KAFKA
action = INVESTIGATE_KAFKA

HTTP 503 Service Unavailable:
category = API
action = RETRY

Return only the structured decision.
"""


# Track evaluation statistics
passed = 0
failed = 0


# Evaluate every test case
for index, test_case in enumerate(test_cases, start=1):

    # Create the messages for the agent
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=test_case["input"]),
    ]

    # Ask the agent for its decision
    decision = structured_llm.invoke(messages)

    # Compare the agent's category with the expected category
    category_match = (
        decision.category == test_case["expected_category"]
    )

    # Compare the agent's action with the expected action
    action_match = (
        decision.action == test_case["expected_action"]
    )

    # A test passes only when both values are correct
    test_passed = category_match and action_match

    # Display the test result
    print(f"\nTest Case {index}")
    print(f"Input: {test_case['input']}")
    print(f"Expected Category: {test_case['expected_category']}")
    print(f"Actual Category: {decision.category}")
    print(f"Expected Action: {test_case['expected_action']}")
    print(f"Actual Action: {decision.action}")

    # Update evaluation counters
    if test_passed:
        print("Result: PASS")
        passed += 1
    else:
        print("Result: FAIL")
        failed += 1


# Calculate overall accuracy
total = passed + failed
accuracy = (passed / total) * 100


# Display the final evaluation summary
print("\n====================")
print("Evaluation Summary")
print("====================")
print(f"Total Tests: {total}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")
print(f"Accuracy: {accuracy:.2f}%")