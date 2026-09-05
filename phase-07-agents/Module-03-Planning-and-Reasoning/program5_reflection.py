# Import the Ollama chat function.
from ollama import chat

# Define the local Ollama model.
MODEL = "llama3.2:3b"

# Define the customer question.
question = "Why did payment transaction 1001 fail and what should I do?"

# Define the information discovered during the investigation.
observations = [
    "Payment status is FAILED.",
    "Payment failure reason is INSUFFICIENT_FUNDS."
]

# Define an intentionally incomplete first answer.
initial_answer = "The payment failed."

# Print the program title.
print("\nREFLECTION DEMO")

# Print the customer question.
print("\nCUSTOMER QUESTION:")
print(question)

# Print the collected observations.
print("\nOBSERVATIONS:")
for observation in observations:
    print(f"- {observation}")

# Print the initial answer.
print("\nINITIAL ANSWER:")
print(initial_answer)

# Create the reflection prompt.
reflection_prompt = f"""
You are reviewing an AI agent's answer.

Customer question:
{question}

Available observations:
{observations}

Agent answer:
{initial_answer}

Check whether the answer completely answers the customer question.

The answer must:
1. Explain why the payment failed.
2. Tell the customer what to do next.

Return exactly:
PASS
or
IMPROVE

Then give one short reason.
"""

# Ask the LLM to review the answer.
reflection_response = chat(
    model=MODEL,
    messages=[
        {"role": "user", "content": reflection_prompt}
    ]
)

# Extract the reflection result.
reflection = reflection_response["message"]["content"]

# Print the reflection result.
print("\nREFLECTION:")
print(reflection)

# Check whether the reflection says the answer needs improvement.
if "IMPROVE" in reflection.upper():

    # Create a prompt to improve the answer.
    improvement_prompt = f"""
Improve the following answer.

Customer question:
{question}

Observations:
{observations}

Current answer:
{initial_answer}

Give a short and clear answer that:
1. Explains why the payment failed.
2. Tells the customer what to do next.
"""

    # Ask the LLM to improve the answer.
    improvement_response = chat(
        model=MODEL,
        messages=[
            {"role": "user", "content": improvement_prompt}
        ]
    )

    # Extract the improved answer.
    final_answer = improvement_response["message"]["content"]

# Use the original answer when reflection passes.
else:

    # Keep the original answer because it passed reflection.
    final_answer = initial_answer

# Print the final answer.
print("\nFINAL ANSWER:")
print(final_answer)