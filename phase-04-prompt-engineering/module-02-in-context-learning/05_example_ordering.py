import ollama


# =========================================================
# Same demonstrations
# =========================================================

examples = {
    "A": {
        "input": "Spark executor ran out of memory.",
        "output": "OUT_OF_MEMORY"
    },

    "B": {
        "input": "Kafka consumer lag increased significantly.",
        "output": "KAFKA_LAG"
    },

    "C": {
        "input": "Airflow task exceeded the execution timeout.",
        "output": "TIMEOUT"
    }
}


# =========================================================
# New input
# =========================================================

new_input = (
    "Spark executor was killed because it exceeded "
    "the configured memory limit."
)


# =========================================================
# Function to create the prompt
# =========================================================

def build_prompt(order):
    """
    Build the ICL prompt using the supplied
    demonstration order.
    """

    prompt = """
Classify the following DataOps incident into:

OUT_OF_MEMORY
KAFKA_LAG
TIMEOUT

Use the examples to understand the classification pattern.

"""

    # Add demonstrations in the requested order
    for index, example_id in enumerate(order, start=1):

        example = examples[example_id]

        prompt += f"""
Example {index}:

Input:
{example["input"]}

Output:
{example["output"]}

"""

    # Add the new input
    prompt += f"""
Now classify this incident:

Input:
{new_input}

Output:
"""

    return prompt


# =========================================================
# Different demonstration orders
# =========================================================

orders = [
    ["A", "B", "C"],
    ["C", "B", "A"],
    ["B", "A", "C"]
]


# =========================================================
# Run each experiment
# =========================================================

for order in orders:

    print("\n")
    print("=" * 60)

    print(
        f"ORDER: {' → '.join(order)}"
    )

    print("=" * 60)


    # -----------------------------------------------------
    # Build prompt
    # -----------------------------------------------------

    prompt = build_prompt(order)


    # -----------------------------------------------------
    # Call LLM
    # -----------------------------------------------------

    response = ollama.chat(
        model="llama3.2:3b",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )


    # -----------------------------------------------------
    # Print result
    # -----------------------------------------------------

    print("\nLLM RESPONSE:")

    print(
        response["message"]["content"]
    )