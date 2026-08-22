import ollama


# =========================================================
# Historical demonstrations
# =========================================================

examples = [
    {
        "input": "Spark executor ran out of memory",
        "output": "OUT_OF_MEMORY"
    },

    {
        "input": "Kafka consumer lag increased significantly",
        "output": "KAFKA_LAG"
    },

    {
        "input": "Airflow task exceeded execution timeout",
        "output": "TIMEOUT"
    },

    {
        "input": "Spark application failed due to permission denied",
        "output": "PERMISSION_ERROR"
    },

    {
        "input": "Spark executor exceeded configured memory limit",
        "output": "OUT_OF_MEMORY"
    },

    {
        "input": "Kafka consumer stopped processing messages",
        "output": "KAFKA_LAG"
    },

    {
        "input": "Database connection timed out",
        "output": "TIMEOUT"
    },

    {
        "input": "Spark executor was killed because of memory pressure",
        "output": "OUT_OF_MEMORY"
    }
]


# =========================================================
# New incident
# =========================================================

new_input = (
    "Spark executor was killed because it exceeded "
    "the configured memory limit"
)


# =========================================================
# Simple similarity function
# =========================================================

def calculate_similarity(text1, text2):
    """
    Calculate a very simple similarity score.

    We count how many words appear in both texts.

    This is NOT semantic similarity.
    It is only a learning example.
    """

    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    common_words = words1.intersection(words2)

    return len(common_words)


# =========================================================
# Calculate score for every example
# =========================================================

scored_examples = []

for example in examples:

    score = calculate_similarity(
        new_input,
        example["input"]
    )

    scored_examples.append(
        {
            "input": example["input"],
            "output": example["output"],
            "score": score
        }
    )


# =========================================================
# Sort examples by similarity
# =========================================================

scored_examples.sort(
    key=lambda x: x["score"],
    reverse=True
)


# =========================================================
# Display ranking
# =========================================================

print("\n========== EXAMPLE RANKING ==========")

for example in scored_examples:

    print(
        f"Score: {example['score']} | "
        f"{example['input']} "
        f"→ {example['output']}"
    )


# =========================================================
# Select Top-K examples
# =========================================================

TOP_K = 3

selected_examples = scored_examples[:TOP_K]


print("\n========== SELECTED EXAMPLES ==========")

for example in selected_examples:

    print(
        f"{example['input']} "
        f"→ {example['output']}"
    )


# =========================================================
# Build ICL prompt using selected examples
# =========================================================

prompt = """
Classify the DataOps incident into one of:

OUT_OF_MEMORY
KAFKA_LAG
TIMEOUT
PERMISSION_ERROR

Here are relevant examples:

"""


for index, example in enumerate(
    selected_examples,
    start=1
):

    prompt += f"""
Example {index}:

Input:
{example['input']}

Output:
{example['output']}

"""


prompt += f"""
Now classify this incident:

Input:
{new_input}

Output:
"""


# =========================================================
# Send selected examples to LLM
# =========================================================

response = ollama.chat(
    model="llama3.2:3b",

    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# =========================================================
# Final result
# =========================================================

print("\n========== LLM RESPONSE ==========")

print(
    response["message"]["content"]
)