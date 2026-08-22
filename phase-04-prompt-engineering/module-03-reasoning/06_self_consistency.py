import ollama
from collections import Counter


# =========================================================
# Problem
# =========================================================

problem = """
Four services — A, B, C and D — must be deployed
one at a time.

Rules:

1. A must be deployed before C.
2. B must be deployed before D.
3. C must be deployed after B.

Which option is a valid deployment order?

1. A → B → C → D
2. B → D → A → C
3. C → A → B → D
4. D → B → A → C

Analyze the constraints and return only the
option number.
"""


# =========================================================
# Number of reasoning paths
# =========================================================

NUM_PATHS = 5


answers = []


# =========================================================
# Generate multiple candidate answers
# =========================================================

for i in range(NUM_PATHS):

    response = ollama.chat(
        model="llama3.2:3b",

        messages=[
            {
                "role": "user",
                "content": problem
            }
        ]
    )

    answer = response["message"]["content"].strip()

    answers.append(answer)

    print(
        f"Path {i + 1}: {answer}"
    )


# =========================================================
# Count candidate answers
# =========================================================

counts = Counter(answers)


# =========================================================
# Select the most common answer
# =========================================================

final_answer = counts.most_common(1)[0][0]


print("\n========== SELF-CONSISTENCY ==========")

print("All answers:")
print(answers)

print("\nVote counts:")
print(counts)

print("\nFinal answer:")
print(final_answer)