import ollama
from collections import Counter


problem = """
A data pipeline has four stages:

Extract
Transform
Validate
Load

Rules:

- Transform must happen after Extract.
- Validate must happen after Transform.
- Load must happen after Validate.

What is the correct execution order?

Give a concise reasoning summary and then provide
the final order.
"""


NUM_PATHS = 5

answers = []


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

    result = response["message"]["content"].strip()

    print(f"\n========== PATH {i + 1} ==========")
    print(result)

    answers.append(result)


print("\n========== ALL CANDIDATES ==========")

for answer in answers:
    print(answer)