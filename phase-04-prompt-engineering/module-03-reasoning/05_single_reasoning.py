import ollama


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

print("========== SINGLE RESPONSE ==========")
print(answer)