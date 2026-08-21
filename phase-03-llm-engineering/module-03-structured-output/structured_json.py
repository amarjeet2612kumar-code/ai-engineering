import json
import ollama

response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": """
            Analyze this incident:

            "Spark job failed because executor ran out of memory."

            Return JSON containing:
            - error_type
            - root_cause
            - priority
            """
        }
    ],
    format="json"
)

content = response["message"]["content"]

data = json.loads(content)

print(data)
print(type(data))
print(data["error_type"])
print(data["root_cause"])
print(data["priority"])