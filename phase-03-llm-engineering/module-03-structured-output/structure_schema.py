import ollama
import json

schema = {
    "type": "object",
    "properties": {
        "error_type": {
            "type": "string"
        },
        "root_cause": {
            "type": "string"
        },
        "priority": {
            "type": "string"
        }
    },
    "required": [
        "error_type",
        "root_cause",
        "priority"
    ]
}

response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": """
            Analyze this incident:

            Spark job failed because executor
            ran out of memory.
            """
        }
    ],
    format=schema
)

content = response["message"]["content"]

data = json.loads(content)

print(data)
print(data["error_type"])
print(data["root_cause"])
print(data["priority"])