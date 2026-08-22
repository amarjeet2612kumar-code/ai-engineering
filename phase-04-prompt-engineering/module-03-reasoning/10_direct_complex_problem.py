import ollama


problem = """
We need to migrate a production data pipeline from
an on-prem Hadoop environment to AWS.

The pipeline uses:

- HDFS
- Hive
- Spark
- Kafka

Design a practical migration plan covering:

1. Data migration
2. Spark migration
3. Hive migration
4. Kafka migration
5. Validation
6. Cutover

Provide the final migration plan.
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


print("========== DIRECT APPROACH ==========")

print(response["message"]["content"])