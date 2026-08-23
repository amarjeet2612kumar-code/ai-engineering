import ollama


# =========================================================
# Base question
# =========================================================

question = """
Explain what Apache Spark is.
"""


# =========================================================
# Different amounts of context
# =========================================================

contexts = [

    "Spark is a distributed data processing framework.",

    """
    Spark is a distributed data processing framework.
    It supports batch processing, streaming, SQL,
    machine learning, and graph processing.
    """,

    """
    Spark is a distributed data processing framework.

    Spark provides APIs for Python, Scala, Java, and R.

    It supports batch processing, streaming, SQL,
    machine learning, and graph processing.

    Spark applications run using drivers and executors.
    Executors perform tasks on worker nodes.

    Spark can read data from HDFS, S3, Kafka,
    databases, and other storage systems.
    """
]


# =========================================================
# Run each context size
# =========================================================

for index, context in enumerate(contexts, start=1):

    prompt = f"""
Context:

{context}

Question:

{question}
"""


    response = ollama.chat(
        model="llama3.2:3b",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )


    print(
        f"\n========== EXPERIMENT {index} =========="
    )

    print(
        "Input tokens:",
        response.get("prompt_eval_count")
    )

    print(
        "Output tokens:",
        response.get("eval_count")
    )

    print(
        "\nAnswer:"
    )

    print(
        response["message"]["content"]
    )