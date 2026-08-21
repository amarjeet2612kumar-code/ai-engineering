import ollama


def select_model(task):
    simple_tasks = [
        "classification",
        "simple",
        "summarization"
    ]

    if any(word in task.lower() for word in simple_tasks):
        return "llama3.2:3b"

    return "qwen2.5:7b"


tasks = [
    ("simple", "Explain Spark partitioning in 3 sentences."),
    ("complex", "Analyze this Spark failure and explain the likely root cause.")
]


for task_type, prompt in tasks:

    model = select_model(task_type)

    print(f"\nTask: {task_type}")
    print(f"Selected model: {model}")

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print(response["message"]["content"])