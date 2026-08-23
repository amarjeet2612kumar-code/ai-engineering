import ollama


MODEL = "llama3.2:3b"


prompt = """
Analyze this Spark job:

The Spark job CRM360-123 completed successfully.

Return the status using exactly this format:

STATUS: SUCCESS

If the job failed, use:

STATUS: FAILED
"""


response = ollama.chat(

    model=MODEL,

    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


print(
    response["message"]["content"]
)