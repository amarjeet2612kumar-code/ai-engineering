import ollama


# =========================================================
# Simulated DataOps tools
#
# These functions represent tools that a real DataOps
# Copilot could call.
# =========================================================

def get_job_status(job_id):
    """
    Return the current status of a job.
    """

    return {
        "job_id": job_id,
        "status": "FAILED"
    }


def get_job_error(job_id):
    """
    Return the error associated with the failed job.
    """

    return {
        "job_id": job_id,
        "error": "ExecutorLostFailure: executor exceeded memory limit"
    }


def get_spark_metrics(job_id):
    """
    Return Spark metrics for the job.
    """

    return {
        "job_id": job_id,
        "executor_memory_utilization": "96%",
        "memory_limit": "16 GB"
    }


# =========================================================
# Tool dispatcher
#
# The LLM decides which tool should be used.
# Our Python program executes it.
# =========================================================

def execute_tool(tool_name, job_id):

    if tool_name == "get_job_status":
        return get_job_status(job_id)

    elif tool_name == "get_job_error":
        return get_job_error(job_id)

    elif tool_name == "get_spark_metrics":
        return get_spark_metrics(job_id)

    else:
        return {
            "error": f"Unknown tool: {tool_name}"
        }


# =========================================================
# Initial user question
# =========================================================

job_id = "CRM360-123"

question = f"""
Investigate why Spark job {job_id} failed.

You have access to these tools:

1. get_job_status
2. get_job_error
3. get_spark_metrics

Follow a ReAct-style investigation:

Reason about what information you need.
Then choose an action.
Use the observation to decide the next action.

When you have enough evidence, provide the final diagnosis.
"""


# =========================================================
# Initial LLM call
# =========================================================

response = ollama.chat(
    model="llama3.2:3b",

    messages=[
        {
            "role": "user",
            "content": question
        }
    ]
)


print("========== INITIAL LLM RESPONSE ==========")

print(
    response["message"]["content"]
)