import ollama


# =========================================================
# Simulated tools
# =========================================================

def get_job_status(job_id):
    """
    Returns the current status of the job.
    """

    return {
        "job_id": job_id,
        "status": "FAILED"
    }


def get_job_error(job_id):
    """
    Returns the error associated with the job.
    """

    return {
        "job_id": job_id,
        "error": "ExecutorLostFailure: executor exceeded memory limit"
    }


def get_spark_metrics(job_id):
    """
    Returns Spark memory information.
    """

    return {
        "job_id": job_id,
        "executor_memory_utilization": "96%",
        "memory_limit": "16 GB"
    }


# =========================================================
# Tool dispatcher
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
            "error": "Unknown tool"
        }


# =========================================================
# Job being investigated
# =========================================================

job_id = "CRM360-123"


# =========================================================
# Conversation history
# =========================================================

messages = [
    {
        "role": "system",
        "content": """
You are a DataOps investigation agent.

You are investigating a failed Spark job.

Available tools:

- get_job_status
- get_job_error
- get_spark_metrics

After receiving a tool observation, decide what
information is still needed.

When you need another tool, return exactly:

ACTION: tool_name

When enough evidence is available, return:

FINAL: diagnosis
"""
    },

    {
        "role": "user",
        "content": f"""
Investigate why Spark job {job_id} failed.
"""
    }
]


# =========================================================
# ReAct loop
# =========================================================

MAX_ITERATIONS = 5


for step in range(MAX_ITERATIONS):

    print(f"\n========== STEP {step + 1} ==========")


    # -----------------------------------------------------
    # Ask LLM what to do next
    # -----------------------------------------------------

    response = ollama.chat(
        model="llama3.2:3b",
        messages=messages
    )

    content = response["message"]["content"].strip()

    print("LLM:")
    print(content)


    # -----------------------------------------------------
    # Check for final answer
    # -----------------------------------------------------

    if content.startswith("FINAL:"):

        final_answer = content.replace(
            "FINAL:",
            ""
        ).strip()

        print("\n========== FINAL RESULT ==========")
        print(final_answer)

        break


    # -----------------------------------------------------
    # Check for action
    # -----------------------------------------------------

    if content.startswith("ACTION:"):

        tool_name = content.replace(
            "ACTION:",
            ""
        ).strip()


        print(
            f"\nExecuting tool: {tool_name}"
        )


        # -------------------------------------------------
        # Execute tool
        # -------------------------------------------------

        observation = execute_tool(
            tool_name,
            job_id
        )


        print("Observation:")
        print(observation)


        # -------------------------------------------------
        # Add LLM action to conversation
        # -------------------------------------------------

        messages.append(
            {
                "role": "assistant",
                "content": content
            }
        )


        # -------------------------------------------------
        # Add tool observation
        # -------------------------------------------------

        messages.append(
            {
                "role": "user",
                "content": f"""
Observation from {tool_name}:

{observation}

Decide what to do next.
"""
            }
        )


    else:

        print("\nUnexpected LLM response.")

        break