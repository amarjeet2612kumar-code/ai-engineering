import json
import ollama


# =========================================================
# 1. Normal Python function
# =========================================================

def get_job_status(job_id):
    """
    Get the status of a data engineering job.

    In a real application, this function could query:
    - Airflow
    - Spark
    - Database
    - Monitoring system
    """

    print(f"\n🔧 Executing get_job_status() for: {job_id}")

    # For now, we are returning a hard-coded result.
    # Later this could come from a real system.
    return {
        "job_id": job_id,
        "status": "FAILED",
        "error": "ExecutorLostFailure"
    }


# =========================================================
# 2. Describe the function to the LLM
# =========================================================

tools = [
    {
        "type": "function",

        "function": {
            "name": "get_job_status",

            "description": (
                "Get the current status of a data engineering job."
            ),

            "parameters": {
                "type": "object",

                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The ID of the job to check."
                    }
                },

                "required": ["job_id"]
            }
        }
    }
]


# =========================================================
# 3. Send user request + tool definition to LLM
# =========================================================

response = ollama.chat(

    # Local model
    model="llama3.2:3b",

    # Conversation
    messages=[
        {
            "role": "user",
            "content": "Check the status of job CRM360-123."
        }
    ],

    # Give the LLM information about available tools
    tools=tools
)


# =========================================================
# 4. Extract the assistant message
# =========================================================

message = response["message"]

print("\n========== LLM RESPONSE ==========")

print(message)


# =========================================================
# 5. Check whether the LLM requested a tool
# =========================================================

if message.get("tool_calls"):

    print("\n========== TOOL CALL REQUESTED ==========")

    for tool_call in message["tool_calls"]:

        # Get function name
        function_name = tool_call["function"]["name"]

        # Get arguments
        arguments = tool_call["function"]["arguments"]

        print("Function :", function_name)
        print("Arguments:", arguments)


        # =================================================
        # 6. Execute the requested Python function
        # =================================================

        if function_name == "get_job_status":

            result = get_job_status(
                arguments["job_id"]
            )

            print("\n========== TOOL RESULT ==========")
            print(result)

else:

    print("\n========== NO TOOL CALL ==========")

    print("LLM returned a normal response.")

    print(message.get("content"))