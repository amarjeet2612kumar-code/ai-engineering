import json
import ollama


# =========================================================
# 1. Actual Python functions
# =========================================================
#
# These are the functions that our application can execute.
# =========================================================

def get_job_status(job_id):
    """
    Get the current status of a data engineering job.
    """

    print(f"\n🔧 Executing get_job_status({job_id})")

    return {
        "job_id": job_id,
        "status": "FAILED",
        "error": "ExecutorLostFailure"
    }


def get_kafka_lag(topic):
    """
    Get the current consumer lag for a Kafka topic.
    """

    print(f"\n🔧 Executing get_kafka_lag({topic})")

    return {
        "topic": topic,
        "consumer_lag": 1250
    }


# =========================================================
# 2. Tool definitions
# =========================================================

tools = [

    {
        "type": "function",

        "function": {
            "name": "get_job_status",

            "description": (
                "Get the current execution status "
                "of a data engineering job."
            ),

            "parameters": {
                "type": "object",

                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The ID of the job."
                    }
                },

                "required": ["job_id"]
            }
        }
    },

    {
        "type": "function",

        "function": {
            "name": "get_kafka_lag",

            "description": (
                "Get the current consumer lag "
                "for a Kafka topic."
            ),

            "parameters": {
                "type": "object",

                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Kafka topic name."
                    }
                },

                "required": ["topic"]
            }
        }
    }
]


# =========================================================
# 3. User question
# =========================================================

user_question = input(
    "\nAsk your DataOps question: "
)


# =========================================================
# 4. First LLM call
# =========================================================
#
# We give the LLM:
#
# - User question
# - Available tools
#
# The LLM decides whether it needs a tool.
# =========================================================

messages = [
    {
        "role": "user",
        "content": user_question
    }
]


response = ollama.chat(
    model="llama3.2:3b",

    messages=messages,

    tools=tools
)


# =========================================================
# 5. Get the assistant message
# =========================================================

assistant_message = response["message"]


# =========================================================
# 6. Check whether the LLM requested a tool
# =========================================================

if not assistant_message.get("tool_calls"):

    print("\n========== FINAL ANSWER ==========")

    print(assistant_message.get("content"))

    exit()


# =========================================================
# 7. Add the assistant message to the conversation
# =========================================================
#
# We keep the assistant's tool request in the conversation.
#
# This is important because the LLM needs to know:
#
# "I previously requested this tool."
# =========================================================

messages.append(assistant_message)


# =========================================================
# 8. Execute every requested tool
# =========================================================

for tool_call in assistant_message["tool_calls"]:

    # -----------------------------------------------------
    # Get function name
    # -----------------------------------------------------

    function_name = tool_call["function"]["name"]

    # -----------------------------------------------------
    # Get function arguments
    # -----------------------------------------------------

    arguments = tool_call["function"]["arguments"]

    print("\n========== TOOL CALL ==========")

    print("Function :", function_name)
    print("Arguments:", arguments)


    # =====================================================
    # 9. Execute the selected function
    # =====================================================

    if function_name == "get_job_status":

        result = get_job_status(
            arguments["job_id"]
        )


    elif function_name == "get_kafka_lag":

        result = get_kafka_lag(
            arguments["topic"]
        )


    else:

        result = {
            "error": f"Unknown tool: {function_name}"
        }


    # =====================================================
    # 10. Convert tool result to JSON
    # =====================================================

    result_json = json.dumps(result)


    print("\n========== TOOL RESULT ==========")

    print(result_json)


    # =====================================================
    # 11. Send tool result back to the LLM
    # =====================================================
    #
    # role="tool" tells the LLM:
    #
    # "This is the result produced by the tool
    # you requested."
    #
    # tool_name identifies which tool produced the result.
    # =====================================================

    messages.append(
        {
            "role": "tool",

            "content": result_json,

            "tool_name": function_name
        }
    )


# =========================================================
# 12. Second LLM call
# =========================================================
#
# Now the LLM has:
#
# - Original user question
# - Its own tool request
# - Tool result
#
# It can now generate the final answer.
# =========================================================

final_response = ollama.chat(
    model="llama3.2:3b",

    messages=messages,

    tools=tools
)


# =========================================================
# 13. Print final answer
# =========================================================

print("\n========== FINAL ANSWER ==========")

print(
    final_response["message"]["content"]
)