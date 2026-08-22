import ollama


# =========================================================
# 1. Define the actual Python functions
# =========================================================
#
# These are the functions that our application can execute.
#
# IMPORTANT:
# The LLM does NOT execute these functions.
#
# The Python application executes them.
# =========================================================


def get_job_status(job_id):
    """
    Get the current status of a data engineering job.

    In a real application, this could call:
    - Airflow API
    - Spark
    - Database
    - Monitoring system
    """

    print(f"\n🔧 Executing get_job_status({job_id})")

    return {
        "job_id": job_id,
        "status": "FAILED",
        "error": "ExecutorLostFailure"
    }


def get_kafka_lag(topic):
    """
    Get the consumer lag for a Kafka topic.

    In a real application, this could query
    a Kafka monitoring system.
    """

    print(f"\n🔧 Executing get_kafka_lag({topic})")

    return {
        "topic": topic,
        "consumer_lag": 1250
    }


# =========================================================
# 2. Define the tools available to the LLM
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
# 3. Ask the user
# =========================================================

user_question = input(
    "\nAsk your DataOps question: "
)


# =========================================================
# 4. Send the request to the LLM
# =========================================================

response = ollama.chat(

    model="llama3.2:3b",

    messages=[
        {
            "role": "user",
            "content": user_question
        }
    ],

    tools=tools
)


# =========================================================
# 5. Get the LLM response
# =========================================================

message = response["message"]


# =========================================================
# 6. Check whether LLM requested a tool
# =========================================================

if not message.get("tool_calls"):

    print("\n========== NO TOOL CALL ==========")

    print(message.get("content"))

    exit()


# =========================================================
# 7. Process the tool call
# =========================================================

for tool_call in message["tool_calls"]:

    # -----------------------------------------------------
    # Get the function name selected by the LLM
    # -----------------------------------------------------

    function_name = tool_call["function"]["name"]

    print("\n========== TOOL SELECTED ==========")

    print("Function:", function_name)


    # -----------------------------------------------------
    # Get the arguments selected by the LLM
    # -----------------------------------------------------

    arguments = tool_call["function"]["arguments"]

    print("Arguments:", arguments)


    # =====================================================
    # 8. Execute the correct Python function
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

        print(
            f"❌ Unknown tool: {function_name}"
        )

        continue


    # =====================================================
    # 9. Print the result
    # =====================================================

    print("\n========== TOOL EXECUTED ==========")

    print("Result:")
    print(result)