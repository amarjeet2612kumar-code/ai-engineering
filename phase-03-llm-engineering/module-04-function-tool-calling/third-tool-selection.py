import ollama


# =========================================================
# 1. Define the tools
# =========================================================
#
# These are the capabilities we make available to the LLM.
#
# IMPORTANT:
# These definitions tell the LLM WHAT the tools do.
#
# The actual Python implementations come later.
# =========================================================

tools = [

    # -----------------------------------------------------
    # Tool 1: Get job status
    # -----------------------------------------------------

    {
        "type": "function",

        "function": {
            "name": "get_job_status",

            "description": (
                "Get the current execution status of a "
                "data engineering job."
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


    # -----------------------------------------------------
    # Tool 2: Get job logs
    # -----------------------------------------------------

    {
        "type": "function",

        "function": {
            "name": "get_job_logs",

            "description": (
                "Retrieve logs from a data engineering job "
                "to investigate errors or failures."
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


    # -----------------------------------------------------
    # Tool 3: Get Kafka lag
    # -----------------------------------------------------

    {
        "type": "function",

        "function": {
            "name": "get_kafka_lag",

            "description": (
                "Get the current consumer lag for a Kafka topic."
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
# 2. Ask the user question
# =========================================================

user_question = input("\nAsk your DataOps question: ")


# =========================================================
# 3. Send question + available tools to the LLM
# =========================================================

response = ollama.chat(

    # Use our local model
    model="llama3.2:3b",

    messages=[
        {
            "role": "user",
            "content": user_question
        }
    ],

    # Give all available tools to the LLM
    tools=tools
)


# =========================================================
# 4. Get the LLM message
# =========================================================

message = response["message"]


# =========================================================
# 5. Check whether the LLM selected a tool
# =========================================================

if message.get("tool_calls"):

    print("\n========== TOOL SELECTED ==========")

    for tool_call in message["tool_calls"]:

        # -------------------------------------------------
        # Get selected function name
        # -------------------------------------------------

        function_name = tool_call["function"]["name"]

        # -------------------------------------------------
        # Get arguments selected by the LLM
        # -------------------------------------------------

        arguments = tool_call["function"]["arguments"]

        print("Selected Tool :", function_name)
        print("Arguments     :", arguments)


else:

    print("\n========== NO TOOL SELECTED ==========")

    print("LLM response:")
    print(message.get("content"))