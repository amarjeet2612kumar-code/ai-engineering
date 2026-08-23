import json

import mysql.connector
import ollama


# =========================================================
# CONFIGURATION
# =========================================================

MODEL = "llama3.2:3b"


# =========================================================
# TOOL FUNCTION
#
# Purpose:
# Query MySQL for orders having a specific status.
#
# The LLM will NOT execute this function directly.
# Our Python application executes it after receiving
# the tool call from the LLM.
# =========================================================

def get_orders_by_status(status: str):

    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="retail_db"
    )

    cursor = connection.cursor(dictionary=True)

    query = """
        SELECT
            order_id,
            order_date,
            order_customer_id,
            order_status
        FROM orders
        WHERE order_status = %s
    """

    cursor.execute(
        query,
        (status,)
    )

    rows = cursor.fetchall()

    cursor.close()
    connection.close()

    return rows


# =========================================================
# TOOL DEFINITION
#
# This tells the LLM what tool is available.
#
# We already learned this structure earlier.
# Here we focus on using it in the complete loop.
# =========================================================

tools = [

    {
        "type": "function",

        "function": {

            "name": "get_orders_by_status",

            "description": (
                "Get orders from the retail database "
                "filtered by order status."
            ),

            "parameters": {

                "type": "object",

                "properties": {

                    "status": {
                        "type": "string",
                        "description": (
                            "Order status such as "
                            "PENDING, COMPLETE, "
                            "CLOSED, or PROCESSING."
                        )
                    }

                },

                "required": [
                    "status"
                ]
            }
        }
    }
]


# =========================================================
# CONVERSATION
# =========================================================

messages = [

    {
        "role": "system",

        "content": """
You are a retail data assistant.

When the user asks about orders,
use the available database tool.

Do not invent database results.

After receiving the tool result,
answer the user using that result.
"""
    },

    {
        "role": "user",

        "content": (
            "Which orders are currently pending?"
        )
    }
]


# =========================================================
# FIRST LLM CALL
#
# The LLM sees:
# - system instructions
# - user question
# - available tools
#
# It can decide whether a tool is required.
# =========================================================

response = ollama.chat(

    model=MODEL,

    messages=messages,

    tools=tools
)


print("\n========================================")
print("FIRST LLM RESPONSE")
print("========================================")

print(response)


# =========================================================
# EXTRACT TOOL CALL
# =========================================================

tool_calls = response["message"].get(
    "tool_calls",
    []
)


if not tool_calls:

    print("\nLLM did not request a tool.")

    print(
        response["message"]["content"]
    )

    raise SystemExit


# =========================================================
# PROCESS TOOL CALL
# =========================================================

tool_call = tool_calls[0]

function_name = tool_call["function"]["name"]

arguments = tool_call["function"]["arguments"]


print("\n========================================")
print("TOOL REQUEST")
print("========================================")

print("Function:", function_name)

print("Arguments:", arguments)


# =========================================================
# EXECUTE THE FUNCTION
#
# IMPORTANT:
#
# The LLM requested the function.
# Python executes the function.
# =========================================================

if function_name == "get_orders_by_status":

    status = arguments["status"]

    tool_result = get_orders_by_status(
        status
    )

else:

    raise ValueError(
        f"Unknown function: {function_name}"
    )


print("\n========================================")
print("TOOL RESULT")
print("========================================")

print(tool_result)


# =========================================================
# ADD THE ASSISTANT TOOL-CALL MESSAGE
#
# This tells the LLM what tool call it previously made.
# =========================================================

messages.append(
    response["message"]
)


# =========================================================
# ADD TOOL RESULT
#
# Convert database result into JSON text.
# default=str handles MySQL date/datetime objects.
# =========================================================

messages.append(
    {
        "role": "tool",
        "content": json.dumps(
            tool_result,
            default=str
        )
    }
)


# =========================================================
# SECOND LLM CALL
#
# Now the LLM has:
#
# - original user question
# - its previous tool call
# - database result
#
# It can produce the final answer.
# =========================================================

final_response = ollama.chat(

    model=MODEL,

    messages=messages,

    tools=tools
)


print("\n========================================")
print("FINAL LLM RESPONSE")
print("========================================")

print(
    final_response["message"]["content"]
)