import mysql.connector
import ollama


# =========================================================
# STEP 1
# Connect to MySQL
# =========================================================

connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="retail_db"
)


cursor = connection.cursor(dictionary=True)


# =========================================================
# STEP 2
# Get available order information
#
# This represents the larger pool of information
# available to our application.
# =========================================================

cursor.execute("""
    SELECT
        order_id,
        order_date,
        order_customer_id,
        order_status
    FROM orders
""")

orders = cursor.fetchall()


print("========== ALL DATABASE CONTEXT ==========")

for order in orders:
    print(order)


# =========================================================
# STEP 3
# Context Selection
#
# The user only wants pending orders.
#
# Therefore we don't send every order to the LLM.
# We select only the relevant records.
# =========================================================

selected_orders = [
    {
        "order_id": order["order_id"],
        "order_status": order["order_status"]
    }
    for order in orders
    if order["order_status"] == "PENDING"
]


print("\n========== SELECTED CONTEXT ==========")

for order in selected_orders:
    print(order)


# =========================================================
# STEP 4
# Convert selected context into text
# =========================================================

context = "\n".join(
    str(order)
    for order in selected_orders
)


# =========================================================
# STEP 5
# Send ONLY the selected context to the LLM
# =========================================================

prompt = f"""
The following database records are relevant
to the user's question:

{context}

User question:

Which orders are currently pending?

Answer using only the provided records.
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


# =========================================================
# STEP 6
# Display answer
# =========================================================

print("\n========== LLM ANSWER ==========")

print(
    response["message"]["content"]
)


# =========================================================
# STEP 7
# Cleanup
# =========================================================

cursor.close()
connection.close()