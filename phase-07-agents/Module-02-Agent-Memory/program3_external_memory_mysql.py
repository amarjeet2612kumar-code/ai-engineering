# ============================================================
# Program 3: External Memory with MySQL
# Module 2: Agent Memory
# ============================================================
#
# Real-world scenario:
# Customer Support Agent
#
# In Program 2, conversation memory existed only inside
# a Python list.
#
# In this program, we store conversation messages in MySQL.
#
# This means the memory survives even after the Python
# program stops.
#
# Architecture:
#
# Customer
#    ↓
# Agent
#    ↓
# MySQL
#    ↓
# Retrieve previous conversation
#    ↓
# Ollama
#    ↓
# Agent response
# ============================================================


# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

# os is used for environment variables.
import os

# Ollama Python library.
import ollama

# MySQL Python connector.
import mysql.connector

# Load variables from .env.
from dotenv import load_dotenv


# ------------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# ------------------------------------------------------------

load_dotenv()


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

# Local Ollama model.
MODEL = "llama3.2:3b"


# MySQL configuration.
#
# Change these values if your MySQL configuration is
# different.
#

MYSQL_HOST = os.getenv(
    "MYSQL_HOST",
    "localhost"
)

MYSQL_PORT = int(
    os.getenv(
        "MYSQL_PORT",
        "3306"
    )
)

MYSQL_USER = os.getenv(
    "MYSQL_USER",
    "root"
)

MYSQL_PASSWORD = os.getenv(
    "MYSQL_PASSWORD",
    ""
)

MYSQL_DATABASE = "agent_memory"


# ============================================================
# FUNCTION: CONNECT TO MYSQL
# ============================================================

def get_mysql_connection():

    # Create a connection to MySQL.
    connection = mysql.connector.connect(

        host=MYSQL_HOST,

        port=MYSQL_PORT,

        user=MYSQL_USER,

        password=MYSQL_PASSWORD,

        database=MYSQL_DATABASE
    )

    return connection


# ============================================================
# FUNCTION: SAVE MEMORY
# ============================================================

def save_memory(
    customer_id,
    role,
    message
):

    # Connect to MySQL.
    connection = get_mysql_connection()

    # Create a cursor.
    cursor = connection.cursor()


    # SQL statement used to insert the message.
    sql = """
        INSERT INTO conversation_memory
        (customer_id, role, message)
        VALUES (%s, %s, %s)
    """


    # Values that will be inserted.
    values = (
        customer_id,
        role,
        message
    )


    # Execute INSERT.
    cursor.execute(
        sql,
        values
    )


    # Save the transaction.
    connection.commit()


    # Close cursor.
    cursor.close()

    # Close database connection.
    connection.close()


# ============================================================
# FUNCTION: LOAD MEMORY
# ============================================================

def load_memory(customer_id):

    # Connect to MySQL.
    connection = get_mysql_connection()

    # Create cursor.
    cursor = connection.cursor()


    # Retrieve previous messages for this customer.
    sql = """
        SELECT role, message
        FROM conversation_memory
        WHERE customer_id = %s
        ORDER BY created_at, id
    """


    # Execute SELECT.
    cursor.execute(
        sql,
        (customer_id,)
    )


    # Get all records.
    rows = cursor.fetchall()


    # Close cursor.
    cursor.close()

    # Close connection.
    connection.close()


    # Convert database rows into a Python list.
    memory = []

    for row in rows:

        memory.append(
            {
                "role": row[0],
                "content": row[1]
            }
        )


    return memory


# ============================================================
# FUNCTION: ASK OLLAMA
# ============================================================

def ask_llm(
    conversation_memory,
    user_message
):

    # --------------------------------------------------------
    # Build conversation context.
    # --------------------------------------------------------

    conversation_text = ""


    # Add previous memories.
    for message in conversation_memory:

        conversation_text += (
            f'{message["role"]}: '
            f'{message["content"]}\n'
        )


    # Add current message.
    conversation_text += (
        f"customer: {user_message}\n"
    )


    # --------------------------------------------------------
    # Create LLM prompt.
    # --------------------------------------------------------

    prompt = f"""
You are a customer support agent.

Use the previous customer conversation to answer
the latest customer message.

Previous conversation:

{conversation_text}

Rules:

1. Use information from previous conversation.
2. If the customer refers to an earlier problem,
   use the stored conversation to understand it.
3. Do not ask the customer to repeat information
   that is already available.
4. Keep the response concise.
"""


    # --------------------------------------------------------
    # Call Ollama.
    # --------------------------------------------------------

    response = ollama.chat(

        model=MODEL,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )


    # Return the generated response.
    return response["message"]["content"].strip()


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("EXTERNAL MEMORY - MYSQL")
    print("=" * 60)


    # --------------------------------------------------------
    # Customer ID
    # --------------------------------------------------------
    #
    # In a real application this would normally come from
    # the authenticated customer/session.
    #

    customer_id = "CUST1001"


    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    user_message = (
        "My account reference is CUST1001."
    )


    print("\nCustomer:")
    print(user_message)


    # Save customer message in MySQL.
    save_memory(
        customer_id,
        "customer",
        user_message
    )


    # Load existing memory.
    memory = load_memory(
        customer_id
    )


    # Generate response.
    agent_response = ask_llm(
        memory,
        user_message
    )


    print("\nAgent:")
    print(agent_response)


    # Save agent response in MySQL.
    save_memory(
        customer_id,
        "agent",
        agent_response
    )


    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    user_message = (
        "I have a problem with my payment."
    )


    print("\nCustomer:")
    print(user_message)


    # Save customer message.
    save_memory(
        customer_id,
        "customer",
        user_message
    )


    # Load memory again.
    memory = load_memory(
        customer_id
    )


    # Ask LLM using stored memory.
    agent_response = ask_llm(
        memory,
        user_message
    )


    print("\nAgent:")
    print(agent_response)


    # Save agent response.
    save_memory(
        customer_id,
        "agent",
        agent_response
    )


    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------
    #
    # Simulate a new request.
    #
    # The important point:
    #
    # We retrieve the conversation from MySQL.
    #

    print("\n")
    print("=" * 60)
    print("RETRIEVING MEMORY FROM MYSQL")
    print("=" * 60)


    memory = load_memory(
        customer_id
    )


    # Display stored memory.
    for message in memory:

        print(
            f'{message["role"]}: '
            f'{message["content"]}'
        )


    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------
    #
    # Ask a question that depends on previous conversation.
    #

    user_message = (
        "Do you remember what problem I had?"
    )


    print("\nCustomer:")
    print(user_message)


    # Ask LLM using MySQL memory.
    agent_response = ask_llm(
        memory,
        user_message
    )


    print("\nAgent:")
    print(agent_response)


    # Save final interaction.
    save_memory(
        customer_id,
        "customer",
        user_message
    )

    save_memory(
        customer_id,
        "agent",
        agent_response
    )


    print("\n")
    print("=" * 60)
    print("PROGRAM COMPLETED")
    print("=" * 60)