# ============================================================
# Program 4: Episodic Memory
# Module 2: Agent Memory
# ============================================================
#
# Real-world scenario:
# Banking Customer Support Agent
#
# The agent remembers important PAST EVENTS involving
# a customer.
#
# Example:
#
# Payment failed
# Customer contacted support
# Payment issue was resolved
#
# These are stored as episodes in MySQL.
#
# The agent can later retrieve those episodes and use them
# to answer a new customer question.
#
# IMPORTANT:
#
# This is different from conversation memory.
#
# Conversation memory:
#     Stores messages.
#
# Episodic memory:
#     Stores important events / experiences.
# ============================================================


# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

# os is used for environment variables.
import os

# Ollama local LLM.
import ollama

# MySQL connector.
import mysql.connector

# Load .env file.
from dotenv import load_dotenv


# ------------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# ------------------------------------------------------------

load_dotenv()


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

# Local LLM.
MODEL = "llama3.2:3b"


# MySQL configuration.
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
# FUNCTION: GET MYSQL CONNECTION
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
# FUNCTION: SAVE EPISODE
# ============================================================

def save_episode(
    customer_id,
    event_type,
    event_description,
    event_date
):

    # Connect to MySQL.
    connection = get_mysql_connection()

    # Create cursor.
    cursor = connection.cursor()


    # SQL INSERT statement.
    sql = """
        INSERT INTO customer_episodes
        (
            customer_id,
            event_type,
            event_description,
            event_date
        )
        VALUES (%s, %s, %s, %s)
    """


    # Values to insert.
    values = (
        customer_id,
        event_type,
        event_description,
        event_date
    )


    # Execute INSERT.
    cursor.execute(
        sql,
        values
    )


    # Commit the transaction.
    connection.commit()


    # Close cursor.
    cursor.close()

    # Close connection.
    connection.close()


# ============================================================
# FUNCTION: LOAD CUSTOMER EPISODES
# ============================================================

def load_episodes(customer_id):

    # Connect to MySQL.
    connection = get_mysql_connection()

    # Create cursor.
    cursor = connection.cursor()


    # Retrieve episodes for this customer.
    sql = """
        SELECT
            event_type,
            event_description,
            event_date
        FROM customer_episodes
        WHERE customer_id = %s
        ORDER BY event_date, id
    """


    # Execute query.
    cursor.execute(
        sql,
        (customer_id,)
    )


    # Retrieve all records.
    rows = cursor.fetchall()


    # Close database resources.
    cursor.close()

    connection.close()


    # Convert database rows into Python dictionaries.
    episodes = []

    for row in rows:

        episodes.append(
            {
                "event_type": row[0],
                "description": row[1],
                "date": str(row[2])
            }
        )


    return episodes


# ============================================================
# FUNCTION: ASK LLM
# ============================================================

def ask_llm(
    episodes,
    customer_question
):

    # --------------------------------------------------------
    # Build the episodic memory context.
    # --------------------------------------------------------

    episode_text = ""


    # Add every episode to the context.
    for episode in episodes:

        episode_text += (
            f'Date: {episode["date"]}\n'
            f'Event: {episode["event_type"]}\n'
            f'Description: {episode["description"]}\n\n'
        )


    # --------------------------------------------------------
    # Create the prompt.
    # --------------------------------------------------------

    prompt = f"""
You are a banking customer support agent.

The following information represents important
past events for the customer.

CUSTOMER EPISODIC MEMORY:

{episode_text}

Customer question:

{customer_question}

Rules:

1. Answer using only the information in the
   episodic memory.
2. Do not invent past events.
3. If there is no relevant event, say that no
   relevant event was found.
4. Keep the answer concise.
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


    # Return LLM response.
    return response["message"]["content"].strip()


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("EPISODIC MEMORY DEMO")
    print("=" * 60)


    # --------------------------------------------------------
    # Customer
    # --------------------------------------------------------

    customer_id = "CUST1001"


    # --------------------------------------------------------
    # Display customer.
    # --------------------------------------------------------

    print("\nCustomer ID:")
    print(customer_id)


    # --------------------------------------------------------
    # Load past episodes from MySQL.
    # --------------------------------------------------------

    episodes = load_episodes(
        customer_id
    )


    # --------------------------------------------------------
    # Display retrieved episodes.
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("RETRIEVED EPISODIC MEMORY")
    print("=" * 60)


    for episode in episodes:

        print(
            f'\nDate: {episode["date"]}'
        )

        print(
            f'Event: {episode["event_type"]}'
        )

        print(
            f'Description: '
            f'{episode["description"]}'
        )


    # --------------------------------------------------------
    # Customer asks a question about the past.
    # --------------------------------------------------------

    customer_question = (
        "Have I had any payment problems recently?"
    )


    print("\n")
    print("Customer:")
    print(customer_question)


    # --------------------------------------------------------
    # Ask the LLM using episodic memory.
    # --------------------------------------------------------

    answer = ask_llm(
        episodes,
        customer_question
    )


    # --------------------------------------------------------
    # Display answer.
    # --------------------------------------------------------

    print("\nAgent:")
    print(answer)


    print("\n")
    print("=" * 60)
    print("PROGRAM COMPLETED")
    print("=" * 60)