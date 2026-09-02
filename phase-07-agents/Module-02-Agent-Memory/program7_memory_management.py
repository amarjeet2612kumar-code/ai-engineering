
# Overall architecture:
#
#              User Message
#                    |
#                    v
#              Memory Manager
#                    |
#                    v
#               Ollama LLM
#                    |
#          +---------+---------+
#          |         |         |
#          v         v         v
#       EPISODIC  SEMANTIC  PROCEDURAL
#          |         |         |
#          v         v         v
#        MySQL    ChromaDB   ChromaDB
#

# ============================================================
# Program 7: Memory Management
# Module 2: Agent Memory
# ============================================================
#
# Banking customer-support agent memory manager.
#
# Memory types:
#   EPISODIC   -> MySQL
#   SEMANTIC   -> ChromaDB
#   PROCEDURAL  -> ChromaDB
#   IGNORE     -> Nothing
#
# The application validates the LLM output and NEVER crashes
# just because the LLM returns malformed JSON or empty memory.
# ============================================================

import json
import os
import uuid

import chromadb
import mysql.connector
import ollama
from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = "agent_memory"


# ============================================================
# CHROMADB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path="./chroma_memory_management"
)

semantic_collection = chroma_client.get_or_create_collection(
    name="semantic_memories"
)

procedural_collection = chroma_client.get_or_create_collection(
    name="procedural_memories"
)


# ============================================================
# MYSQL DATABASE
# ============================================================

def create_mysql_database():
    """Create the memory database if it does not exist."""

    connection = mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
    )

    cursor = connection.cursor()

    cursor.execute(
        f"CREATE DATABASE IF NOT EXISTS {MYSQL_DATABASE}"
    )

    connection.commit()
    cursor.close()
    connection.close()


def create_mysql_table():
    """Create the episodic-memory table."""

    connection = mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS episodic_memories (
            id INT AUTO_INCREMENT PRIMARY KEY,
            memory TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    connection.commit()
    cursor.close()
    connection.close()


# ============================================================
# EMBEDDINGS
# ============================================================

def create_embedding(text):
    """Create an embedding using Ollama."""

    response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=text,
    )

    return response["embedding"]


# ============================================================
# LLM MEMORY CLASSIFICATION
# ============================================================

def classify_memory(message):
    """
    Ask the LLM to classify the message.

    Important:
    The LLM output is treated as untrusted data.
    Python validates everything before using it.
    """

    prompt = f"""
You are a memory classifier for a banking customer-support agent.

Classify the customer message into exactly one category:

EPISODIC
SEMANTIC
PROCEDURAL
IGNORE

Definitions:

EPISODIC:
A specific event that happened.
Example: "My payment failed yesterday."

SEMANTIC:
A persistent fact, preference, or recurring pattern.
Example: "I frequently have problems with international payments."

PROCEDURAL:
A rule or instruction describing how something should be done.
Example: "Always verify the customer before processing a payment dispute."

IGNORE:
Information that does not need long-term memory.
Example: "Thanks for your help."

Rules:
- For EPISODIC, SEMANTIC, and PROCEDURAL, memory must contain
  a concise version of the customer message.
- For IGNORE, memory must be an empty string.
- Return JSON only.
- Do not add extra fields.

Required JSON format:

{{
  "memory_type": "EPISODIC | SEMANTIC | PROCEDURAL | IGNORE",
  "memory": "..."
}}

Customer message:
{message}
"""

    # JSON schema makes the expected structure explicit.
    response_schema = {
        "type": "object",
        "properties": {
            "memory_type": {
                "type": "string",
                "enum": [
                    "EPISODIC",
                    "SEMANTIC",
                    "PROCEDURAL",
                    "IGNORE",
                ],
            },
            "memory": {
                "type": "string",
            },
        },
        "required": [
            "memory_type",
            "memory",
        ],
    }

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        format=response_schema,
    )

    raw_response = (
        response.get("message", {})
        .get("content", "")
        .strip()
    )

    print("\nRaw LLM Response:")
    print(raw_response)

    # --------------------------------------------------------
    # VALIDATION 1: JSON
    # --------------------------------------------------------

    try:
        decision = json.loads(raw_response)
    except json.JSONDecodeError:
        print("\nWARNING: Invalid JSON from LLM.")
        print("Using safe fallback.")

        return {
            "memory_type": "SEMANTIC",
            "memory": message.strip(),
        }

    # --------------------------------------------------------
    # VALIDATION 2: OBJECT
    # --------------------------------------------------------

    if not isinstance(decision, dict):
        print("\nWARNING: LLM did not return a JSON object.")
        print("Using safe fallback.")

        return {
            "memory_type": "SEMANTIC",
            "memory": message.strip(),
        }

    # --------------------------------------------------------
    # VALIDATION 3: MEMORY TYPE
    # --------------------------------------------------------

    memory_type = str(
        decision.get("memory_type", "")
    ).strip().upper()

    allowed_types = {
        "EPISODIC",
        "SEMANTIC",
        "PROCEDURAL",
        "IGNORE",
    }

    if memory_type not in allowed_types:
        print("\nWARNING: Invalid memory type.")
        print("Using safe fallback.")

        return {
            "memory_type": "SEMANTIC",
            "memory": message.strip(),
        }

    # --------------------------------------------------------
    # VALIDATION 4: MEMORY CONTENT
    # --------------------------------------------------------

    memory = decision.get("memory", "")

    if memory is None:
        memory = ""

    memory = str(memory).strip()

    # IGNORE is allowed to have empty memory.
    if memory_type == "IGNORE":
        return {
            "memory_type": "IGNORE",
            "memory": "",
        }

    # For all other types, memory MUST exist.
    #
    # If the LLM returns:
    #
    # {"memory_type": "SEMANTIC", "memory": ""}
    #
    # we do NOT crash.
    #
    # We use the original user message.
    if not memory:
        print("\nWARNING: LLM returned empty memory.")
        print("Using original user message as fallback.")

        memory = message.strip()

    # Final safety check.
    if not memory:
        print("\nWARNING: User message is also empty.")
        print("Ignoring memory.")

        return {
            "memory_type": "IGNORE",
            "memory": "",
        }

    return {
        "memory_type": memory_type,
        "memory": memory,
    }


# ============================================================
# MYSQL: STORE EPISODIC MEMORY
# ============================================================

def store_episodic_memory(memory):
    """Store an event in MySQL."""

    connection = mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO episodic_memories (memory)
        VALUES (%s)
        """,
        (memory,),
    )

    connection.commit()
    cursor.close()
    connection.close()


# ============================================================
# CHROMADB: STORE SEMANTIC / PROCEDURAL MEMORY
# ============================================================

def store_vector_memory(collection, memory, memory_type):
    """Store memory as an embedding in ChromaDB."""

    embedding = create_embedding(memory)

    memory_id = str(uuid.uuid4())

    collection.add(
        ids=[memory_id],
        embeddings=[embedding],
        documents=[memory],
        metadatas=[
            {
                "memory_type": memory_type,
            }
        ],
    )


# ============================================================
# MEMORY ROUTER
# ============================================================

def store_memory(memory_type, memory):
    """
    Route memory to the correct persistence layer.
    """

    if memory_type == "EPISODIC":
        print("Persistence: MySQL")

        store_episodic_memory(memory)

    elif memory_type == "SEMANTIC":
        print("Persistence: ChromaDB")

        store_vector_memory(
            semantic_collection,
            memory,
            "SEMANTIC",
        )

    elif memory_type == "PROCEDURAL":
        print("Persistence: ChromaDB")

        store_vector_memory(
            procedural_collection,
            memory,
            "PROCEDURAL",
        )

    elif memory_type == "IGNORE":
        print("Persistence: None")
        print("Memory ignored.")


# ============================================================
# PROCESS ONE USER MESSAGE
# ============================================================

def process_message(message):
    """Run one message through the complete memory pipeline."""

    print("\n" + "=" * 60)
    print("USER MESSAGE")
    print("=" * 60)
    print(message)

    # 1. Ask LLM what type of memory this is.
    decision = classify_memory(message)

    print("\nMEMORY DECISION")
    print("=" * 60)

    print("Memory Type:", decision["memory_type"])
    print("Memory:", decision["memory"])

    # 2. Store the memory using the appropriate persistence.
    store_memory(
        decision["memory_type"],
        decision["memory"],
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MEMORY MANAGEMENT DEMO")
    print("=" * 60)

    # Prepare MySQL.
    print("\nPreparing MySQL...")

    create_mysql_database()
    create_mysql_table()

    print("MySQL is ready.")

    # --------------------------------------------------------
    # TEST 1: EPISODIC
    # --------------------------------------------------------

    process_message(
        "My payment failed yesterday."
    )

    # --------------------------------------------------------
    # TEST 2: SEMANTIC
    # --------------------------------------------------------

    process_message(
        "I frequently have problems with international payments."
    )

    # --------------------------------------------------------
    # TEST 3: PROCEDURAL
    # --------------------------------------------------------

    process_message(
        "Always verify the customer before processing a payment dispute."
    )

    # --------------------------------------------------------
    # TEST 4: IGNORE
    # --------------------------------------------------------

    process_message(
        "Thanks for your help."
    )

    print("\n" + "=" * 60)
    print("MEMORY MANAGEMENT COMPLETED")
    print("=" * 60)
