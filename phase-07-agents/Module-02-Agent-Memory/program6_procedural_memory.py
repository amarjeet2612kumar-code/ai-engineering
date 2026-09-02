# ============================================================
# Program 6: Procedural Memory
# Module 2: Agent Memory
# ============================================================
#
# Real-world scenario:
# Banking Customer Support Agent
#
# Procedural memory stores HOW a task should be performed.
#
# Example:
#
# Payment Dispute Procedure
#
# 1. Verify customer identity
# 2. Find transaction
# 3. Check transaction status
# 4. Check dispute eligibility
# 5. Create dispute
# 6. Notify customer
#
# The agent retrieves the correct procedure based on the
# customer's request.
#
# Architecture:
#
# User Request
#      ↓
# Embedding
#      ↓
# ChromaDB
#      ↓
# Retrieve relevant procedure
#      ↓
# Ollama
#      ↓
# Explain the procedure / next steps
# ============================================================


# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

# Ollama is used for:
# 1. Embeddings
# 2. LLM response
import ollama

# ChromaDB is our vector database.
import chromadb


# ============================================================
# CONFIGURATION
# ============================================================

# Local LLM.
MODEL = "llama3.2:3b"

# Local embedding model.
EMBEDDING_MODEL = "nomic-embed-text"


# ============================================================
# CREATE CHROMADB CLIENT
# ============================================================

# Store the database on disk.
#
# PersistentClient means the data survives after the
# Python program stops.
#
client = chromadb.PersistentClient(
    path="./chroma_procedural_memory"
)


# ============================================================
# CREATE PROCEDURAL MEMORY COLLECTION
# ============================================================

# A separate collection is used for procedures.
#
# This keeps procedural memory logically separate from
# semantic customer memory.
#
collection = client.get_or_create_collection(
    name="agent_procedures"
)


# ============================================================
# FUNCTION: CREATE EMBEDDING
# ============================================================

def create_embedding(text):

    # Convert text into a vector using Ollama.
    response = ollama.embeddings(

        model=EMBEDDING_MODEL,

        prompt=text
    )

    # Return the vector.
    return response["embedding"]


# ============================================================
# FUNCTION: STORE PROCEDURE
# ============================================================

def store_procedure(
    procedure_id,
    procedure_name,
    procedure_text
):

    # Create an embedding for the procedure.
    embedding = create_embedding(
        procedure_text
    )


    # Store procedure in ChromaDB.
    collection.add(

        ids=[procedure_id],

        embeddings=[embedding],

        documents=[procedure_text],

        metadatas=[
            {
                "procedure_name": procedure_name
            }
        ]
    )


# ============================================================
# FUNCTION: SEARCH PROCEDURE
# ============================================================

def search_procedure(
    user_request
):

    # Convert user request into an embedding.
    query_embedding = create_embedding(
        user_request
    )


    # Search for the most relevant procedure.
    results = collection.query(

        query_embeddings=[query_embedding],

        n_results=1
    )


    return results


# ============================================================
# FUNCTION: ASK LLM
# ============================================================

def ask_llm(
    user_request,
    procedure
):

    # Build the prompt using the retrieved procedure.
    prompt = f"""
You are a banking customer support agent.

The customer has made this request:

{user_request}

The following procedure was retrieved from the
bank's procedural memory:

{procedure}

Explain what steps should be followed.

Rules:

1. Follow the retrieved procedure.
2. Do not invent additional banking procedures.
3. Do not claim that an action was actually completed.
4. Clearly explain the next steps.
5. Keep the answer concise.
"""


    # Send the prompt to Ollama.
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
    print("PROCEDURAL MEMORY DEMO")
    print("=" * 60)


    # --------------------------------------------------------
    # PROCEDURE 1
    # --------------------------------------------------------

    store_procedure(

        procedure_id="procedure_001",

        procedure_name="Payment Dispute",

        procedure_text="""
Payment Dispute Procedure:

1. Verify the customer's identity.
2. Find the customer's transaction.
3. Check the transaction status.
4. Check whether the transaction is eligible
   for a dispute.
5. Create a dispute if the transaction is eligible.
6. Notify the customer about the dispute status.
"""
    )


    # --------------------------------------------------------
    # PROCEDURE 2
    # --------------------------------------------------------

    store_procedure(

        procedure_id="procedure_002",

        procedure_name="Card Block",

        procedure_text="""
Card Block Procedure:

1. Verify the customer's identity.
2. Confirm that the customer wants to block the card.
3. Block the card.
4. Confirm that the card has been blocked.
5. Explain the replacement-card process.
"""
    )


    # --------------------------------------------------------
    # PROCEDURE 3
    # --------------------------------------------------------

    store_procedure(

        procedure_id="procedure_003",

        procedure_name="KYC Update",

        procedure_text="""
KYC Update Procedure:

1. Verify the customer's identity.
2. Collect the required KYC information.
3. Validate the submitted information.
4. Update the customer record.
5. Confirm the KYC update status.
"""
    )


    print("\nProcedural memories stored.")


    # --------------------------------------------------------
    # CUSTOMER REQUEST
    # --------------------------------------------------------

    user_request = (
        "I want to dispute a payment that I made."
    )


    print("\n")
    print("Customer:")
    print(user_request)


    # --------------------------------------------------------
    # SEARCH PROCEDURAL MEMORY
    # --------------------------------------------------------

    results = search_procedure(
        user_request
    )


    # --------------------------------------------------------
    # GET RETRIEVED PROCEDURE
    # --------------------------------------------------------

    retrieved_procedure = (
        results["documents"][0][0]
    )


    # --------------------------------------------------------
    # DISPLAY RETRIEVED PROCEDURE
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("RETRIEVED PROCEDURAL MEMORY")
    print("=" * 60)

    print(
        retrieved_procedure
    )


    # --------------------------------------------------------
    # ASK LLM TO EXPLAIN NEXT STEPS
    # --------------------------------------------------------

    answer = ask_llm(

        user_request,

        retrieved_procedure
    )


    # --------------------------------------------------------
    # DISPLAY FINAL ANSWER
    # --------------------------------------------------------

    print("\n")
    print("Agent:")
    print(answer)


    print("\n")
    print("=" * 60)
    print("PROGRAM COMPLETED")
    print("=" * 60)