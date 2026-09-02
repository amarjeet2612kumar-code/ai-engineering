# ============================================================
# Program 5: Semantic Memory
# Module 2: Agent Memory
# ============================================================
#
# Real-world scenario:
# Banking Customer Support Agent
#
# The agent stores facts/knowledge about a customer.
#
# Example:
#
# "Customer frequently experiences payment failures."
#
# Later the customer asks:
#
# "Why do my payments keep failing?"
#
# We use embeddings and vector similarity to find memories
# related to the customer's question.
#
# Architecture:
#
# Customer Question
#       ↓
# Embedding
#       ↓
# Query Vector
#       ↓
# ChromaDB
#       ↓
# Similarity Search
#       ↓
# Relevant Memories
#       ↓
# Ollama
#       ↓
# Final Answer
# ============================================================


# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

# Ollama is used for the local LLM.
import ollama

# ChromaDB is our vector database.
import chromadb


# ============================================================
# CONFIGURATION
# ============================================================

# Local Ollama model.
MODEL = "llama3.2:3b"

# Name of the embedding model.
#
# This model converts text into vectors.
#
# We use a small local embedding model so the entire
# example can run locally.
#
EMBEDDING_MODEL = "nomic-embed-text"


# ============================================================
# CREATE CHROMADB CLIENT
# ============================================================

# PersistentClient stores the vector database on disk.
#
# This means our semantic memories are not lost when
# the Python program stops.
#
client = chromadb.PersistentClient(
    path="./chroma_semantic_memory"
)


# ============================================================
# CREATE / GET COLLECTION
# ============================================================

# A Chroma collection is similar to a table in a
# traditional database.
#
collection = client.get_or_create_collection(
    name="customer_semantic_memory"
)


# ============================================================
# FUNCTION: CREATE EMBEDDING
# ============================================================

def create_embedding(text):

    # Send the text to the Ollama embedding model.
    response = ollama.embeddings(

        model=EMBEDDING_MODEL,

        prompt=text
    )

    # Return the generated vector.
    return response["embedding"]


# ============================================================
# FUNCTION: STORE SEMANTIC MEMORY
# ============================================================

def store_memory(
    memory_id,
    customer_id,
    memory_text
):

    # Create embedding for the memory.
    embedding = create_embedding(
        memory_text
    )


    # Store the memory and its vector in ChromaDB.
    collection.add(

        ids=[memory_id],

        embeddings=[embedding],

        documents=[memory_text],

        metadatas=[
            {
                "customer_id": customer_id
            }
        ]
    )


# ============================================================
# FUNCTION: SEARCH SEMANTIC MEMORY
# ============================================================

def search_memory(
    customer_id,
    question,
    number_of_results=2
):

    # Convert the customer's question into an embedding.
    query_embedding = create_embedding(
        question
    )


    # Search ChromaDB using vector similarity.
    results = collection.query(

        query_embeddings=[query_embedding],

        n_results=number_of_results,

        where={
            "customer_id": customer_id
        }
    )


    # Return search results.
    return results


# ============================================================
# FUNCTION: ASK OLLAMA
# ============================================================

def ask_llm(
    question,
    relevant_memories
):

    # Build the memory context.
    memory_context = ""


    # Add each retrieved memory to the context.
    for memory in relevant_memories:

        memory_context += (
            f"- {memory}\n"
        )


    # Create the prompt.
    prompt = f"""
You are a banking customer support agent.

You have retrieved the following semantic memories
about the customer:

{memory_context}

Customer question:

{question}

Rules:

1. Answer using only the retrieved memories.
2. Do not invent customer history.
3. If the memories do not contain enough information,
   clearly say that.
4. Keep the answer concise.
"""


    # Call the local Ollama model.
    response = ollama.chat(

        model=MODEL,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )


    # Return the generated answer.
    return response["message"]["content"].strip()


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SEMANTIC MEMORY DEMO")
    print("=" * 60)


    # --------------------------------------------------------
    # Customer
    # --------------------------------------------------------

    customer_id = "CUST1001"


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # These are facts/knowledge about the customer.
    #
    # They are NOT individual conversation messages.
    # --------------------------------------------------------

    memories = [

        (
            "memory_001",
            "Customer frequently experiences "
            "payment failures."
        ),

        (
            "memory_002",
            "Customer prefers email notifications "
            "for banking transactions."
        ),

        (
            "memory_003",
            "Customer had an international transaction "
            "declined previously."
        ),

        (
            "memory_004",
            "Customer completed KYC verification "
            "successfully."
        ),

        (
            "memory_005",
            "Customer prefers electronic account "
            "statements."
        )
    ]


    # --------------------------------------------------------
    # Store memories in ChromaDB.
    # --------------------------------------------------------

    print("\nStoring semantic memories...")


    for memory_id, memory_text in memories:

        store_memory(

            memory_id,

            customer_id,

            memory_text
        )


    print("Semantic memories stored.")


    # --------------------------------------------------------
    # Customer asks a question.
    # --------------------------------------------------------

    question = (
        "Why do my card payments keep failing?"
    )


    print("\n")
    print("Customer:")
    print(question)


    # --------------------------------------------------------
    # Search semantic memory.
    # --------------------------------------------------------

    results = search_memory(

        customer_id,

        question,

        number_of_results=2
    )


    # --------------------------------------------------------
    # Extract retrieved memories.
    # --------------------------------------------------------

    retrieved_memories = results["documents"][0]


    # --------------------------------------------------------
    # Display retrieved memories.
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("RETRIEVED SEMANTIC MEMORIES")
    print("=" * 60)


    for memory in retrieved_memories:

        print(
            f"\n- {memory}"
        )


    # --------------------------------------------------------
    # Send relevant memories to the LLM.
    # --------------------------------------------------------

    answer = ask_llm(

        question,

        retrieved_memories
    )


    # --------------------------------------------------------
    # Display final answer.
    # --------------------------------------------------------

    print("\n")
    print("Agent:")
    print(answer)


    print("\n")
    print("=" * 60)
    print("PROGRAM COMPLETED")
    print("=" * 60)