# ============================================================
# Program 2: Conversation Memory
# Module 2: Agent Memory
# ============================================================
#
# Real-world scenario:
# Customer Support Agent
#
# The agent maintains conversation history and uses that
# history to answer future questions.
#
# Example:
#
# Customer:
# My order number is 1025.
#
# Customer:
# My package is delayed.
#
# Customer:
# Which order are we discussing?
#
# The agent should remember that the order is 1025.
#
# This is short-term / in-context conversation memory.
#
# LLM:
# Ollama - llama3.2:3b
#
# No external database is used in this program.
# ============================================================


# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

# Ollama Python library.
import ollama


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

# Local Ollama model.
MODEL = "llama3.2:3b"


# ============================================================
# FUNCTION: ASK LLM
# ============================================================

def ask_llm(conversation_history, user_message):

    # --------------------------------------------------------
    # Build the conversation text.
    # --------------------------------------------------------
    #
    # We take all previous messages from memory and combine
    # them with the current user message.
    #

    conversation_text = ""


    # Loop through previous messages.
    for message in conversation_history:

        conversation_text += (
            f'{message["role"]}: '
            f'{message["content"]}\n'
        )


    # Add the current user message.
    conversation_text += (
        f"customer: {user_message}\n"
    )


    # --------------------------------------------------------
    # Create the prompt.
    # --------------------------------------------------------

    prompt = f"""
You are a customer support agent.

Use the conversation history to answer the
customer's latest message.

Conversation history:

{conversation_text}

Rules:

1. Use information from previous messages.
2. Do not ask the customer to repeat information
   that already exists in the conversation.
3. Be concise and helpful.
4. If the customer asks which order we are discussing,
   identify the order number from the conversation.
"""


    # --------------------------------------------------------
    # Send the prompt to Ollama.
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


    # --------------------------------------------------------
    # Extract the LLM response.
    # --------------------------------------------------------

    return response["message"]["content"].strip()


# ============================================================
# FUNCTION: ADD MESSAGE TO MEMORY
# ============================================================

def add_message(
    conversation_history,
    role,
    content
):

    # Add the message to our conversation history.
    conversation_history.append(
        {
            "role": role,
            "content": content
        }
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CONVERSATION MEMORY DEMO - OLLAMA")
    print("=" * 60)


    # --------------------------------------------------------
    # Create empty conversation memory.
    # --------------------------------------------------------
    #
    # At the beginning there are no messages.
    #

    conversation_history = []


    # ========================================================
    # TURN 1
    # ========================================================

    user_message = (
        "My order number is 1025."
    )


    print("\nCustomer:")
    print(user_message)


    # Ask the LLM.
    agent_response = ask_llm(
        conversation_history,
        user_message
    )


    print("\nAgent:")
    print(agent_response)


    # --------------------------------------------------------
    # Save customer message in memory.
    # --------------------------------------------------------

    add_message(
        conversation_history,
        "customer",
        user_message
    )


    # --------------------------------------------------------
    # Save agent response in memory.
    # --------------------------------------------------------

    add_message(
        conversation_history,
        "agent",
        agent_response
    )


    # ========================================================
    # TURN 2
    # ========================================================

    user_message = (
        "The package was supposed to arrive yesterday."
    )


    print("\nCustomer:")
    print(user_message)


    # Ask the LLM using previous conversation history.
    agent_response = ask_llm(
        conversation_history,
        user_message
    )


    print("\nAgent:")
    print(agent_response)


    # Save customer message.
    add_message(
        conversation_history,
        "customer",
        user_message
    )


    # Save agent response.
    add_message(
        conversation_history,
        "agent",
        agent_response
    )


    # ========================================================
    # TURN 3
    # ========================================================

    user_message = (
        "Which order are we discussing?"
    )


    print("\nCustomer:")
    print(user_message)


    # Ask the LLM using the accumulated conversation.
    agent_response = ask_llm(
        conversation_history,
        user_message
    )


    print("\nAgent:")
    print(agent_response)


    # Save final customer message.
    add_message(
        conversation_history,
        "customer",
        user_message
    )


    # Save final agent response.
    add_message(
        conversation_history,
        "agent",
        agent_response
    )


    # ========================================================
    # DISPLAY MEMORY
    # ========================================================

    print("\n")
    print("=" * 60)
    print("FINAL CONVERSATION MEMORY")
    print("=" * 60)


    # Print everything currently stored in memory.
    for message in conversation_history:

        print(
            f'{message["role"]}: '
            f'{message["content"]}'
        )