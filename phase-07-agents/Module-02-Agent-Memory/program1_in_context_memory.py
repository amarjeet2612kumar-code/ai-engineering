# ============================================================
# Program 1: In-Context Memory
# Module 2: Agent Memory
# ============================================================
#
# Real-world scenario:
# Customer Support Agent
#
# The customer gives the order number in an earlier message.
# Later, the customer asks about the order without repeating
# the order number.
#
# The agent should remember the order number from the
# conversation history.
#
# This is called IN-CONTEXT MEMORY.
#
# No MySQL
# No Redis
# No Vector Database
#
# The memory exists inside the current conversation context.
# ============================================================


# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

# os is used to read environment variables.
import os


# load_dotenv loads variables from the .env file.
from dotenv import load_dotenv


# Gemini SDK.
from google import genai


# ------------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# ------------------------------------------------------------

# Load the .env file.
load_dotenv()


# Read Gemini API key.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# Make sure the API key exists.
if not GEMINI_API_KEY:

    raise ValueError(
        "GEMINI_API_KEY is not set in the .env file."
    )


# ------------------------------------------------------------
# GEMINI CONFIGURATION
# ------------------------------------------------------------

# Create Gemini client.
client = genai.Client(
    api_key=GEMINI_API_KEY
)


# Model used for our practical.
MODEL = "gemini-3.7-flash"


# ============================================================
# FUNCTION: ASK GEMINI
# ============================================================

def ask_llm(conversation):

    # --------------------------------------------------------
    # Build a prompt using the complete conversation.
    # --------------------------------------------------------
    #
    # The conversation itself acts as our memory.
    #
    # We are NOT storing this information in MySQL.
    # We are simply giving the previous messages to the LLM.
    #

    prompt = f"""
You are a customer support agent.

Use the conversation history to answer the
customer's latest question.

Conversation history:
{conversation}

Important:
- Use information from previous messages when necessary.
- Do not ask the customer to repeat information
  that already exists in the conversation.
- Give a concise answer.
"""


    # Send the conversation to Gemini.
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )


    # Return the LLM response.
    return response.text.strip()


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("IN-CONTEXT MEMORY DEMO")
    print("=" * 60)


    # --------------------------------------------------------
    # Conversation history
    # --------------------------------------------------------
    #
    # This is our temporary memory.
    #
    # The customer gave the order number earlier.
    #

    conversation = """

Customer:
My order number is 1025.

Agent:
Got it. What problem are you facing?

Customer:
The package is delayed.

Customer:
Can you check my order?
"""


    # Display the conversation.
    print("\nConversation:")
    print(conversation)


    # --------------------------------------------------------
    # Ask the LLM to answer the latest question.
    # --------------------------------------------------------

    answer = ask_llm(
        conversation
    )


    # --------------------------------------------------------
    # Display the answer.
    # --------------------------------------------------------

    print("\nAgent Response:")
    print(answer)