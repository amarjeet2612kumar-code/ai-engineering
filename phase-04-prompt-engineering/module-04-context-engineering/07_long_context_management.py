import ollama


MODEL = "llama3.2:3b"

# =========================================================
# How many recent messages should we keep directly?
#
# This is a simple educational threshold.
# Production systems should usually consider token count.
# =========================================================

MAX_RECENT_MESSAGES = 6


# =========================================================
# Conversation history
# =========================================================

messages = []


# =========================================================
# Compressed memory of older conversation
# =========================================================

compressed_history = ""


# =========================================================
# Add a user message
# =========================================================

def add_user_message(content):

    messages.append(
        {
            "role": "user",
            "content": content
        }
    )


# =========================================================
# Add an assistant message
# =========================================================

def add_assistant_message(content):

    messages.append(
        {
            "role": "assistant",
            "content": content
        }
    )


# =========================================================
# Compress older messages
#
# Purpose:
# Convert older conversation into a compact state
# while preserving important information.
# =========================================================

def compress_old_messages(old_messages):

    conversation_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in old_messages
    )

    prompt = f"""
You are maintaining long-term context for a DataOps
investigation.

Compress the following old conversation.

Preserve:

- important facts
- decisions
- job IDs
- errors
- metrics
- conclusions
- unresolved questions
- important user requirements

Remove:

- greetings
- repetition
- irrelevant discussion
- unnecessary wording

Do not invent information.

Old conversation:

{conversation_text}

Return a concise structured summary.
"""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# =========================================================
# Build context for the current LLM call
# =========================================================

def build_context():

    context = []

    # -----------------------------------------------------
    # Add compressed history if available
    # -----------------------------------------------------

    if compressed_history:

        context.append(
            {
                "role": "system",
                "content": f"""
Important information from earlier conversation:

{compressed_history}
"""
            }
        )

    # -----------------------------------------------------
    # Add recent conversation
    # -----------------------------------------------------

    context.extend(messages)

    return context


# =========================================================
# Ask the LLM
# =========================================================

def ask_llm(question):

    global messages
    global compressed_history

    # -----------------------------------------------------
    # Add current user question
    # -----------------------------------------------------

    add_user_message(question)


    # -----------------------------------------------------
    # Check whether conversation is getting too large
    # -----------------------------------------------------

    if len(messages) > MAX_RECENT_MESSAGES:

        # Keep the newest messages.
        recent_messages = messages[
            -MAX_RECENT_MESSAGES:
        ]

        # Everything before recent messages
        # becomes old history.
        old_messages = messages[
            :-MAX_RECENT_MESSAGES
        ]


        print(
            "\n========== CONTEXT COMPRESSION =========="
        )

        print(
            "Old messages:",
            len(old_messages)
        )


        # -------------------------------------------------
        # Compress old history
        # -------------------------------------------------

        new_summary = compress_old_messages(
            old_messages
        )


        # -------------------------------------------------
        # If previous compressed history exists,
        # combine it with the new summary.
        # -------------------------------------------------

        if compressed_history:

            compressed_history = (
                compressed_history
                + "\n\n"
                + new_summary
            )

        else:

            compressed_history = new_summary


        # -------------------------------------------------
        # Replace full history with only recent messages.
        # -------------------------------------------------

        messages = recent_messages


    # -----------------------------------------------------
    # Build final context
    # -----------------------------------------------------

    context = build_context()


    # -----------------------------------------------------
    # Call LLM
    # -----------------------------------------------------

    response = ollama.chat(
        model=MODEL,
        messages=context
    )


    answer = response["message"]["content"]


    # -----------------------------------------------------
    # Store assistant response
    # -----------------------------------------------------

    add_assistant_message(answer)


    return answer


# =========================================================
# SIMULATED LONG CONVERSATION
# =========================================================

questions = [

    "We are investigating Spark job CRM360-123.",

    "The job failed during the aggregation stage.",

    "The main error was ExecutorLostFailure.",

    "Executor 3 reached 96 percent memory utilization.",

    "The configured executor memory is 16 GB.",

    "The current dataset is 2.5 times larger than the previous run.",

    "Kafka does not show any errors.",

    "Airflow correctly detected the Spark failure.",

    "The database connection was successful.",

    "What should we investigate next?"
]


# =========================================================
# Run conversation
# =========================================================

for number, question in enumerate(
    questions,
    start=1
):

    print(
        f"\n\n========== TURN {number} =========="
    )

    print(
        "USER:",
        question
    )


    answer = ask_llm(question)


    print(
        "\nASSISTANT:",
        answer
    )


    # -----------------------------------------------------
    # Show current context state
    # -----------------------------------------------------

    print(
        "\nRecent messages:",
        len(messages)
    )

    print(
        "Compressed history exists:",
        bool(compressed_history)
    )