# ============================================================
# Program 4: Agent vs Chain
# Module 1: Agent Fundamentals
# ============================================================
#
# Purpose:
# Understand the difference between:
#
# 1. A traditional Chain
# 2. An AI Agent
#
# We use the same business problem for both:
#
# "What is the status of order 2?"
#
# The database is MySQL.
# The LLM is Gemini.
# ============================================================


# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

# os is used to read environment variables.
import os


# load_dotenv loads values from our .env file.
from dotenv import load_dotenv


# Gemini SDK.
from google import genai


# MySQL connector allows Python to communicate with MySQL.
import mysql.connector


# ------------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# ------------------------------------------------------------

# Load the .env file.
load_dotenv()


# Read the Gemini API key.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# Make sure the Gemini API key exists.
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found in the .env file."
    )


# Read the MySQL password.
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")


# Make sure the MySQL password exists.
if MYSQL_PASSWORD is None:
    raise ValueError(
        "MYSQL_PASSWORD was not found in the .env file."
    )


# ------------------------------------------------------------
# GEMINI CONFIGURATION
# ------------------------------------------------------------

# Create the Gemini client.
client = genai.Client(
    api_key=GEMINI_API_KEY
)


# Gemini model used for this practical.
MODEL = "gemini-3.7-flash"


# ------------------------------------------------------------
# MYSQL CONFIGURATION
# ------------------------------------------------------------

# MySQL host.
MYSQL_HOST = os.getenv(
    "MYSQL_HOST",
    "localhost"
)


# MySQL username.
MYSQL_USER = os.getenv(
    "MYSQL_USER",
    "root"
)


# Database we created for our agent practicals.
MYSQL_DATABASE = "agent_demo"


# ------------------------------------------------------------
# GEMINI HELPER FUNCTION
# ------------------------------------------------------------

def ask_llm(prompt):

    # Send the prompt to Gemini.
    #
    # We use generate_content() because this is
    # a normal text-generation request.
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )


    # Return only the generated text.
    return response.text.strip()


# ------------------------------------------------------------
# MYSQL FUNCTION
# ------------------------------------------------------------

def get_order_status(order_id):

    # Create a connection to MySQL.
    connection = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE
    )


    # Create a cursor.
    #
    # Cursor is used to execute SQL queries.
    cursor = connection.cursor()


    # SQL query.
    #
    # IMPORTANT:
    # The LLM does NOT generate this SQL.
    # The application controls the query.
    query = """
        SELECT status
        FROM orders
        WHERE id = %s
    """


    # Execute the query.
    #
    # order_id is passed separately.
    # This is safer than building SQL using string concatenation.
    cursor.execute(
        query,
        (order_id,)
    )


    # Read the first result.
    result = cursor.fetchone()


    # Close the cursor.
    cursor.close()


    # Close the database connection.
    connection.close()


    # If the order doesn't exist,
    # return a meaningful result.
    if result is None:
        return "Order not found."


    # Return the status.
    return result[0]


# ============================================================
# PART 1 — TRADITIONAL CHAIN
# ============================================================

def run_chain(question, order_id):

    print("\n")
    print("=" * 60)
    print("TRADITIONAL CHAIN")
    print("=" * 60)


    # --------------------------------------------------------
    # Step 1
    # --------------------------------------------------------

    print("\nStep 1: Receive user question")

    print(f"Question: {question}")


    # --------------------------------------------------------
    # Step 2
    # --------------------------------------------------------
    #
    # This step is predefined.
    #
    # The developer already knows that the application
    # needs to query MySQL.
    #

    print("\nStep 2: Query MySQL")

    status = get_order_status(order_id)


    # --------------------------------------------------------
    # Step 3
    # --------------------------------------------------------
    #
    # This step is also predefined.
    #
    # After getting the database result,
    # we ask the LLM to generate a user-friendly answer.
    #

    print("\nStep 3: Generate final answer")


    # Create a prompt for Gemini.
    prompt = f"""
Answer the user's question using the database result.

User question:
{question}

Database result:
Order {order_id} status is {status}.

Give a short and clear answer.
"""


    # Ask Gemini to create the final response.
    answer = ask_llm(prompt)


    # Display the final answer.
    print(f"\nFinal Answer: {answer}")


# ============================================================
# PART 2 — AI AGENT
# ============================================================

def run_agent(question, order_id):

    print("\n")
    print("=" * 60)
    print("AI AGENT")
    print("=" * 60)


    # --------------------------------------------------------
    # Create initial agent state
    # --------------------------------------------------------

    # State stores the current information
    # available to the agent.
    state = {

        "goal": question,

        "action": None,

        "observation": None,

        "status": "STARTED"
    }


    # Display the initial state.
    print("\nInitial State:")
    print(state)


    # --------------------------------------------------------
    # Agent Decision
    # --------------------------------------------------------
    #
    # Unlike the Chain, the next action is decided
    # by the LLM.
    #

    prompt = f"""
You are an AI agent.

Your goal:
{state["goal"]}

Current observation:
{state["observation"]}

Available actions:

QUERY_DATABASE
FINISH

Decide what action should be taken.

If information is required from the database:
choose QUERY_DATABASE.

If the goal is already completed:
choose FINISH.

Return only the action name.
"""


    # Ask Gemini to decide the next action.
    decision = ask_llm(prompt)


    # Convert the decision to uppercase.
    decision = decision.upper()


    # Display the decision.
    print(f"\nAgent Decision: {decision}")


    # --------------------------------------------------------
    # Execute the selected action
    # --------------------------------------------------------

    if decision == "QUERY_DATABASE":

        # Store the action in the state.
        state["action"] = decision


        print("\nAgent Action: QUERY_DATABASE")


        # Execute the database action.
        status = get_order_status(order_id)


        # The database result becomes
        # the agent's observation.
        state["observation"] = (
            f"Order {order_id} status is {status}."
        )


        # Display the observation.
        print(
            f"Agent Observation: "
            f"{state['observation']}"
        )


        # ----------------------------------------------------
        # Generate final answer
        # ----------------------------------------------------
        #
        # Now the agent has the information it needed.
        #

        answer_prompt = f"""
Answer the user's question using the observation.

User question:
{question}

Observation:
{state["observation"]}

Give a short and clear answer.
"""


        # Generate the final answer.
        answer = ask_llm(answer_prompt)


        # Update the state.
        state["status"] = "COMPLETED"


        # Display the answer.
        print(f"\nFinal Answer: {answer}")


    elif decision == "FINISH":

        # Store the selected action.
        state["action"] = decision


        # Mark the task as completed.
        state["status"] = "COMPLETED"


        print("\nAgent decided to finish.")


    else:

        # If Gemini returns something unexpected,
        # stop the agent.
        state["status"] = "STOPPED"


        print(
            "\nUnexpected action returned by Gemini."
        )


    # --------------------------------------------------------
    # Final State
    # --------------------------------------------------------

    print("\nFinal State:")
    print(state)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # User's question.
    question = "What is the status of order 2?"


    # Order ID we want to investigate.
    order_id = 2


    # --------------------------------------------------------
    # Run the Chain
    # --------------------------------------------------------

    run_chain(
        question,
        order_id
    )


    # --------------------------------------------------------
    # Run the Agent
    # --------------------------------------------------------

    run_agent(
        question,
        order_id
    )