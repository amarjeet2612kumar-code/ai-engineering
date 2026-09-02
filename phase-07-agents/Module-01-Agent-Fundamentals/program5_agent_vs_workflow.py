# ============================================================
# Program 5: Agent vs Workflow
# Module 1: Agent Fundamentals
# ============================================================
#
# Purpose:
# Understand the difference between:
#
# 1. A traditional workflow
# 2. An AI agent
#
# Business scenario:
#
# "Check order 2 and tell me what I should do
#  if it is not delivered."
#
# We will solve the same problem using both approaches.
#
# MySQL = Environment / Data source
# Gemini = Decision maker for the Agent
# ============================================================


# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

# os allows us to read environment variables.
import os


# Load values from the .env file.
from dotenv import load_dotenv


# Gemini SDK.
from google import genai


# MySQL connector.
# Used to communicate with MySQL.
import mysql.connector


# ------------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# ------------------------------------------------------------

# Load the .env file.
load_dotenv()


# Read the Gemini API key.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# Check that the API key exists.
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found in the .env file."
    )


# Read the MySQL password.
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")


# Check that the MySQL password exists.
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


# Gemini model.
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


# Database used for our agent exercises.
MYSQL_DATABASE = "agent_demo"


# ------------------------------------------------------------
# GEMINI HELPER
# ------------------------------------------------------------

def ask_llm(prompt):

    # Send the prompt to Gemini.
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )


    # Return the generated text.
    return response.text.strip()


# ------------------------------------------------------------
# MYSQL FUNCTION
# ------------------------------------------------------------

def get_order_status(order_id):

    # Connect to MySQL.
    connection = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE
    )


    # Create a cursor.
    cursor = connection.cursor()


    # SQL query.
    #
    # The SQL is controlled by the application.
    # The LLM does not generate this query.
    query = """
        SELECT status
        FROM orders
        WHERE id = %s
    """


    # Execute the query.
    cursor.execute(
        query,
        (order_id,)
    )


    # Read the first result.
    result = cursor.fetchone()


    # Close database resources.
    cursor.close()
    connection.close()


    # Handle an order that does not exist.
    if result is None:
        return None


    # Return the order status.
    return result[0]


# ============================================================
# PART 1 — TRADITIONAL WORKFLOW
# ============================================================

def run_workflow(order_id):

    print("\n")
    print("=" * 60)
    print("TRADITIONAL WORKFLOW")
    print("=" * 60)


    # --------------------------------------------------------
    # Step 1: Get the order
    # --------------------------------------------------------

    print("\nStep 1: Get order status")


    status = get_order_status(order_id)


    print(f"Order status: {status}")


    # --------------------------------------------------------
    # Step 2: Developer-defined decision
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # The workflow does NOT ask an LLM what to do.
    #
    # The developer has already defined the rule:
    #
    # If delivered -> tell customer it is delivered.
    #
    # Otherwise -> tell customer to contact support.
    #

    print("\nStep 2: Apply workflow rule")


    if status == "DELIVERED":

        # Fixed workflow branch.
        result = (
            f"Order {order_id} is delivered. "
            "No further action is required."
        )


    elif status is None:

        # Another predefined branch.
        result = (
            f"Order {order_id} was not found."
        )


    else:

        # Predefined rule for non-delivered orders.
        result = (
            f"Order {order_id} is {status}. "
            "Please contact customer support."
        )


    # --------------------------------------------------------
    # Step 3: Return result
    # --------------------------------------------------------

    print("\nStep 3: Return result")

    print(f"Workflow Result: {result}")


# ============================================================
# PART 2 — AI AGENT
# ============================================================

def run_agent(order_id):

    print("\n")
    print("=" * 60)
    print("AI AGENT")
    print("=" * 60)


    # --------------------------------------------------------
    # Initial agent state
    # --------------------------------------------------------

    # State stores what the agent currently knows.
    state = {

        "goal": (
            f"Check order {order_id} and determine "
            "what should be done if it is not delivered."
        ),

        "action": None,

        "observation": None,

        "status": "STARTED"
    }


    # Display the initial state.
    print("\nInitial State:")
    print(state)


    # --------------------------------------------------------
    # Agent decision
    # --------------------------------------------------------

    prompt = f"""
You are an AI agent.

Your goal:
{state["goal"]}

Current observation:
{state["observation"]}

Available actions:

QUERY_DATABASE
FINISH

Rules:

- If you need order information, choose QUERY_DATABASE.
- If the goal has already been completed, choose FINISH.

Return only the action name.
"""


    # Ask Gemini to decide the next action.
    decision = ask_llm(prompt)


    # Normalize the response.
    decision = decision.upper()


    # Display the decision.
    print(f"\nAgent Decision: {decision}")


    # --------------------------------------------------------
    # Execute the selected action
    # --------------------------------------------------------

    if decision == "QUERY_DATABASE":

        # Store the action in the state.
        state["action"] = decision


        print("\nExecuting action: QUERY_DATABASE")


        # Query MySQL.
        status = get_order_status(order_id)


        # Convert database result into
        # an agent observation.
        if status is None:

            state["observation"] = (
                f"Order {order_id} was not found."
            )

        else:

            state["observation"] = (
                f"Order {order_id} status is {status}."
            )


        # Display the observation.
        print(
            f"Observation: "
            f"{state['observation']}"
        )


        # ----------------------------------------------------
        # Second agent decision
        # ----------------------------------------------------
        #
        # Now the agent has new information.
        #
        # It can use the observation to decide
        # what to do next.
        #

        second_prompt = f"""
You are an AI agent.

Goal:
{state["goal"]}

Current observation:
{state["observation"]}

The database query has already been completed.

Decide what to do next.

Available actions:

FINISH
CONTACT_SUPPORT

If the order is delivered:
choose FINISH.

If the order is not delivered:
choose CONTACT_SUPPORT.

If the order does not exist:
choose FINISH.

Return only the action name.
"""


        # Ask Gemini for the next decision.
        second_decision = ask_llm(
            second_prompt
        )


        # Normalize the response.
        second_decision = (
            second_decision
            .upper()
        )


        # Display the second decision.
        print(
            f"\nAgent Decision 2: "
            f"{second_decision}"
        )


        # ----------------------------------------------------
        # Execute second action
        # ----------------------------------------------------

        if second_decision == "CONTACT_SUPPORT":

            # Store the action.
            state["action"] = second_decision


            # Create the observation/result.
            state["observation"] += (
                " Recommended action: "
                "Contact customer support."
            )


            print(
                "\nAgent Action: CONTACT_SUPPORT"
            )


        elif second_decision == "FINISH":

            # Store the action.
            state["action"] = second_decision


            print(
                "\nAgent Action: FINISH"
            )


        else:

            # Handle an unexpected response.
            print(
                "\nUnexpected second action."
            )


        # Mark the agent as completed.
        state["status"] = "COMPLETED"


    else:

        # Handle an unexpected first action.
        state["status"] = "STOPPED"


        print(
            "\nUnexpected action returned by Gemini."
        )


    # --------------------------------------------------------
    # Final state
    # --------------------------------------------------------

    print("\nFinal State:")
    print(state)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # Order that we want to investigate.
    order_id = 2


    # Run the traditional workflow.
    run_workflow(order_id)


    # Run the AI agent.
    run_agent(order_id)