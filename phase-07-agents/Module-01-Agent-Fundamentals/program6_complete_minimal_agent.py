# ============================================================
# Program 6: Complete Minimal Agent
# Module 1: Agent Fundamentals
# ============================================================
#
# Purpose:
# Build a complete minimal AI Agent.
#
# Concepts demonstrated:
#
# 1. Goal
# 2. State
# 3. Observation
# 4. Action
# 5. Agent Loop
# 6. LLM Decision
# 7. Environment Interaction
# 8. State Update
# 9. Completion
#
# Technology:
#
# Gemini = LLM / Decision Maker
# MySQL  = Environment / Data Source
# Python = Agent Controller
# ============================================================


# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

# os is used to read environment variables.
import os


# time is used to wait briefly between LLM requests
# if necessary.
import time


# Load values from the .env file.
from dotenv import load_dotenv


# Gemini SDK.
from google import genai


# MySQL connector.
import mysql.connector


# ------------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# ------------------------------------------------------------

# Load the .env file.
load_dotenv()


# Read the Gemini API key.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# Make sure the API key exists.
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

# Create Gemini client.
client = genai.Client(
    api_key=GEMINI_API_KEY
)


# Model used for our practical.
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


# Database used in our agent exercises.
MYSQL_DATABASE = "agent_demo"


# ============================================================
# LLM FUNCTION
# ============================================================

def ask_llm(prompt):

    # Send the prompt to Gemini.
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )


    # Return the generated text.
    return response.text.strip()


# ============================================================
# DATABASE TOOL
# ============================================================

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
    # The application controls this query.
    # The LLM does not generate SQL.
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


    # If the order does not exist,
    # return None.
    if result is None:

        return None


    # Return the status.
    return result[0]


# ============================================================
# AGENT DECISION FUNCTION
# ============================================================

def decide_next_action(state):

    # Create a prompt containing the current agent state.
    #
    # The LLM uses this information to decide
    # what action should happen next.
    prompt = f"""
You are an AI agent.

Your goal:
{state["goal"]}

Current state:
{state}

Available actions:

QUERY_DATABASE
CONTACT_SUPPORT
FINISH

Rules:

- If you need order information, choose QUERY_DATABASE.
- If the order requires customer support, choose CONTACT_SUPPORT.
- If the goal is completed, choose FINISH.

Return ONLY one action name.
"""


    # Ask Gemini for the next action.
    decision = ask_llm(prompt)


    # Remove unnecessary spaces.
    decision = decision.strip()


    # Convert to uppercase.
    decision = decision.upper()


    # Return the selected action.
    return decision


# ============================================================
# ACTION EXECUTION
# ============================================================

def execute_action(action, order_id, state):

    # --------------------------------------------------------
    # ACTION 1: QUERY_DATABASE
    # --------------------------------------------------------

    if action == "QUERY_DATABASE":

        print("Executing: QUERY_DATABASE")


        # Query MySQL.
        status = get_order_status(order_id)


        # Convert database result into an observation.
        if status is None:

            observation = (
                f"Order {order_id} was not found."
            )

        else:

            observation = (
                f"Order {order_id} status is {status}."
            )


        # Return the observation.
        return observation


    # --------------------------------------------------------
    # ACTION 2: CONTACT_SUPPORT
    # --------------------------------------------------------

    elif action == "CONTACT_SUPPORT":

        print("Executing: CONTACT_SUPPORT")


        # In this learning example we don't actually
        # contact a support system.
        #
        # We simply simulate the action.
        observation = (
            f"Customer support action recorded "
            f"for order {order_id}."
        )


        # Return the observation.
        return observation


    # --------------------------------------------------------
    # ACTION 3: FINISH
    # --------------------------------------------------------

    elif action == "FINISH":

        print("Executing: FINISH")


        # FINISH does not require an external system.
        observation = "Task completed."


        # Return the observation.
        return observation


    # --------------------------------------------------------
    # UNKNOWN ACTION
    # --------------------------------------------------------

    else:

        # If the LLM returns something we don't recognize,
        # return an error observation.
        return (
            f"Unknown action returned by LLM: {action}"
        )


# ============================================================
# COMPLETE AGENT LOOP
# ============================================================

def run_agent(goal, order_id):

    print("\n")
    print("=" * 60)
    print("COMPLETE MINIMAL AGENT")
    print("=" * 60)


    # --------------------------------------------------------
    # INITIAL STATE
    # --------------------------------------------------------

    # State represents everything the agent currently knows.
    state = {

        "goal": goal,

        "action": None,

        "observation": None,

        "status": "STARTED"
    }


    # Display initial state.
    print("\nInitial State:")
    print(state)


    # --------------------------------------------------------
    # AGENT LOOP
    # --------------------------------------------------------
    #
    # The agent repeatedly:
    #
    # 1. Looks at state
    # 2. Decides an action
    # 3. Executes the action
    # 4. Receives an observation
    # 5. Updates state
    #
    # This is the core Agent Loop.
    #

    max_iterations = 5


    for iteration in range(1, max_iterations + 1):

        print("\n")
        print(
            f"--- Agent Iteration {iteration} ---"
        )


        # ----------------------------------------------------
        # STEP 1: DECIDE
        # ----------------------------------------------------

        action = decide_next_action(state)


        # Display the decision.
        print(
            f"Decision: {action}"
        )


        # ----------------------------------------------------
        # Validate the action
        # ----------------------------------------------------

        allowed_actions = [
            "QUERY_DATABASE",
            "CONTACT_SUPPORT",
            "FINISH"
        ]


        # Check whether the LLM selected
        # an action that our application supports.
        if action not in allowed_actions:

            print(
                "Invalid action returned by LLM."
            )


            # Stop the agent safely.
            state["status"] = "FAILED"


            break


        # ----------------------------------------------------
        # STEP 2: STORE ACTION
        # ----------------------------------------------------

        # Save the selected action in state.
        state["action"] = action


        # ----------------------------------------------------
        # STEP 3: FINISH CHECK
        # ----------------------------------------------------

        # If the agent decides to finish,
        # there is no need for another action.
        if action == "FINISH":

            # Mark the agent as completed.
            state["status"] = "COMPLETED"


            # Store the final observation.
            state["observation"] = (
                "Agent decided that the task is complete."
            )


            print(
                "Agent decided to finish."
            )


            # Exit the loop.
            break


        # ----------------------------------------------------
        # STEP 4: EXECUTE ACTION
        # ----------------------------------------------------

        # Execute the selected action.
        observation = execute_action(
            action,
            order_id,
            state
        )


        # ----------------------------------------------------
        # STEP 5: UPDATE OBSERVATION
        # ----------------------------------------------------

        # Store the result from the environment.
        state["observation"] = observation


        # Display the observation.
        print(
            f"Observation: {observation}"
        )


        # ----------------------------------------------------
        # STEP 6: DISPLAY UPDATED STATE
        # ----------------------------------------------------

        print(
            f"Updated State: {state}"
        )


        # ----------------------------------------------------
        # Small delay
        # ----------------------------------------------------
        #
        # This is only for easier reading of the output.
        #

        time.sleep(0.5)


    else:

        # This block runs if the loop reaches
        # max_iterations without using break.
        state["status"] = "MAX_ITERATIONS"


        print(
            "\nAgent stopped because maximum "
            "iterations were reached."
        )


    # --------------------------------------------------------
    # FINAL STATE
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("FINAL STATE")
    print("=" * 60)


    print(state)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # Goal given to the agent.
    goal = (
        "Investigate order 3 and determine "
        "whether any action is required."
    )


    # Order we want the agent to investigate.
    order_id = 3


    # Start the agent.
    run_agent(
        goal,
        order_id
    )