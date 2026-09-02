# Import the os module.
# We use it to read environment variables.
import os


# Import load_dotenv.
# This loads variables from our .env file.
from dotenv import load_dotenv


# Import Google's Gemini SDK.
from google import genai


# Import MySQL connector.
# This allows Python to communicate with MySQL.
import mysql.connector


# -----------------------------------------
# Load environment variables
# -----------------------------------------

# Load the .env file.
load_dotenv()


# Read the Gemini API key.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# Check whether the Gemini API key exists.
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found in the .env file."
    )


# -----------------------------------------
# Gemini configuration
# -----------------------------------------

# Create the Gemini client.
client = genai.Client(
    api_key=GEMINI_API_KEY
)


# Gemini model.
MODEL = "gemini-3.7-flash"


# -----------------------------------------
# MySQL configuration
# -----------------------------------------

# These values describe how Python
# connects to our MySQL database.
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = "agent_demo"


# Check whether the MySQL password exists.
if MYSQL_PASSWORD is None:
    raise ValueError(
        "MYSQL_PASSWORD was not found in the .env file."
    )


# -----------------------------------------
# Ask Gemini
# -----------------------------------------

# def ask_llm(prompt):

#     # Send the prompt to Gemini.
#     response = client.interactions.create(
#         model=MODEL,
#         input=prompt
#     )

#     # Return only the generated text.
#     return response.output_text

def ask_llm(prompt):

    # Send the prompt to Gemini.
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    # Return the generated text.
    return response.text

# -----------------------------------------
# Execute database action
# -----------------------------------------

def query_order_status(order_id):

    # Connect to the MySQL database.
    connection = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE
    )


    # Create a cursor.
    # The cursor allows us to execute SQL.
    cursor = connection.cursor()


    # This is a predefined SQL query.
    #
    # We are intentionally NOT allowing
    # the LLM to generate SQL yet.
    query = """
        SELECT status
        FROM orders
        WHERE id = %s
    """


    # Execute the query.
    #
    # The tuple (order_id,) safely provides
    # the value for the %s placeholder.
    cursor.execute(
        query,
        (order_id,)
    )


    # Get the first row returned by MySQL.
    result = cursor.fetchone()


    # Close the cursor.
    cursor.close()


    # Close the database connection.
    connection.close()


    # If no order was found,
    # return an appropriate observation.
    if result is None:
        return f"Order {order_id} was not found."


    # Return the database result.
    return f"Order {order_id} status is {result[0]}."


# -----------------------------------------
# Agent
# -----------------------------------------

def run_agent(goal, order_id):

    # -----------------------------------------
    # Initial State
    # -----------------------------------------

    # State stores information about
    # the current agent execution.
    state = {
        "goal": goal,
        "action": None,
        "observation": None,
        "status": "STARTED"
    }


    # Display the initial state.
    print("\nInitial State:")
    print(state)


    # -----------------------------------------
    # Agent Loop
    # -----------------------------------------

    # Maximum number of iterations.
    # This prevents an infinite loop.
    max_iterations = 3


    for step in range(max_iterations):

        print(f"\n--- Iteration {step + 1} ---")


        # -----------------------------------------
        # Ask Gemini for the next action
        # -----------------------------------------

        prompt = f"""
You are a simple AI agent.

Goal:
{state["goal"]}

Current action:
{state["action"]}

Current observation:
{state["observation"]}

Available action:

QUERY_DATABASE

The action checks the status of order {order_id}
in the MySQL database.

Choose the next action.

If the database needs to be checked:
QUERY_DATABASE

Return ONLY:
QUERY_DATABASE
"""


        # Ask Gemini to choose an action.
        decision = ask_llm(prompt)


        # Remove extra spaces.
        # Convert the response to uppercase.
        decision = decision.strip().upper()


        # Display the LLM decision.
        print(f"LLM Decision: {decision}")


        # -----------------------------------------
        # Validate the action
        # -----------------------------------------

        if decision != "QUERY_DATABASE":

            print("Unexpected action from Gemini.")

            # Stop the agent instead of
            # continuing indefinitely.
            state["status"] = "STOPPED"

            break


        # Store the action in the state.
        state["action"] = decision


        # -----------------------------------------
        # Execute the Action
        # -----------------------------------------

        print("Executing action: QUERY_DATABASE")


        # Execute the database action.
        observation = query_order_status(order_id)


        # -----------------------------------------
        # Observation
        # -----------------------------------------

        # Store the database result
        # as the agent's observation.
        state["observation"] = observation


        # Display the observation.
        print(f"Observation: {observation}")


        # -----------------------------------------
        # Update State
        # -----------------------------------------

        # We received the required information,
        # so the goal is completed.
        state["status"] = "COMPLETED"


        # Display the updated state.
        print("Updated State:")
        print(state)


        # Stop the loop because the goal
        # has been completed.
        break


    # -----------------------------------------
    # Final State
    # -----------------------------------------

    print("\nFinal State:")
    print(state)


# -----------------------------------------
# Program Entry Point
# -----------------------------------------

if __name__ == "__main__":

    # Define the user's goal.
    goal = "Find the status of order 2."


    # The order ID we want to investigate.
    order_id = 2


    # Start the agent.
    run_agent(
        goal,
        order_id
    )