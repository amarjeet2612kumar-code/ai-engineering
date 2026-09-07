from typing import TypedDict  # Import TypedDict to define the workflow state schema.

from langgraph.graph import StateGraph, START, END  # Import LangGraph components for building the workflow.


class InvestigationState(TypedDict):  # Define the structure of the investigation state.
    job_id: int  # Store the job ID.
    iteration: int  # Store the current investigation iteration.
    root_cause: str  # Store the discovered root cause.
    status: str  # Store the current investigation status.


MAX_ITERATIONS = 5  # Define the maximum number of investigation attempts.


def investigate(state: InvestigationState) -> InvestigationState:  # Define the investigation node.
    state["iteration"] += 1  # Increase the investigation iteration count.

    print(f"Investigation attempt: {state['iteration']}")  # Display the current attempt number.

    if state["iteration"] == 3:  # Simulate finding the root cause on the third attempt.
        state["root_cause"] = "Insufficient executor memory"  # Store the discovered root cause.
        state["status"] = "ROOT_CAUSE_FOUND"  # Update the investigation status.
        print("Root cause found!")  # Display that the investigation succeeded.
    else:  # Handle attempts where the root cause is not found.
        state["status"] = "INVESTIGATING"  # Keep the investigation active.
        print("Root cause not found yet.")  # Display that investigation must continue.

    return state  # Return the updated state.


def route_investigation(state: InvestigationState) -> str:  # Decide whether to continue or stop.
    if state["root_cause"]:  # Check whether a root cause has been discovered.
        return "finish"  # Stop the workflow when the root cause is found.

    if state["iteration"] >= MAX_ITERATIONS:  # Check whether the maximum iteration limit was reached.
        return "finish"  # Stop the workflow to prevent an infinite loop.

    return "investigate"  # Continue investigating when neither condition is met.


builder = StateGraph(InvestigationState)  # Create a LangGraph builder using the investigation state schema.

builder.add_node("investigate", investigate)  # Register the investigation node.

builder.add_edge(START, "investigate")  # Start the workflow with the investigation node.

builder.add_conditional_edges(  # Add conditional routing after each investigation attempt.
    "investigate",  # Specify the node from which routing should happen.
    route_investigation,  # Specify the function that decides whether to continue or finish.
    {  # Define the allowed routing destinations.
        "investigate": "investigate",  # Route back to the investigation node for another attempt.
        "finish": END,  # Route to END when investigation should stop.
    },
)

graph = builder.compile()  # Compile the graph into an executable workflow.


initial_state: InvestigationState = {  # Create the initial workflow state.
    "job_id": 101,  # Set the job ID.
    "iteration": 0,  # Start with zero investigation attempts.
    "root_cause": "",  # Start without a discovered root cause.
    "status": "STARTED",  # Set the initial investigation status.
}


result = graph.invoke(initial_state)  # Execute the workflow with the initial state.

print("Final State:", result)  # Display the final state after the loop completes.