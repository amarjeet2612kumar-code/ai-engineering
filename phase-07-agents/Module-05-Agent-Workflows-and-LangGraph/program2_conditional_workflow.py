from typing import TypedDict  # Import TypedDict to define the workflow state schema.

from langgraph.graph import StateGraph, START, END  # Import LangGraph components for building the graph.


class JobState(TypedDict):  # Define the structure of the workflow state.
    job_id: int  # Store the job ID.
    job_status: str  # Store the current job status.
    message: str  # Store the workflow result message.


def check_job(state: JobState) -> JobState:  # Define the node that checks the job.
    print("Checking job...")  # Display the current workflow step.

    state["job_status"] = "FAILED"  # Simulate a failed job.

    return state  # Return the updated state.


def route_job(state: JobState) -> str:  # Define the routing function that chooses the next node.
    if state["job_status"] == "FAILED":  # Check whether the job failed.
        return "investigate"  # Route failed jobs to the investigation node.

    return "finish"  # Route all other jobs to the finish node.


def investigate_job(state: JobState) -> JobState:  # Define the failure investigation node.
    print("Investigating failed job...")  # Display the current workflow step.

    state["message"] = "Failure investigation started"  # Store the investigation result.

    return state  # Return the updated state.


def finish_job(state: JobState) -> JobState:  # Define the successful completion node.
    print("Finishing job...")  # Display the current workflow step.

    state["message"] = "Job completed successfully"  # Store the completion result.

    return state  # Return the updated state.


builder = StateGraph(JobState)  # Create a LangGraph builder using the JobState schema.

builder.add_node("check_job", check_job)  # Register the job checking node.

builder.add_node("investigate", investigate_job)  # Register the failure investigation node.

builder.add_node("finish", finish_job)  # Register the successful completion node.

builder.add_edge(START, "check_job")  # Connect workflow start to the job checking node.

builder.add_conditional_edges(  # Add conditional routing after the job checking node.
    "check_job",  # Specify the node from which routing should happen.
    route_job,  # Specify the function that decides the next node.
    {  # Define the allowed routing destinations.
        "investigate": "investigate",  # Map the investigate decision to the investigate node.
        "finish": "finish",  # Map the finish decision to the finish node.
    },
)

builder.add_edge("investigate", END)  # Connect the investigation node to workflow completion.

builder.add_edge("finish", END)  # Connect the finish node to workflow completion.

graph = builder.compile()  # Compile the graph into an executable workflow.


initial_state: JobState = {  # Create the initial workflow state.
    "job_id": 101,  # Set the job ID.
    "job_status": "UNKNOWN",  # Set the initial job status.
    "message": "",  # Start with an empty message.
}


result = graph.invoke(initial_state)  # Execute the workflow with the initial state.

print("Final State:", result)  # Display the final state after workflow execution.