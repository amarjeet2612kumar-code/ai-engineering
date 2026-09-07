from typing import TypedDict  # Import TypedDict to define the workflow state schema.

from langgraph.graph import StateGraph, START, END  # Import LangGraph components for building the graph.


class JobState(TypedDict):  # Define the structure of the workflow state.
    job_id: int  # Store the job ID.
    job_name: str  # Store the job name.
    status: str  # Store the current job status.


def read_job(state: JobState) -> JobState:  # Define the first workflow node.
    print("Reading job...")  # Display the current workflow step.

    state["status"] = "READ"  # Update the job status in the workflow state.

    return state  # Return the updated state to the workflow.


def process_job(state: JobState) -> JobState:  # Define the second workflow node.
    print("Processing job...")  # Display the current workflow step.

    state["status"] = "PROCESSED"  # Update the job status after processing.

    return state  # Return the updated state to the workflow.


builder = StateGraph(JobState)  # Create a LangGraph builder using our JobState schema.

builder.add_node("read_job", read_job)  # Register the read_job function as a workflow node.

builder.add_node("process_job", process_job)  # Register the process_job function as a workflow node.

builder.add_edge(START, "read_job")  # Connect the workflow start to the read_job node.

builder.add_edge("read_job", "process_job")  # Connect read_job to process_job.

builder.add_edge("process_job", END)  # Connect process_job to workflow completion.

graph = builder.compile()  # Compile the graph into an executable workflow.


initial_state: JobState = {  # Create the initial workflow state.
    "job_id": 101,  # Set the job ID.
    "job_name": "customer_etl",  # Set the job name.
    "status": "NEW",  # Set the initial status.
}


result = graph.invoke(initial_state)  # Execute the LangGraph workflow with the initial state.

print("Final State:", result)  # Display the final state after workflow execution.