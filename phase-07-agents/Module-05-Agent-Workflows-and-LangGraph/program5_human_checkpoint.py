from typing import TypedDict  # Import TypedDict to define the workflow state schema.

from langgraph.graph import StateGraph, START, END  # Import LangGraph components for building the workflow.

from langgraph.checkpoint.memory import MemorySaver  # Import an in-memory checkpointer for learning.


class DataOpsState(TypedDict):  # Define the structure of our DataOps workflow state.
    job_id: int  # Store the failed job ID.
    root_cause: str  # Store the identified root cause.
    recommendation: str  # Store the recommended remediation.
    approval_status: str  # Store the human approval decision.
    remediation_result: str  # Store the remediation result.


def analyze_failure(state: DataOpsState) -> DataOpsState:  # Define the failure analysis node.
    print("Analyzing failure...")  # Display the current workflow step.

    state["root_cause"] = "Insufficient executor memory"  # Store the simulated root cause.

    return state  # Return the updated state.


def recommend_fix(state: DataOpsState) -> DataOpsState:  # Define the remediation recommendation node.
    print("Generating remediation recommendation...")  # Display the current workflow step.

    state["recommendation"] = "Increase Spark executor memory"  # Store the recommended remediation.

    return state  # Return the updated state.


def request_approval(state: DataOpsState) -> DataOpsState:  # Define the human approval node.
    print("\nHuman Approval Required")  # Display that human input is required.

    print(f"Recommendation: {state['recommendation']}")  # Display the recommended remediation.

    approval = input("Approve remediation? (yes/no): ").strip().lower()  # Ask the human for an approval decision.

    state["approval_status"] = approval  # Store the human decision in workflow state.

    return state  # Return the updated state.


def route_after_approval(state: DataOpsState) -> str:  # Decide what happens after human approval.
    if state["approval_status"] == "yes":  # Check whether the human approved the remediation.
        return "execute"  # Route approved requests to remediation execution.

    return "finish"  # Stop the workflow when remediation is rejected.


def execute_remediation(state: DataOpsState) -> DataOpsState:  # Define the remediation execution node.
    print("Executing remediation...")  # Display the current workflow step.

    state["remediation_result"] = "Executor memory increased successfully"  # Store the simulated execution result.

    return state  # Return the updated state.


builder = StateGraph(DataOpsState)  # Create a LangGraph builder using our DataOps state schema.

builder.add_node("analyze_failure", analyze_failure)  # Register the failure analysis node.

builder.add_node("recommend_fix", recommend_fix)  # Register the recommendation node.

builder.add_node("request_approval", request_approval)  # Register the human approval node.

builder.add_node("execute", execute_remediation)  # Register the remediation execution node.

builder.add_edge(START, "analyze_failure")  # Start the workflow with failure analysis.

builder.add_edge("analyze_failure", "recommend_fix")  # Connect analysis to recommendation.

builder.add_edge("recommend_fix", "request_approval")  # Connect recommendation to human approval.

builder.add_conditional_edges(  # Add conditional routing after human approval.
    "request_approval",  # Specify the node from which routing happens.
    route_after_approval,  # Specify the function that decides the next step.
    {  # Define the allowed routing destinations.
        "execute": "execute",  # Route approved requests to remediation.
        "finish": END,  # Stop rejected requests.
    },
)

builder.add_edge("execute", END)  # Finish the workflow after remediation.

checkpointer = MemorySaver()  # Create an in-memory checkpoint store.

graph = builder.compile(checkpointer=checkpointer)  # Compile the graph with checkpoint persistence.


initial_state: DataOpsState = {  # Create the initial workflow state.
    "job_id": 101,  # Set the failed job ID.
    "root_cause": "",  # Start without a root cause.
    "recommendation": "",  # Start without a recommendation.
    "approval_status": "",  # Start without an approval decision.
    "remediation_result": "",  # Start without a remediation result.
}


config = {  # Create configuration identifying this workflow execution.
    "configurable": {  # Provide LangGraph runtime configuration.
        "thread_id": "job-101",  # Give this workflow execution a unique thread identifier.
    }
}


result = graph.invoke(initial_state, config)  # Execute the workflow using the checkpoint configuration.

print("\nFinal State:", result)  # Display the final workflow state.