from typing import TypedDict  # Import TypedDict to define the workflow state schema.

from langchain_ollama import ChatOllama  # Import the local Ollama chat model.

from langchain_core.tools import tool  # Import the decorator for creating tools.

from langgraph.graph import StateGraph, START, END  # Import LangGraph graph components.

from langgraph.checkpoint.memory import MemorySaver  # Import an in-memory checkpoint implementation.


class DataOpsState(TypedDict):  # Define the complete DataOps workflow state.
    job_id: int  # Store the failed job ID.
    logs: str  # Store the job logs.
    evidence: str  # Store additional investigation evidence.
    root_cause: str  # Store the identified root cause.
    recommendation: str  # Store the recommended remediation.
    approval_status: str  # Store the human approval decision.
    remediation_result: str  # Store the remediation result.
    iteration: int  # Store the number of investigation attempts.


@tool  # Convert the function into a tool available to the agent.
def get_job_metrics(job_id: int) -> str:  # Define a tool for retrieving job metrics.
    """Get metrics for a data engineering job."""  # Describe the purpose of the tool.

    metrics = {  # Create simulated job metrics.
        101: "Executor memory usage: 98%, Executor count: 4",  # Store metrics for job 101.
        102: "Executor memory usage: 45%, Executor count: 4",  # Store metrics for job 102.
    }

    return metrics.get(job_id, "Metrics not found")  # Return metrics for the requested job.


llm = ChatOllama(  # Create the local Ollama model.
    model="llama3.2:3b",  # Use the lightweight local model.
    temperature=0,  # Make responses more deterministic.
)


def collect_logs(state: DataOpsState) -> DataOpsState:  # Define the log collection node.
    print("\nCollecting logs...")  # Display the current workflow step.

    state["logs"] = "Spark job failed with Executor OutOfMemory"  # Store simulated Spark logs.

    return state  # Return the updated state.


def analyze_failure(state: DataOpsState) -> DataOpsState:  # Define the failure analysis node.
    state["iteration"] += 1  # Increase the investigation attempt count.

    print(f"Analyzing failure - attempt {state['iteration']}...")  # Display the current investigation attempt.

    prompt = f"""Analyze this Spark job failure.

Logs:
{state["logs"]}

Additional evidence:
{state["evidence"]}

Identify the likely root cause.
If there is enough evidence, return the root cause.
If there is not enough evidence, say MORE_EVIDENCE_NEEDED.
"""  # Build the analysis prompt for the LLM.

    response = llm.invoke(prompt)  # Ask Ollama to analyze the failure.

    analysis = response.content.strip()  # Extract the LLM response.

    print("LLM Analysis:", analysis)  # Display the LLM's analysis.

    if "MORE_EVIDENCE_NEEDED" not in analysis.upper():  # Check whether the LLM identified a root cause.
        state["root_cause"] = analysis  # Store the identified root cause.

    return state  # Return the updated state.


def gather_evidence(state: DataOpsState) -> DataOpsState:  # Define the evidence gathering node.
    print("Gathering additional evidence...")  # Display the current workflow step.

    metrics = get_job_metrics.invoke({"job_id": state["job_id"]})  # Execute the job metrics tool.

    state["evidence"] = metrics  # Store the tool result in workflow state.

    return state  # Return the updated state.


def route_after_analysis(state: DataOpsState) -> str:  # Decide whether to investigate further or continue.
    if state["root_cause"]:  # Check whether a root cause has been identified.
        return "recommend"  # Continue to remediation recommendation.

    if state["iteration"] >= 3:  # Stop investigating after three attempts.
        return "recommend"  # Continue using the available information.

    return "gather_evidence"  # Gather more evidence and investigate again.


def recommend_fix(state: DataOpsState) -> DataOpsState:  # Define the remediation recommendation node.
    print("Generating remediation recommendation...")  # Display the current workflow step.

    prompt = f"""Recommend a safe remediation for this Spark failure.

Root cause:
{state["root_cause"]}

Evidence:
{state["evidence"]}
"""  # Build the remediation recommendation prompt.

    response = llm.invoke(prompt)  # Ask Ollama for a remediation recommendation.

    state["recommendation"] = response.content.strip()  # Store the recommendation in workflow state.

    print("Recommendation:", state["recommendation"])  # Display the recommendation.

    return state  # Return the updated state.


def request_approval(state: DataOpsState) -> DataOpsState:  # Define the human approval node.
    print("\nHuman Approval Required")  # Display that human approval is required.

    print(f"Recommended action: {state['recommendation']}")  # Display the proposed remediation.

    approval = input("Approve remediation? (yes/no): ").strip().lower()  # Ask the human for approval.

    state["approval_status"] = approval  # Store the human decision.

    return state  # Return the updated state.


def route_after_approval(state: DataOpsState) -> str:  # Decide whether remediation can execute.
    if state["approval_status"] == "yes":  # Check whether the human approved the action.
        return "execute"  # Route approved remediation to execution.

    return "finish"  # Stop the workflow when remediation is rejected.


def execute_remediation(state: DataOpsState) -> DataOpsState:  # Define the remediation execution node.
    print("\nExecuting remediation...")  # Display the current workflow step.

    state["remediation_result"] = "Spark executor memory configuration updated"  # Simulate remediation execution.

    return state  # Return the updated state.


def validate_remediation(state: DataOpsState) -> DataOpsState:  # Define the remediation validation node.
    print("Validating remediation...")  # Display the current workflow step.

    state["remediation_result"] += " and validation completed successfully"  # Simulate successful validation.

    return state  # Return the updated state.


builder = StateGraph(DataOpsState)  # Create the LangGraph workflow builder.

builder.add_node("collect_logs", collect_logs)  # Register the log collection node.

builder.add_node("analyze_failure", analyze_failure)  # Register the failure analysis node.

builder.add_node("gather_evidence", gather_evidence)  # Register the evidence gathering node.

builder.add_node("recommend", recommend_fix)  # Register the recommendation node.

builder.add_node("request_approval", request_approval)  # Register the human approval node.

builder.add_node("execute", execute_remediation)  # Register the remediation node.

builder.add_node("validate", validate_remediation)  # Register the validation node.

builder.add_edge(START, "collect_logs")  # Start the workflow with log collection.

builder.add_edge("collect_logs", "analyze_failure")  # Send collected logs to failure analysis.

builder.add_conditional_edges(  # Add conditional routing after failure analysis.
    "analyze_failure",  # Specify the analysis node as the routing source.
    route_after_analysis,  # Use the routing function to determine the next step.
    {  # Define the allowed routing destinations.
        "gather_evidence": "gather_evidence",  # Route to evidence gathering.
        "recommend": "recommend",  # Route to remediation recommendation.
    },
)

builder.add_edge("gather_evidence", "analyze_failure")  # Loop back to analysis after gathering evidence.

builder.add_edge("recommend", "request_approval")  # Send the recommendation to human approval.

builder.add_conditional_edges(  # Add conditional routing after human approval.
    "request_approval",  # Specify the approval node as the routing source.
    route_after_approval,  # Use the approval routing function.
    {  # Define the allowed approval destinations.
        "execute": "execute",  # Route approved remediation to execution.
        "finish": END,  # End the workflow when remediation is rejected.
    },
)

builder.add_edge("execute", "validate")  # Validate the remediation after execution.

builder.add_edge("validate", END)  # Finish the workflow after validation.

checkpointer = MemorySaver()  # Create an in-memory checkpoint store.

graph = builder.compile(checkpointer=checkpointer)  # Compile the workflow with checkpointing.


initial_state: DataOpsState = {  # Create the initial workflow state.
    "job_id": 101,  # Set the failed job ID.
    "logs": "",  # Start with empty logs.
    "evidence": "",  # Start without additional evidence.
    "root_cause": "",  # Start without a root cause.
    "recommendation": "",  # Start without a recommendation.
    "approval_status": "",  # Start without an approval decision.
    "remediation_result": "",  # Start without a remediation result.
    "iteration": 0,  # Start with zero investigation attempts.
}


config = {  # Create configuration for this workflow execution.
    "configurable": {  # Provide LangGraph runtime configuration.
        "thread_id": "dataops-job-101",  # Assign a unique workflow execution ID.
    }
}


result = graph.invoke(initial_state, config)  # Execute the complete DataOps workflow.

print("\n========== FINAL STATE ==========")  # Display a separator before the final state.

print(result)  # Display the complete workflow state.