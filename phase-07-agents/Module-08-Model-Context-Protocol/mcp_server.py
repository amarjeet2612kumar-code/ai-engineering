# Import FastMCP to create and expose an MCP server.
from mcp.server.fastmcp import FastMCP

# Import Pydantic classes for input validation and structured output.
from pydantic import BaseModel, Field


# Create the MCP server application.
mcp = FastMCP("DataOps Server")


# ============================================================
# Pydantic Models
# ============================================================


# Define the input schema shared by job-related tools.
class JobRequest(BaseModel):
    # Require a positive Spark job ID.
    job_id: int = Field(gt=0, description="Positive Spark job ID")


# Define the structured response returned by the status tool.
class JobStatusResponse(BaseModel):
    # Store the requested job ID.
    job_id: int

    # Store the job name when the job exists.
    job_name: str | None = None

    # Store the current job status.
    status: str

    # Store the failure reason when available.
    error: str | None = None


# ============================================================
# Simulated DataOps Backend
# ============================================================


# Simulate Spark job information.
JOBS = {
    101: {
        "job_name": "daily_customer_etl",
        "status": "FAILED",
        "error": "ExecutorOutOfMemory",
        "logs": """
Executor memory exceeded.
Container killed by YARN for exceeding memory limits.
Spark job failed during aggregation stage.
""",
    },
    102: {
        "job_name": "customer_dimension",
        "status": "RUNNING",
        "error": None,
        "logs": """
Spark job is running successfully.
No errors detected.
""",
    },
}


# Simulate Spark cluster information.
CLUSTER_STATUS = {
    "cluster": "local-spark-cluster",
    "status": "HEALTHY",
    "active_workers": 2,
}


# ============================================================
# MCP TOOLS
# ============================================================


# Expose job status as an executable MCP Tool.
@mcp.tool()
def get_job_status(request: JobRequest) -> JobStatusResponse:
    """
    Get the current status of a Spark job.
    """

    # Look up the requested job.
    job = JOBS.get(request.job_id)

    # Return a structured response when the job does not exist.
    if job is None:
        return JobStatusResponse(
            job_id=request.job_id,
            status="NOT_FOUND",
        )

    # Return the structured job status.
    return JobStatusResponse(
        job_id=request.job_id,
        job_name=job["job_name"],
        status=job["status"],
        error=job["error"],
    )


# Expose job logs as an executable MCP Tool.
@mcp.tool()
def get_job_logs(request: JobRequest) -> str:
    """
    Retrieve the logs of a Spark job.
    """

    # Look up the requested job.
    job = JOBS.get(request.job_id)

    # Return a message when the job does not exist.
    if job is None:
        return f"Job {request.job_id} was not found."

    # Return the Spark job logs.
    return job["logs"]


# ============================================================
# MCP RESOURCES
# ============================================================


# Expose job logs as an MCP Resource Template.
@mcp.resource("spark://jobs/{job_id}/logs")
def spark_job_logs(job_id: int) -> str:
    """
    Provide Spark job logs as read-only context.
    """

    # Look up the requested job.
    job = JOBS.get(job_id)

    # Return a message when the job does not exist.
    if job is None:
        return f"Job {job_id} was not found."

    # Return the job logs as resource content.
    return job["logs"]


# Expose cluster information as an MCP Resource.
@mcp.resource("spark://cluster/status")
def spark_cluster_status() -> str:
    """
    Provide current Spark cluster status.
    """

    # Return cluster information as readable resource content.
    return (
        f"Cluster: {CLUSTER_STATUS['cluster']}\n"
        f"Status: {CLUSTER_STATUS['status']}\n"
        f"Active Workers: {CLUSTER_STATUS['active_workers']}"
    )


# ============================================================
# MCP PROMPTS
# ============================================================


# Expose a reusable Spark failure-analysis MCP Prompt.
@mcp.prompt()
def analyze_spark_failure(job_id: int) -> str:
    """
    Provide instructions for analyzing a Spark failure.
    """

    # Return reusable instructions for the LLM.
    return (
        f"Analyze Spark job {job_id} failure.\n"
        "Use the job status and logs as evidence.\n"
        "Identify the likely failure reason.\n"
        "Separate evidence from inference.\n"
        "Do not invent information.\n"
        "Provide a concise recommendation."
    )


# ============================================================
# SERVER STARTUP
# ============================================================


# Start the MCP server when this file is executed directly.
if __name__ == "__main__":
    mcp.run()