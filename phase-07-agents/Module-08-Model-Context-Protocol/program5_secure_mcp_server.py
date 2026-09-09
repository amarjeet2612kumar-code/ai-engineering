import os
import json
from datetime import datetime

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field


# Create the secure MCP server.
mcp = FastMCP("Secure DataOps MCP Server")


# Read the authenticated role from the server environment.
USER_ROLE = os.getenv("DATAOPS_ROLE", "viewer")


# Define jobs available in the simulated DataOps system.
JOBS = {
    101: {
        "job_id": 101,
        "job_name": "daily_customer_etl",
        "status": "FAILED",
        "error": "ExecutorOutOfMemory",
    },
    102: {
        "job_id": 102,
        "job_name": "customer_dimension_load",
        "status": "RUNNING",
        "error": None,
    },
}


# Define the input schema for job operations.
class JobRequest(BaseModel):
    job_id: int = Field(
        gt=0,
        description="Positive Spark job ID",
    )


# Define which roles are allowed to call each tool.
TOOL_PERMISSIONS = {
    "get_job_status": {
        "viewer",
        "operator",
        "admin",
    },
    "restart_job": {
        "operator",
        "admin",
    },
}


# Write security-related actions to an audit log.
def audit_log(
    tool_name: str,
    job_id: int,
    action: str,
):

    # Create a timestamp for the audit event.
    timestamp = datetime.now().isoformat()

    # Build the audit event.
    event = {
        "timestamp": timestamp,
        "role": USER_ROLE,
        "tool": tool_name,
        "job_id": job_id,
        "action": action,
    }

    # Write the event to the local audit file.
    with open("program5_audit.log", "a") as file:
        file.write(
            json.dumps(event) + "\n"
        )


# Check whether the current role is allowed to use a tool.
def authorize(tool_name: str):

    # Get the roles allowed for this tool.
    allowed_roles = TOOL_PERMISSIONS.get(
        tool_name,
        set(),
    )

    # Check whether the current role has permission.
    if USER_ROLE not in allowed_roles:

        # Record the denied authorization attempt.
        audit_log(
            tool_name,
            0,
            "AUTHORIZATION_DENIED",
        )

        # Stop execution immediately.
        raise PermissionError(
            f"Role '{USER_ROLE}' is not authorized "
            f"to use '{tool_name}'."
        )


# Expose the read-only job status tool.
@mcp.tool()
def get_job_status(
    request: JobRequest,
) -> dict:

    # Check authorization before accessing job data.
    authorize("get_job_status")

    # Extract the validated job ID.
    job_id = request.job_id

    # Check whether the job exists.
    if job_id not in JOBS:

        # Record the failed lookup.
        audit_log(
            "get_job_status",
            job_id,
            "JOB_NOT_FOUND",
        )

        # Return a safe error.
        return {
            "error": f"Job {job_id} not found."
        }

    # Record the successful read operation.
    audit_log(
        "get_job_status",
        job_id,
        "READ",
    )

    # Return the job information.
    return JOBS[job_id]


# Expose the sensitive restart operation.
@mcp.tool()
def restart_job(
    request: JobRequest,
) -> dict:

    # Check authorization before performing the action.
    authorize("restart_job")

    # Extract the validated job ID.
    job_id = request.job_id

    # Check whether the job exists.
    if job_id not in JOBS:

        # Record the failed operation.
        audit_log(
            "restart_job",
            job_id,
            "JOB_NOT_FOUND",
        )

        # Return a safe error.
        return {
            "error": f"Job {job_id} not found."
        }

    # Prevent restarting an already running job.
    if JOBS[job_id]["status"] == "RUNNING":

        # Record the rejected operation.
        audit_log(
            "restart_job",
            job_id,
            "ALREADY_RUNNING",
        )

        # Return the current state.
        return {
            "job_id": job_id,
            "status": "ALREADY_RUNNING",
        }

    # Simulate the restart operation.
    JOBS[job_id]["status"] = "RUNNING"

    # Clear the previous failure after the simulated restart.
    JOBS[job_id]["error"] = None

    # Record the successful action.
    audit_log(
        "restart_job",
        job_id,
        "RESTART_EXECUTED",
    )

    # Return the operation result.
    return {
        "job_id": job_id,
        "status": "RESTARTED",
        "message": f"Job {job_id} restarted successfully.",
    }


# Start the MCP server.
if __name__ == "__main__":

    # Run the server using stdio transport.
    mcp.run()