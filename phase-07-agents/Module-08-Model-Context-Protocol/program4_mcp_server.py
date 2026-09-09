import requests

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field


# Create the MCP server.
mcp = FastMCP("DataOps REST MCP Server")


# Define the input structure for the MCP tool.
class JobRequest(BaseModel):
    job_id: int = Field(
        gt=0,
        description="Positive Spark job ID",
    )


# Expose the REST-backed capability as an MCP tool.
@mcp.tool()
def get_job_status(request: JobRequest) -> dict:

    # Build the REST API endpoint for the requested job.
    url = (
        f"http://127.0.0.1:8000/jobs/"
        f"{request.job_id}"
    )

    # Call the existing REST API.
    response = requests.get(
        url,
        timeout=5,
    )

    # Raise an error when the REST API returns 4xx/5xx.
    response.raise_for_status()

    # Return the REST API JSON response to the MCP client.
    return response.json()


# Start the MCP server when this file is executed directly.
if __name__ == "__main__":

    # Run MCP using stdio transport.
    mcp.run()