import asyncio
import os
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# Get the directory containing this client.
BASE_DIR = Path(__file__).resolve().parent


# Locate the secure MCP server.
SERVER_FILE = BASE_DIR / "program5_secure_mcp_server.py"


# Define the role used for this demonstration.
USER_ROLE = os.getenv(
    "DATAOPS_ROLE",
    "operator",
)


# Pass the authenticated role to the MCP server process.
SERVER_PARAMS = StdioServerParameters(
    command="python",
    args=[str(SERVER_FILE)],
    env={
        **os.environ,
        "DATAOPS_ROLE": USER_ROLE,
    },
)


# Run the secure MCP workflow.
async def main():

    # Start the MCP server.
    async with stdio_client(
        SERVER_PARAMS
    ) as (read, write):

        # Create the MCP client session.
        async with ClientSession(
            read,
            write,
        ) as session:

            # Initialize the MCP connection.
            await session.initialize()

            # Discover available MCP tools.
            response = await session.list_tools()

            # Display the available tools.
            print("\n=== AVAILABLE TOOLS ===")

            for tool in response.tools:
                print(
                    f"- {tool.name}: "
                    f"{tool.description}"
                )

            # Display the current user role.
            print("\n=== USER ROLE ===")
            print(USER_ROLE)

            # Read the status of Spark job 101.
            print("\n=== GET JOB STATUS ===")

            status_result = await session.call_tool(
                "get_job_status",
                {
                    "request": {
                        "job_id": 101
                    }
                },
            )

            # Display the status result.
            print(status_result)

            # Ask the human for approval before the dangerous action.
            print("\n=== HUMAN APPROVAL REQUIRED ===")

            approval = input(
                "Approve restart of job 101? (yes/no): "
            )

            # Stop when the human rejects the operation.
            if approval.lower() != "yes":

                print(
                    "\nRestart cancelled by human."
                )

                return

            # Execute the restart after explicit approval.
            print("\n=== RESTART JOB ===")

            restart_result = await session.call_tool(
                "restart_job",
                {
                    "request": {
                        "job_id": 101
                    }
                },
            )

            # Display the restart result.
            print(restart_result)


# Run the asynchronous client.
if __name__ == "__main__":
    asyncio.run(main())