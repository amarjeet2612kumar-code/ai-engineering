import asyncio
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# Get the directory containing this Python file.
BASE_DIR = Path(__file__).resolve().parent


# Build the path to the MCP server.
SERVER_FILE = BASE_DIR / "program4_mcp_server.py"


# Define how the MCP server should be started.
SERVER_PARAMS = StdioServerParameters(
    command="python",
    args=[str(SERVER_FILE)],
)


# Run the MCP client.
async def main():

    # Start the MCP server and establish stdio communication.
    async with stdio_client(SERVER_PARAMS) as (read, write):

        # Create an MCP client session.
        async with ClientSession(read, write) as session:

            # Initialize the MCP connection.
            await session.initialize()

            # Discover tools exposed by the MCP server.
            response = await session.list_tools()

            # Display the available tools.
            print("\n=== MCP TOOLS ===")

            for tool in response.tools:
                print(f"- {tool.name}")

            # Call the MCP tool.
            result = await session.call_tool(
                "get_job_status",
                {
                    "request": {
                        "job_id": 101
                    }
                },
            )

            # Display the MCP result.
            print("\n=== MCP RESULT ===")
            print(result)


# Execute the client.
if __name__ == "__main__":
    asyncio.run(main())