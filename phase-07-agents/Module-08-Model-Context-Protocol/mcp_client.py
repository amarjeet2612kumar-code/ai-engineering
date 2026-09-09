# Import asyncio for asynchronous MCP communication.
import asyncio

# Import Path so we can construct the server file path reliably.
from pathlib import Path

# Import the MCP client session.
from mcp import ClientSession, StdioServerParameters

# Import the stdio transport helper.
from mcp.client.stdio import stdio_client


# Get the directory containing this client file.
BASE_DIR = Path(__file__).resolve().parent

# Build the absolute path to the MCP server.
SERVER_FILE = BASE_DIR / "mcp_server.py"


# Define how the MCP Client should start the local MCP Server.
server_params = StdioServerParameters(
    command="python",
    args=[str(SERVER_FILE)],
)


# Create the asynchronous client workflow.
async def main():

    # Start the MCP Server as a subprocess and communicate through stdio.
    async with stdio_client(server_params) as (read_stream, write_stream):

        # Create an MCP session over the transport.
        async with ClientSession(read_stream, write_stream) as session:

            # Initialize the MCP connection.
            await session.initialize()

            # Discover the tools exposed by the MCP Server.
            tools = await session.list_tools()

            # Display the discovered tools.
            print("Available Tools:")

            # Print each available tool name.
            for tool in tools.tools:
                print(f"- {tool.name}")

            # Call the MCP Tool with job ID 101.
            result = await session.call_tool(
                "get_job_status",
                {"job_id": 101},
            )

            # Display the returned tool result.
            print("\nTool Result:")
            print(result)


# Run the asynchronous client.
if __name__ == "__main__":
    asyncio.run(main())