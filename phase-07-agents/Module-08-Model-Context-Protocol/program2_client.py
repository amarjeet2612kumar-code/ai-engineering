# Import asyncio for asynchronous MCP communication.
import asyncio

# Import Path for reliable server-file resolution.
from pathlib import Path

# Import MCP client session and server parameters.
from mcp import ClientSession, StdioServerParameters

# Import the stdio transport helper.
from mcp.client.stdio import stdio_client


# Get the directory containing this client file.
BASE_DIR = Path(__file__).resolve().parent

# Build the absolute path to the MCP server.
SERVER_FILE = BASE_DIR / "mcp_server.py"


# Configure how the client starts the local MCP server.
server_params = StdioServerParameters(
    command="python",
    args=[str(SERVER_FILE)],
)


# Create the asynchronous client workflow.
async def main():

    # Start the MCP server as a local subprocess.
    async with stdio_client(server_params) as (read_stream, write_stream):

        # Create an MCP session over the transport.
        async with ClientSession(read_stream, write_stream) as session:

            # Initialize the MCP connection.
            await session.initialize()

            # Discover MCP tools.
            tools = await session.list_tools()

            # Display tool information.
            print("=== TOOLS ===")

            # Display each tool's name and input schema.
            for tool in tools.tools:
                print(f"Name: {tool.name}")
                print(f"Description: {tool.description}")
                print(f"Input Schema: {tool.inputSchema}")
                print()

            # Call the MCP Tool using the Pydantic-defined input structure.
            tool_result = await session.call_tool(
                "get_job_status",
                {
                    "request": {
                        "job_id": 101
                    }
                },
            )

            # Display the structured tool result.
            print("=== TOOL RESULT ===")
            print(tool_result)

            # Discover MCP resources.
            resources = await session.list_resources()

            # Display discovered resources.
            print("\n=== RESOURCES ===")

            # Print each resource URI.
            for resource in resources.resources:
                print(f"- {resource.uri}")

            # Read the Spark job logs resource.
            log_result = await session.read_resource(
                "spark://jobs/101/logs"
            )

            # Display the logs.
            print("\n=== JOB LOG RESOURCE ===")
            print(log_result)

            # Read the cluster resource.
            cluster_result = await session.read_resource(
                "spark://cluster/status"
            )

            # Display cluster information.
            print("\n=== CLUSTER RESOURCE ===")
            print(cluster_result)

            # Discover MCP prompts.
            prompts = await session.list_prompts()

            # Display discovered prompts.
            print("\n=== PROMPTS ===")

            # Print each prompt name.
            for prompt in prompts.prompts:
                print(f"- {prompt.name}")

            # Request the failure-analysis prompt.
            prompt_result = await session.get_prompt(
                "analyze_spark_failure",
                {
                    "job_id": "101"
                },
            )

            # Display the prompt.
            print("\n=== PROMPT RESULT ===")
            print(prompt_result)


# Run the asynchronous client.
if __name__ == "__main__":
    asyncio.run(main())